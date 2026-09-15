import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO      = Path(__file__).resolve().parent.parent
SEEDS     = [0, 1, 2, 3, 4]
N_TRIALS  = 50
OUT       = REPO / "results" / "dl_optuna_results.json"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

data = pd.read_csv(REPO / "data/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})

CN = ["Co","Cr","Cu","Fe","Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(REPO / "data/features_13.csv")
X = pd.concat([Xc, feat_df], axis=1)
X = X.loc[:, ~X.columns.duplicated()]
n_features = X.shape[1]
print(f"Features ({n_features}): {list(X.columns)}")

ALL_TARGETS = {
    "Youngs_modulus": "Young's modulus(Gpa)",
    "UTS": "UTS(Gpa)",
    "Disloc_0pct": "Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc_20pct": "Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC_0pct": "FCC 0%(%)", "HCP_0pct": "HCP 0%(%)",
    "BCC_0pct": "BCC 0%(%)", "FCC_20pct": "FCC 20%(%)",
    "HCP_20pct": "HCP 20%(%)", "BCC_20pct": "BCC 20%(%)",
    "Other_0pct": "Other 0%(%)", "Other_20pct": "Other 20%(%)",
}

n_val = max(1, int(len(X) * 0.15))
tr_idx = np.arange(len(X) - n_val)
va_idx = np.arange(len(X) - n_val, len(X))



class DeepMLP(nn.Module):
    def __init__(self, n_feat, h1, h2, h3, dropout):
        super().__init__()
        layers = [nn.Linear(n_feat, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
                  nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
                  nn.Linear(h2, h3), nn.ReLU(), nn.Linear(h3, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)


class TabularTransformer(nn.Module):
    def __init__(self, n_feat, d_model, nhead, num_layers):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_feat)])
        self.pos_enc = nn.Parameter(torch.randn(1, n_feat, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(n_feat*d_model, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        tokens = torch.stack([emb(x[:, i:i+1]) for i, emb in enumerate(self.embeddings)], dim=1)
        tokens = tokens + self.pos_enc
        return self.head(self.transformer(tokens).flatten(1)).squeeze(-1)


class AttentionMLP(nn.Module):
    def __init__(self, n_feat, embed_dim, nhead):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_feat)])
        self.attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(n_feat*embed_dim, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        tokens = torch.stack([p(x[:, i:i+1]) for i, p in enumerate(self.projections)], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return self.head(tokens.flatten(1)).squeeze(-1)


def train_model(model, X_tr, y_tr, X_va, y_va, lr, wd, epochs=500, patience=60):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    crit = nn.MSELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
    yv = torch.tensor(y_va, dtype=torch.float32).to(device)

    best_loss, wait, best_state = float('inf'), 0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        crit(model(Xt), yt).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        sched.step(vl)
        if vl < best_loss:
            best_loss, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def train_fullfit(model, X_all, y_all, lr, wd, epochs=500):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    Xt = torch.tensor(X_all, dtype=torch.float32).to(device)
    yt = torch.tensor(y_all, dtype=torch.float32).to(device)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        crit(model(Xt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xt).cpu().numpy()
    return float(r2_score(y_all, pred))


def build_model(mname, params, n_feat):
    if mname == "DeepMLP":
        return DeepMLP(n_feat, params["h1"], params["h2"], params["h3"], params["dropout"])
    elif mname == "TabularTransformer":
        return TabularTransformer(n_feat, params["d_model"], params["nhead"], params["num_layers"])
    elif mname == "AttentionMLP":
        return AttentionMLP(n_feat, params["embed_dim"], params["nhead"])


def make_objective(mname, X_data, y_data, n_feat):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_data.iloc[tr_idx].values)
    X_va = scaler.transform(X_data.iloc[va_idx].values)
    y_tr = y_data[tr_idx]
    y_va = y_data[va_idx]

    def objective(trial):
        if mname == "DeepMLP":
            params = {
                "h1": trial.suggest_categorical("h1", [64, 128, 256, 512]),
                "h2": trial.suggest_categorical("h2", [32, 64, 128, 256]),
                "h3": trial.suggest_categorical("h3", [16, 32, 64]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "wd": trial.suggest_float("wd", 1e-5, 1e-2, log=True),
            }
        elif mname == "TabularTransformer":
            d_model = trial.suggest_categorical("d_model", [16, 32, 64])
            nhead = trial.suggest_categorical("nhead", [2, 4])
            if d_model % nhead != 0:
                raise optuna.exceptions.TrialPruned()
            params = {
                "d_model": d_model, "nhead": nhead,
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "wd": trial.suggest_float("wd", 1e-5, 1e-2, log=True),
            }
        elif mname == "AttentionMLP":
            embed_dim = trial.suggest_categorical("embed_dim", [16, 24, 32, 48])
            nhead = trial.suggest_categorical("nhead", [2, 4])
            if embed_dim % nhead != 0:
                raise optuna.exceptions.TrialPruned()
            params = {
                "embed_dim": embed_dim, "nhead": nhead,
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "wd": trial.suggest_float("wd", 1e-5, 1e-2, log=True),
            }

        torch.manual_seed(42)
        model = build_model(mname, params, n_feat)
        model = train_model(model, X_tr, y_tr, X_va, y_va, params["lr"], params["wd"])
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_va, dtype=torch.float32).to(device)).cpu().numpy()
        return float(r2_score(y_va, pred))
    return objective


DL_MODELS = ["DeepMLP", "TabularTransformer", "AttentionMLP"]
results = {}
total = len(ALL_TARGETS) * len(DL_MODELS)
rc = 0

for tkey, tcol in ALL_TARGETS.items():
    y = data[tcol].values.astype(np.float64)
    results[tkey] = {}

    for mname in DL_MODELS:
        rc += 1
        print(f"[{rc}/{total}] {tkey} {mname} — optuna {N_TRIALS} trials...", flush=True)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(mname, X, y, n_features),
                       n_trials=N_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        best_val_r2 = study.best_value
        print(f"  best val R²={best_val_r2:.4f}, params={best_params}")

        scaler = StandardScaler()
        X_all = scaler.fit_transform(X.values)

        seed_r2s = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model(mname, best_params, n_features)
            train_r2 = train_fullfit(model, X_all, y, best_params["lr"], best_params["wd"])
            seed_r2s.append(train_r2)

        results[tkey][mname] = {
            "mean_train_r2": float(np.mean(seed_r2s)),
            "std_train_r2":  float(np.std(seed_r2s)),
            "seed_train_r2s": seed_r2s,
            "best_params": best_params,
            "optuna_best_val_r2": best_val_r2,
        }
        print(f"  Train R²: {np.mean(seed_r2s):.4f} ± {np.std(seed_r2s):.4f}")

os.makedirs(Path(str(OUT)).parent, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
