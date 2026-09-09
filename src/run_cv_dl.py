"""
DL Optuna CV re-run
Loads best HP from dl_optuna_results_13feat.json (per target × model)
Runs 5-fold GroupKFold × 5 seeds CV with those HP
Output: dl_optuna_cv_results_13feat.json (same format as DL_Models)
"""
import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO      = Path(__file__).resolve().parent.parent
SEEDS     = [0, 1, 2, 3, 4]
OUT       = REPO / "results" / "dl_optuna_cv_results.json"
OPT_FILE  = REPO / "results" / "dl_optuna_results.json"

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

groups = (data["Co(%)"].astype(str)+"_"+data["Cr(%)"].astype(str)+"_"+
          data["Cu(%)"].astype(str)+"_"+data["Fe(%)"].astype(str)+"_"+
          data["Ni(%)"].astype(str)).values

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

gkf = GroupKFold(n_splits=5)
opt_results = json.load(open(OPT_FILE))
DL_MODELS = ["DeepMLP", "TabularTransformer", "AttentionMLP"]


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


def build_model(mname, params, n_feat):
    if mname == "DeepMLP":
        return DeepMLP(n_feat, params["h1"], params["h2"], params["h3"], params["dropout"])
    elif mname == "TabularTransformer":
        return TabularTransformer(n_feat, params["d_model"], params["nhead"], params["num_layers"])
    elif mname == "AttentionMLP":
        return AttentionMLP(n_feat, params["embed_dim"], params["nhead"])


def train_eval(model, X_tr, y_tr, X_te, y_te, lr, wd, epochs=500, patience=60):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    crit = nn.MSELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_te, dtype=torch.float32).to(device)
    yv = torch.tensor(y_te, dtype=torch.float32).to(device)

    n_val = max(1, int(len(Xt) * 0.15))
    Xfit, yfit = Xt[:-n_val], yt[:-n_val]
    Xval, yval = Xt[-n_val:], yt[-n_val:]

    best_loss, wait, best_state = float('inf'), 0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        crit(model(Xfit), yfit).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xval), yval).item()
        sched.step(vl)
        if vl < best_loss:
            best_loss, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).cpu().numpy()
    return float(r2_score(y_te, pred))


results = {}
total = len(ALL_TARGETS) * len(DL_MODELS)
rc = 0

for tkey, tcol in ALL_TARGETS.items():
    y = data[tcol].values.astype(np.float64)
    results[tkey] = {}

    for mname in DL_MODELS:
        rc += 1
        best_params = opt_results[tkey][mname]["best_params"]
        print(f"[{rc}/{total}] {tkey} {mname}", flush=True)

        scaler = StandardScaler()
        seed_fold_r2s = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            fold_r2s = []
            for tri, tei in gkf.split(X, y, groups):
                X_tr = scaler.fit_transform(X.iloc[tri].values)
                X_te = scaler.transform(X.iloc[tei].values)
                model = build_model(mname, best_params, n_features)
                r2 = train_eval(model, X_tr, y[tri], X_te, y[tei],
                                best_params["lr"], best_params["wd"])
                fold_r2s.append(float(r2))
            seed_fold_r2s.append(fold_r2s)

        all_r2s = [r for folds in seed_fold_r2s for r in folds]
        results[tkey][mname] = {
            "mean_r2": float(np.mean(all_r2s)),
            "std_r2":  float(np.std(all_r2s)),
            "seed_fold_r2s": seed_fold_r2s,
            "best_params": best_params,
        }
        print(f"  CV R²: {np.mean(all_r2s):.4f} ± {np.std(all_r2s):.4f}")

os.makedirs(Path(str(OUT)).parent, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
