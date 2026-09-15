
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

def _load_sheet(path, sheet_name):
    import pandas as _pd
    _t = int(str(sheet_name).rstrip("Kk"))
    _df = _pd.read_csv(path)
    return _df[_df["T_K"] == _t].drop(columns=["T_K"]).reset_index(drop=True)



warnings.filterwarnings("ignore")
torch.set_num_threads(2)

BASE = str(Path(__file__).resolve().parent.parent)
DATA_FILE = f"{BASE}/data/CoCrCuFeNi_684.csv"
FEAT_FILE = f"{BASE}/data/features_13.csv"
OUT_DIR = f"{BASE}/baselines/sci_rep_transformer"
OUT_JSON = f"{OUT_DIR}/results.json"
OUT_MD = f"{OUT_DIR}/summary.md"

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5

EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 64

device = torch.device("cpu")
print(f"Using device: {device}")
print(f"Threads: torch={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')}")


TEMPS = {"80K": 80, "300K": 300, "1100K": 1100}
dfs = []
for sheet, temp in TEMPS.items():
    df = _load_sheet(DATA_FILE, sheet)
    df["T"] = temp
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

CN = ["Co", "Cr", "Cu", "Fe", "Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(FEAT_FILE)
X_df = pd.concat([Xc, feat_df], axis=1)
X_df = X_df.loc[:, ~X_df.columns.duplicated()]
FEATURE_NAMES = list(X_df.columns)
N_FEATURES = X_df.shape[1]
X_all = X_df.values.astype(np.float32)

groups = (
    data["Co(%)"].astype(str) + "_" + data["Cr(%)"].astype(str) + "_" +
    data["Cu(%)"].astype(str) + "_" + data["Fe(%)"].astype(str) + "_" +
    data["Ni(%)"].astype(str)
).values

print(f"Samples: {len(data)}, Features ({N_FEATURES}): {FEATURE_NAMES}")
assert N_FEATURES == 13, f"Expected 13 features, got {N_FEATURES}"

TARGETS = {
    "Youngs_modulus": "Young's modulus(Gpa)",
    "UTS": "UTS(Gpa)",
    "Disloc_0pct": "Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc_20pct": "Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC_0pct": "FCC 0%(%)",
    "FCC_20pct": "FCC 20%(%)",
    "HCP_0pct": "HCP 0%(%)",
    "HCP_20pct": "HCP 20%(%)",
    "BCC_0pct": "BCC 0%(%)",
    "BCC_20pct": "BCC 20%(%)",
    "Other_0pct": "Other 0%(%)",
    "Other_20pct": "Other 20%(%)",
}


class SciRepTransformer(nn.Module):

    def __init__(self, n_features: int, d_model: int = EMBED_DIM, nhead: int = N_HEADS,
                 num_layers: int = N_LAYERS, dim_ff: int = FF_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        tokens = self.embed(x.unsqueeze(-1))
        tokens = tokens + self.pos
        h = self.encoder(tokens)
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def train_one(model, X_tr, y_tr, X_va, y_va, seed,
              max_epochs=MAX_EPOCHS, lr=LR, batch_size=BATCH_SIZE, patience=PATIENCE):
    g = torch.Generator()
    g.manual_seed(seed)

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    X_va_t = torch.from_numpy(X_va).float()
    y_va_t = torch.from_numpy(y_va).float()

    n = X_tr_t.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = criterion(val_pred, y_va_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).cpu().numpy()


os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(OUT_JSON):
    with open(OUT_JSON, "r") as f:
        results = json.load(f)
    done_targets = set(results.keys())
    print(f"Resuming. Already done: {sorted(done_targets)}")
else:
    results = {}
    done_targets = set()


def save_checkpoint():
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, OUT_JSON)


t0_total = time.time()

for tkey, tcol in TARGETS.items():
    if tkey in done_targets:
        print(f"[skip] {tkey} (already in checkpoint)")
        continue

    y_all = data[tcol].values.astype(np.float64)
    print(f"\n=== Target: {tkey} ({tcol}) ===")
    t0 = time.time()

    fold_seed_r2s = [[None] * len(SEEDS) for _ in range(N_FOLDS)]
    flat_r2s = []

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(X_all, y_all, groups))

    for fold, (tri_full, tei) in enumerate(splits):
        rng = np.random.default_rng(1000 + fold)
        perm = rng.permutation(len(tri_full))
        n_val = max(1, int(len(tri_full) * 0.15))
        val_idx_local = perm[:n_val]
        tr_idx_local = perm[n_val:]
        tri = tri_full[tr_idx_local]
        vai = tri_full[val_idx_local]

        x_scaler = StandardScaler().fit(X_all[tri])
        y_mean = float(np.mean(y_all[tri]))
        y_std = float(np.std(y_all[tri]) + 1e-12)

        X_tr = x_scaler.transform(X_all[tri]).astype(np.float32)
        X_va = x_scaler.transform(X_all[vai]).astype(np.float32)
        X_te = x_scaler.transform(X_all[tei]).astype(np.float32)

        y_tr = ((y_all[tri] - y_mean) / y_std).astype(np.float32)
        y_va = ((y_all[vai] - y_mean) / y_std).astype(np.float32)

        for s_idx, seed in enumerate(SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SciRepTransformer(n_features=N_FEATURES)
            model = train_one(model, X_tr, y_tr, X_va, y_va, seed=seed)

            pred_norm = predict(model, X_te)
            pred = pred_norm * y_std + y_mean
            r2 = float(r2_score(y_all[tei], pred))

            fold_seed_r2s[fold][s_idx] = r2
            flat_r2s.append(r2)

        elapsed = time.time() - t0
        print(f"  fold {fold}: seeds R2 = "
              f"{[f'{r:.3f}' for r in fold_seed_r2s[fold]]}  ({elapsed:.1f}s elapsed)")

    mean_r2 = float(np.mean(flat_r2s))
    std_r2 = float(np.std(flat_r2s))
    elapsed = time.time() - t0

    results[tkey] = {
        "mean_R2": mean_r2,
        "std_R2": std_r2,
        "fold_seed_R2s": fold_seed_r2s,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "time_sec": float(elapsed),
    }
    save_checkpoint()
    print(f"  -> {tkey}: R2 = {mean_r2:.4f} ± {std_r2:.4f}  ({elapsed:.1f}s) [checkpointed]")


total_elapsed = time.time() - t0_total
mean_r2_overall = float(np.mean([results[t]["mean_R2"] for t in TARGETS if t in results]))

lines = [
    "# Sci Rep 2025 Transformer baseline (PT-SR HEA dataset)",
    "",
    "Reference: Korkmaz et al., Sci. Reports (2025) "
    "https://www.nature.com/articles/s41598-025-95170-z",
    "",
    "**Architecture (simplified):** per-feature linear embed -> 2-layer "
    "TransformerEncoder (d_model=64, 4 heads, FF=256) -> mean pool -> MLP head.",
    "",
    f"**Data:** CoCrCuFeNi 696 (232 comps x 3 T), {N_FEATURES} input features.",
    f"**Protocol:** 5-fold composition-grouped CV x 5 seeds (0..4), Adam lr=1e-3, "
    f"up to {MAX_EPOCHS} epochs, early stopping (patience={PATIENCE}).",
    "",
    "## Per-target mean R2 (mean +/- std across all 25 fold*seed values)",
    "",
    "| Target | mean R2 | std R2 |",
    "| --- | ---: | ---: |",
]
for tkey in TARGETS:
    if tkey in results:
        r = results[tkey]
        lines.append(f"| {tkey} | {r['mean_R2']:.4f} | {r['std_R2']:.4f} |")

lines += [
    "",
    f"**Overall mean R2 (12 targets):** {mean_r2_overall:.4f}",
    f"**Total runtime:** {total_elapsed/60:.1f} min",
]

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"\nWrote summary -> {OUT_MD}")
print(f"Wrote results -> {OUT_JSON}")
print(f"Overall mean R2 = {mean_r2_overall:.4f}")
