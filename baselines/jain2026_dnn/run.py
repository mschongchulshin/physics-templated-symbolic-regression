"""
Jain, Jain & Verma 2026 DNN replication for HEA CoCrCuFeNi 696 dataset.

Paper: "Deep learning for high-entropy alloys: Phase prediction and feature
interpretability through hyperparameter optimization."
Proc IMechE Part C, DOI: 10.1177/09544062251414915

Original task: phase prediction (categorical) using 6 compositional Hume-Rothery
features (S_mix, dH_mix, delta, VEC, dChi, + composition).
Adapted task: regression for 12 phase/property targets in our HEA dataset.

Architecture (paper-based, default reasonable config to match PT-SR which is
not tuned):
  - Input layer: 7 features (5 composition + Hume-Rothery 5 + T) ... see below
  - Actually paper specifies 6 compositional features. We follow the paper's
    Hume-Rothery feature set + temperature, giving 7 features:
      S_mix, dH_mix, delta, VEC, dChi, composition_id (5-elem composition vector
      collapsed via 5 separate features), T.
    For fairness with the paper we pass: [S_mix, dH_mix, delta, VEC, dChi,
                                          Co%, Cr%, Cu%, Fe%, Ni%, T]
    -> 11 input features (the paper's 6 compositional features expanded to 5
    binary HR + 5 element fractions = 10, +T = 11). This matches the spirit of
    the paper which uses Hume-Rothery features alongside composition.
  - Hidden: 3 layers x 128 units, ReLU, dropout 0.2 (fixed reasonable default,
    no tuning, same as PT-SR fairness)
  - Output: 1 (regression)
  - Adam optimizer, MSE loss, 200 epochs, early stopping (patience 30)

Protocol:
  - 5-fold composition-grouped CV
  - 5 seeds (0..4)
  - 12 targets

Output: results.json + summary.md
"""
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def _load_sheet(path, sheet_name):
    """Rows for one temperature, from the flat corpus.

    These baselines were written against a workbook with one sheet per
    temperature. The corpus is deposited as a single CSV, so the sheet name is
    read as the temperature it stands for.
    """
    import pandas as _pd
    _t = int(str(sheet_name).rstrip("Kk"))
    _df = _pd.read_csv(path)
    return _df[_df["T_K"] == _t].drop(columns=["T_K"]).reset_index(drop=True)



warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

torch.set_num_threads(2)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE = str(Path(__file__).resolve().parent.parent)
DATA_FILE = f"{BASE}/data/CoCrCuFeNi_684.csv"
OUT_DIR = Path(f"{BASE}/baselines/jain2026_dnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "results.json"
OUT_MD = OUT_DIR / "summary.md"
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Element data
# ---------------------------------------------------------------
ELEMENTS = ["Co", "Cr", "Cu", "Fe", "Ni"]

# Atomic radii (pm) - mendeleev/Pauling consistent
R_I = np.array([125.0, 128.0, 128.0, 126.0, 124.0], dtype=float)
# Pauling electronegativities
CHI_I = np.array([1.88, 1.66, 1.90, 1.83, 1.91], dtype=float)
# Valence electron concentrations
VEC_I = np.array([9, 6, 11, 8, 10], dtype=float)

# Miedema binary mixing enthalpies (Takeuchi-Inoue 2005), kJ/mol
OMEGA = {
    ("Co", "Cr"): -4, ("Co", "Cu"):  6, ("Co", "Fe"): -1, ("Co", "Ni"):  0,
    ("Cr", "Cu"): 12, ("Cr", "Fe"): -1, ("Cr", "Ni"): -7,
    ("Cu", "Fe"): 13, ("Cu", "Ni"):  4,
    ("Fe", "Ni"): -2,
}
R_GAS = 8.314  # J/(mol K)


def hume_rothery_features(comp_frac):
    """Compute 5 Hume-Rothery features for a row of mole fractions."""
    x = np.asarray(comp_frac, dtype=float)
    # Numerical safety
    p = np.clip(x, 1e-12, 1.0)
    s_mix = -R_GAS * np.sum(p * np.log(p))  # J/(mol K)

    h = 0.0
    for i in range(len(ELEMENTS)):
        for j in range(i + 1, len(ELEMENTS)):
            key = (ELEMENTS[i], ELEMENTS[j])
            if key not in OMEGA:
                key = (ELEMENTS[j], ELEMENTS[i])
            h += 4.0 * OMEGA[key] * x[i] * x[j]
    h_mix = h  # kJ/mol

    r_avg = float(np.dot(x, R_I))
    delta = float(np.sqrt(np.sum(x * (1.0 - R_I / r_avg) ** 2)))

    vec = float(np.dot(x, VEC_I))

    chi_avg = float(np.dot(x, CHI_I))
    dchi = float(np.sqrt(np.sum(x * (CHI_I - chi_avg) ** 2)))

    return s_mix, h_mix, delta, vec, dchi


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
TEMPS = {"80K": 80, "300K": 300, "1100K": 1100}

dfs = []
for sheet, temp in TEMPS.items():
    df = _load_sheet(DATA_FILE, sheet)
    df["T"] = temp
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

data["comp_id"] = (
    data["Co(%)"].astype(str) + "_" + data["Cr(%)"].astype(str) + "_"
    + data["Cu(%)"].astype(str) + "_" + data["Fe(%)"].astype(str) + "_"
    + data["Ni(%)"].astype(str)
)

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

# Build feature matrix
comp_cols = [f"{e}(%)" for e in ELEMENTS]
comp_pct = data[comp_cols].values.astype(float)
comp_frac_arr = comp_pct / comp_pct.sum(axis=1, keepdims=True)

hr_feats = np.array([hume_rothery_features(row) for row in comp_frac_arr])
# columns: S_mix, dH_mix, delta, VEC, dChi
T_col = data["T"].values.astype(float).reshape(-1, 1)

# Final feature matrix (11 features, paper-faithful: 5 HR + 5 composition + T)
X_all = np.hstack([hr_feats, comp_pct, T_col]).astype(np.float64)
FEATURE_NAMES = ["S_mix", "dH_mix", "delta", "VEC", "dChi",
                 "Co%", "Cr%", "Cu%", "Fe%", "Ni%", "T"]
groups = data["comp_id"].values

print(f"Samples: {len(data)}, Features: {X_all.shape[1]} -> {FEATURE_NAMES}")


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
class JainDNN(nn.Module):
    """Default reasonable architecture per paper: 3 hidden layers x 128 units."""
    def __init__(self, input_dim, hidden_dim=128, n_hidden=3, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(prev, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
def train_one(model, X_tr, y_tr, X_val, y_val,
              epochs=200, lr=1e-3, batch_size=64, patience=30):
    model = model.to(DEVICE)

    x_scl = StandardScaler().fit(X_tr)
    y_mean = float(np.mean(y_tr))
    y_std = float(np.std(y_tr) + 1e-8)

    X_tr_s = x_scl.transform(X_tr)
    X_val_s = x_scl.transform(X_val)
    y_tr_s = (y_tr - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std

    Xt = torch.FloatTensor(X_tr_s).to(DEVICE)
    yt = torch.FloatTensor(y_tr_s).to(DEVICE)
    Xv = torch.FloatTensor(X_val_s).to(DEVICE)
    yv = torch.FloatTensor(y_val_s).to(DEVICE)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xv), yv).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, x_scl, y_mean, y_std


def predict(model, x_scl, y_mean, y_std, X):
    Xs = x_scl.transform(X)
    Xt = torch.FloatTensor(Xs).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(Xt).cpu().numpy()
    return out * y_std + y_mean


# ---------------------------------------------------------------
# CV loop
# ---------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5

all_results = {}
overall_t0 = time.time()

for tkey, tcol in TARGETS.items():
    y = data[tcol].values.astype(np.float64)
    print(f"\n=== Target: {tkey} ===")
    target_block = {"per_seed": [], "fold_r2_all": [], "fold_mae_all": []}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        gkf = GroupKFold(n_splits=N_FOLDS)
        seed_r2s = []
        seed_maes = []
        t0 = time.time()
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y, groups)):
            X_tr_full, y_tr_full = X_all[tr_idx], y[tr_idx]
            X_te, y_te = X_all[te_idx], y[te_idx]

            # Carve a 15% validation set out of training (deterministic per seed)
            n_val = max(1, int(len(X_tr_full) * 0.15))
            perm = rng.permutation(len(X_tr_full))
            v_idx = perm[:n_val]
            t_idx = perm[n_val:]
            X_t, y_t = X_tr_full[t_idx], y_tr_full[t_idx]
            X_v, y_v = X_tr_full[v_idx], y_tr_full[v_idx]

            model = JainDNN(input_dim=X_all.shape[1], hidden_dim=128,
                            n_hidden=3, dropout=0.2)
            model, x_scl, y_m, y_s = train_one(
                model, X_t, y_t, X_v, y_v,
                epochs=200, lr=1e-3, batch_size=64, patience=30,
            )
            pred = predict(model, x_scl, y_m, y_s, X_te)
            r2 = r2_score(y_te, pred)
            mae = mean_absolute_error(y_te, pred)
            seed_r2s.append(float(r2))
            seed_maes.append(float(mae))

            # Save final fold checkpoint for first seed (for reproducibility)
            if seed == SEEDS[0]:
                ckpt_path = CKPT_DIR / f"{tkey}_seed{seed}_fold{fold}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "x_scaler_mean": x_scl.mean_.tolist(),
                    "x_scaler_scale": x_scl.scale_.tolist(),
                    "y_mean": y_m, "y_std": y_s,
                    "feature_names": FEATURE_NAMES,
                    "fold": fold, "seed": seed,
                    "r2": r2, "mae": mae,
                }, ckpt_path)

        elapsed = time.time() - t0
        target_block["per_seed"].append({
            "seed": seed,
            "r2_mean": float(np.mean(seed_r2s)),
            "r2_std": float(np.std(seed_r2s)),
            "mae_mean": float(np.mean(seed_maes)),
            "mae_std": float(np.std(seed_maes)),
            "fold_r2s": seed_r2s,
            "fold_maes": seed_maes,
            "time_s": elapsed,
        })
        target_block["fold_r2_all"].extend(seed_r2s)
        target_block["fold_mae_all"].extend(seed_maes)
        print(f"  seed={seed}  R2={np.mean(seed_r2s):.4f}±{np.std(seed_r2s):.4f}"
              f"  MAE={np.mean(seed_maes):.4f}  ({elapsed:.1f}s)")

    target_block["r2_mean"] = float(np.mean(target_block["fold_r2_all"]))
    target_block["r2_std"] = float(np.std(target_block["fold_r2_all"]))
    target_block["mae_mean"] = float(np.mean(target_block["fold_mae_all"]))
    target_block["mae_std"] = float(np.std(target_block["fold_mae_all"]))
    all_results[tkey] = target_block
    print(f"  >> {tkey}: R2={target_block['r2_mean']:.4f}"
          f"±{target_block['r2_std']:.4f}")

    # Save incrementally
    with open(OUT_JSON, "w") as f:
        json.dump({
            "config": {
                "model": "JainDNN (3x128, ReLU, dropout=0.2)",
                "paper": "Jain, Jain & Verma 2026 (Proc IMechE Part C)",
                "doi": "10.1177/09544062251414915",
                "input_features": FEATURE_NAMES,
                "n_features": X_all.shape[1],
                "epochs": 200, "lr": 1e-3, "batch_size": 64,
                "patience": 30, "dropout": 0.2,
                "n_folds": N_FOLDS, "seeds": SEEDS,
                "device": str(DEVICE),
            },
            "results": all_results,
        }, f, indent=2)

print(f"\nTotal elapsed: {(time.time() - overall_t0)/60.0:.1f} min")

# ---------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------
lines = [
    "# Jain, Jain & Verma 2026 DNN baseline - HEA CoCrCuFeNi 696",
    "",
    "Paper: \"Deep learning for high-entropy alloys: Phase prediction and "
    "feature interpretability through hyperparameter optimization\"  ",
    "Proc IMechE Part C, DOI: 10.1177/09544062251414915 (2026)",
    "",
    "## Method",
    "- Architecture: 3 hidden layers x 128 units, ReLU, dropout 0.2",
    "- Loss: MSE, Optimizer: Adam (lr=1e-3), 200 epochs, early stopping (patience=30)",
    f"- Input: {len(FEATURE_NAMES)} features = {', '.join(FEATURE_NAMES)}",
    "- Hume-Rothery features: S_mix, dH_mix (Miedema, Takeuchi-Inoue 2005),"
    " delta (atomic size), VEC, dChi (Pauling electronegativity)",
    "- Protocol: 5-fold composition-grouped CV x 5 seeds (0-4)",
    "",
    "## Results",
    "",
    "| Target | R^2 mean | R^2 std | MAE mean | MAE std |",
    "|--------|---------:|--------:|---------:|--------:|",
]
r2_overall = []
for tkey in TARGETS:
    r = all_results[tkey]
    r2_overall.append(r["r2_mean"])
    lines.append(
        f"| {tkey} | {r['r2_mean']:.4f} | {r['r2_std']:.4f} | "
        f"{r['mae_mean']:.4f} | {r['mae_std']:.4f} |"
    )
lines += [
    "",
    f"**Overall mean R^2 across 12 targets: {np.mean(r2_overall):.4f}**",
    "",
    f"Total runtime: {(time.time() - overall_t0)/60.0:.1f} minutes  ",
    f"Device: {DEVICE}",
]
OUT_MD.write_text("\n".join(lines))
print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")
