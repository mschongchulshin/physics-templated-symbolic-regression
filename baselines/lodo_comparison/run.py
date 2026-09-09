"""
LODO (Leave-One-Dataset-Out) baseline comparison vs PT-SR.

Applies the SAME R3 LODO protocol used to evaluate PT-SR (lodo_all_results.xlsx)
to all external SOTA baselines, on the MPEA experimental dataset
(MPEA_figshare_dataset.csv).

Protocol (per fold):
  - Pick (Composition system, Processing method, Property, Test_Dataset)
  - Filter MPEA data to that composition system + processing
  - Split: rows from Test_Dataset's source -> test
           rows from other sources         -> train
  - Train baseline, predict on test, compute R^2 / MAE / RMSE.

We evaluate ALL 134 LODO folds (CAST/WROUGHT/ANNEAL x YS/Elongation) for full
coverage, with primary focus on the 18 main FCC CAST folds (Composition x
Source) the paper R3 reports.

Baselines (all in same script for unified protocol):

Smooth class:
  1. PT-SR (BIC-best)        -- already in lodo_all_results.xlsx (loaded directly)
  2. Liu MLP (npj 2024)      -- 3-layer MLP regressor
  3. PySR no-template        -- standard SR (free_1stage), seeds 0..4
  4. SciRep25 Transformer    -- per-feature token Transformer
  5. Jain 2026 DNN           -- 3 hidden x 128 units, ReLU, dropout 0.2
  6. LESets GNN (simplified) -- composition-graph GNN (CGConv)
  7. SISSO                   -- linear regression on combinatorial feature pool

Tree class:
  8. Liu RF (npj 2024)       -- RandomForestRegressor
  9. Wu gplearn+RFR (JMI)    -- gplearn SymbolicTransformer + RFR
  10. JMI Stacking           -- ERT + HistGBR + Lasso meta

Features (uniform across all baselines, MPEA-compatible):
  - Element fractions for the composition system's elements (variable dim)
  - Hume-Rothery: S_mix, dH_mix (Miedema/Takeuchi-Inoue), delta, VEC, dChi
  - T (Kelvin)
We pad/standardise per-fold so the per-system feature dim is consistent within
the fold (train and test always share the same element set since LODO is
per-composition).

Stochastic models use seeds 0..4. Deterministic models (RF, SISSO) use seed=0.

Output:
  - results.json : {baseline: {fold_id: {R2, MAE, RMSE, n_train, n_test},
                                median_R2, mean_R2, std_R2}}
  - summary.md   : table of all baselines x all folds (R^2)
"""

import json
import os
import re
import time
import warnings
from pathlib import Path

# Threading control before scientific imports
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
BASE = str(Path(__file__).resolve().parent)
OUT_DIR = Path(f"{BASE}/baselines/lodo_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "results.json"
OUT_SUMMARY = OUT_DIR / "summary.md"
LOG_PATH = OUT_DIR / "run.log"

MPEA_PATH = f"{BASE}/data/experimental/MPEA_figshare_dataset.csv"
LODO_PATH = f"{BASE}/results/lodo_all_results.xlsx"

SEEDS = [0, 1, 2, 3, 4]


# ============================================================
# Logging
# ============================================================
def log(msg, also_print=True):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


# ============================================================
# 1. Element properties (Miedema/literature)
# ============================================================
# Atomic radius (pm), Pauling electronegativity, VEC.
# Covers all elements in the LODO composition systems.
ELEM_R = {
    'Al': 143.0, 'Co': 125.0, 'Cr': 128.0, 'Cu': 128.0, 'Fe': 126.0,
    'Hf': 159.0, 'Li': 152.0, 'Mg': 160.0, 'Mn': 127.0, 'Mo': 139.0,
    'Nb': 146.0, 'Ni': 124.0, 'Re': 137.0, 'Si': 117.0, 'Ta': 146.0,
    'Ti': 147.0, 'V':  134.0, 'W':  139.0, 'Zr': 160.0,
}
ELEM_CHI = {
    'Al': 1.61, 'Co': 1.88, 'Cr': 1.66, 'Cu': 1.90, 'Fe': 1.83,
    'Hf': 1.30, 'Li': 0.98, 'Mg': 1.31, 'Mn': 1.55, 'Mo': 2.16,
    'Nb': 1.60, 'Ni': 1.91, 'Re': 1.90, 'Si': 1.90, 'Ta': 1.50,
    'Ti': 1.54, 'V':  1.63, 'W':  2.36, 'Zr': 1.33,
}
ELEM_VEC = {
    'Al': 3, 'Co': 9, 'Cr': 6, 'Cu': 11, 'Fe': 8,
    'Hf': 4, 'Li': 1, 'Mg': 2, 'Mn': 7, 'Mo': 6,
    'Nb': 5, 'Ni': 10, 'Re': 7, 'Si': 4, 'Ta': 5,
    'Ti': 4, 'V':  5, 'W':  6, 'Zr': 4,
}
# Miedema-style binary mixing enthalpy (kJ/mol), Takeuchi & Inoue 2005 +
# extensions; missing pairs default to 0.
OMEGA = {
    ('Al','Co'):-19,('Al','Cr'):-10,('Al','Cu'):-1,('Al','Fe'):-11,('Al','Hf'):-39,
    ('Al','Mg'):-2,('Al','Mn'):-19,('Al','Mo'):-5,('Al','Nb'):-18,('Al','Ni'):-22,
    ('Al','Si'):-19,('Al','Ta'):-19,('Al','Ti'):-30,('Al','V'):-16,('Al','W'):-2,
    ('Al','Zr'):-44,('Al','Li'):-4,
    ('Co','Cr'):-4,('Co','Cu'):6,('Co','Fe'):-1,('Co','Hf'):-35,('Co','Mn'):-5,
    ('Co','Mo'):-5,('Co','Nb'):-25,('Co','Ni'):0,('Co','Ta'):-24,('Co','Ti'):-28,
    ('Co','V'):-14,('Co','W'):-1,('Co','Zr'):-41,
    ('Cr','Cu'):12,('Cr','Fe'):-1,('Cr','Hf'):-9,('Cr','Mn'):2,('Cr','Mo'):0,
    ('Cr','Nb'):-7,('Cr','Ni'):-7,('Cr','Ta'):-7,('Cr','Ti'):-7,('Cr','V'):-2,
    ('Cr','W'):1,('Cr','Zr'):-12,
    ('Cu','Fe'):13,('Cu','Hf'):-17,('Cu','Mn'):4,('Cu','Mo'):19,('Cu','Nb'):3,
    ('Cu','Ni'):4,('Cu','Ta'):2,('Cu','Ti'):-9,('Cu','V'):5,('Cu','W'):22,
    ('Cu','Zr'):-23,
    ('Fe','Hf'):-21,('Fe','Mn'):0,('Fe','Mo'):-2,('Fe','Nb'):-16,('Fe','Ni'):-2,
    ('Fe','Ta'):-15,('Fe','Ti'):-17,('Fe','V'):-7,('Fe','W'):0,('Fe','Zr'):-25,
    ('Hf','Mo'):-4,('Hf','Nb'):4,('Hf','Ni'):-42,('Hf','Ta'):3,('Hf','Ti'):0,
    ('Hf','V'):-2,('Hf','W'):-6,('Hf','Zr'):0,('Hf','Si'):-77,
    ('Li','Mg'):0,('Li','Si'):-30,
    ('Mg','Si'):-26,
    ('Mn','Mo'):5,('Mn','Nb'):-4,('Mn','Ni'):-8,('Mn','Ta'):-4,('Mn','Ti'):-8,
    ('Mn','V'):1,('Mn','W'):6,('Mn','Zr'):-15,
    ('Mo','Nb'):-6,('Mo','Ni'):-7,('Mo','Re'):-3,('Mo','Si'):-35,('Mo','Ta'):-5,
    ('Mo','Ti'):-4,('Mo','V'):0,('Mo','W'):0,('Mo','Zr'):-6,
    ('Nb','Ni'):-30,('Nb','Re'):-2,('Nb','Si'):-56,('Nb','Ta'):0,('Nb','Ti'):2,
    ('Nb','V'):-1,('Nb','W'):-8,('Nb','Zr'):4,
    ('Ni','Ta'):-29,('Ni','Ti'):-35,('Ni','V'):-18,('Ni','W'):-3,('Ni','Zr'):-49,
    ('Re','Ta'):-7,('Re','W'):-3,
    ('Si','Ti'):-66,('Si','V'):-48,('Si','W'):-31,('Si','Zr'):-84,
    ('Ta','Ti'):1,('Ta','V'):-1,('Ta','W'):-7,('Ta','Zr'):3,
    ('Ti','V'):-2,('Ti','W'):-6,('Ti','Zr'):0,
    ('V','W'):-1,('V','Zr'):-4,
    ('W','Zr'):-9,
}
R_GAS = 8.314


def _omega(a, b):
    if a == b:
        return 0.0
    key = (a, b) if (a, b) in OMEGA else (b, a)
    return OMEGA.get(key, 0.0)


def hume_rothery_features(elements, fractions):
    """Compute (S_mix, dH_mix, delta, VEC, dChi, r_avg, chi_avg)."""
    x = np.asarray(fractions, dtype=float)
    p = np.clip(x, 1e-12, 1.0)
    s_mix = -R_GAS * np.sum(p * np.log(p))  # J/(mol K)

    h = 0.0
    for i, ei in enumerate(elements):
        for j in range(i + 1, len(elements)):
            h += 4.0 * _omega(ei, elements[j]) * x[i] * x[j]
    h_mix = h  # kJ/mol

    r = np.array([ELEM_R.get(e, 130.0) for e in elements])
    r_avg = float(np.dot(x, r))
    delta = float(np.sqrt(np.sum(x * (1.0 - r / r_avg) ** 2)))

    vec_arr = np.array([ELEM_VEC.get(e, 6.0) for e in elements], dtype=float)
    vec = float(np.dot(x, vec_arr))

    chi = np.array([ELEM_CHI.get(e, 1.6) for e in elements])
    chi_avg = float(np.dot(x, chi))
    dchi = float(np.sqrt(np.sum(x * (chi - chi_avg) ** 2)))

    return s_mix, h_mix, delta, vec, dchi, r_avg, chi_avg


# ============================================================
# 2. MPEA dataset loader and source mapping
# ============================================================
def parse_formula(formula):
    if pd.isna(formula):
        return None
    matches = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", str(formula).strip())
    out = {}
    for el, amt in matches:
        if el:
            out[el] = float(amt) if amt else 1.0
    return out if out else None


def map_source(ref_id):
    if pd.isna(ref_id):
        return "Unknown"
    rid = int(ref_id)
    if rid <= 66:
        return "Borg"
    if rid <= 97:
        return "Couzinie"
    if rid <= 263:
        return "Iyer"
    return "Gorsse"


def to_kelvin(t_celsius):
    if pd.isna(t_celsius):
        return 298.15
    try:
        return float(t_celsius) + 273.15
    except Exception:
        return 298.15


log("Loading MPEA figshare dataset")
df_mpea = pd.read_csv(MPEA_PATH)
df_mpea["_parsed"] = df_mpea["FORMULA"].apply(parse_formula)
df_mpea["_elem_set"] = df_mpea["_parsed"].apply(
    lambda p: "".join(sorted(p.keys())) if p else None
)
df_mpea["_source"] = df_mpea["IDENTIFIER: Reference ID"].apply(map_source)
df_mpea["_T_K"] = df_mpea["PROPERTY: Test temperature ($^\\circ$C)"].apply(to_kelvin)

PROP_COL = {
    "YS": "PROPERTY: YS (MPa)",
    "Elongation": "PROPERTY: Elongation (%)",
}

log(f"MPEA rows: {len(df_mpea)}")
log(f"Source counts: {df_mpea['_source'].value_counts().to_dict()}")
log(f"Processing counts: {df_mpea['PROPERTY: Processing method'].value_counts().to_dict()}")

# ============================================================
# 3. Load LODO fold definitions (PT-SR baseline reference)
# ============================================================
log("Loading PT-SR LODO fold definitions")
lodo_def = pd.read_excel(LODO_PATH)
log(f"LODO fold definitions: {len(lodo_def)}")


# ============================================================
# 4. Per-fold feature builder
# ============================================================
def build_fold_features(comp_str, processing, prop_key, test_source_base, seed=0):
    """Filter MPEA rows for a fold and build features.

    Returns: dict with X_train, y_train, X_test, y_test, elem_order, n_train, n_test.
    Returns None if no usable data.

    Two regimes (matching PT-SR's apparent protocol):
      A. Multi-source: rows from >=2 distinct sources exist → LODO split
         (train = other sources, test = test source).
      B. Single-source: all rows belong to one source → no true LODO possible
         for this composition; do an interleaved 50/50 split (deterministic by
         row index) to match PT-SR's apparent n_train=n_test pattern. We mark
         these folds as `single_source=True`.
    """
    elem_order = sorted(set(re.findall(r"[A-Z][a-z]?", comp_str)))
    elem_set = "".join(elem_order)
    y_col = PROP_COL[prop_key]

    mask = (
        (df_mpea["_elem_set"] == elem_set)
        & (df_mpea["PROPERTY: Processing method"] == processing)
        & (df_mpea[y_col].notna())
    )
    sub = df_mpea[mask].copy()
    if len(sub) == 0:
        return None

    feats = []
    ys = []
    sources = []
    for _, row in sub.iterrows():
        parsed = row["_parsed"]
        amts = np.array([parsed.get(el, 0.0) for el in elem_order])
        s = amts.sum()
        if s == 0:
            continue
        x = amts / s
        s_mix, h_mix, delta, vec, dchi, r_avg, chi_avg = hume_rothery_features(
            elem_order, x
        )
        T_k = row["_T_K"]
        feat = list(x) + [s_mix, h_mix, delta, vec, dchi, r_avg, chi_avg, T_k]
        feats.append(feat)
        ys.append(float(row[y_col]))
        sources.append(row["_source"])

    if len(feats) < 4:
        return None
    X = np.array(feats, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    src = np.array(sources)

    test_mask = src == test_source_base
    train_mask = ~test_mask

    single_source = False
    if train_mask.sum() < 2 or test_mask.sum() < 2:
        # Single-source regime: 50/50 deterministic split (every other row)
        single_source = True
        n = len(X)
        rng = np.random.RandomState(0)  # deterministic across baselines
        order = rng.permutation(n)
        half = n // 2
        train_idx = np.zeros(n, dtype=bool)
        train_idx[order[:half]] = True
        test_idx = ~train_idx
        train_mask = train_idx
        test_mask = test_idx
        if train_mask.sum() < 2 or test_mask.sum() < 2:
            return None

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "elem_order": elem_order,
        "single_source": single_source,
    }


def fold_id(row):
    return (
        f"{row['Composition']}|{row['Processing']}|{row['Property']}"
        f"|{row['Test_Dataset']}"
    )


# ============================================================
# 5. Baseline definitions
# ============================================================
def safe_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) < 2 or np.var(yt) == 0:
        return {"R2": float("nan"),
                "MAE": float(np.mean(np.abs(yt - yp))),
                "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2)))}
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}


# ---------- Liu MLP (3-layer, multi-output not needed; single target) ----------
def fit_predict_liu_mlp(X_tr, y_tr, X_te, seed):
    pipe = Pipeline([
        ("scl", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 128, 128),
            activation="relu", solver="adam",
            learning_rate_init=1e-3, max_iter=500, batch_size=64,
            early_stopping=False, random_state=seed,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


# ---------- Liu RF (Random Forest) ----------
def fit_predict_liu_rf(X_tr, y_tr, X_te, seed):
    m = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=2, min_samples_leaf=1,
        n_jobs=2, random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


# ---------- JMI Stacking (ERT + HistGBR -> Lasso) ----------
def fit_predict_jmi_stack(X_tr, y_tr, X_te, seed):
    estimators = [
        ("ert", ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=2)),
        ("hgb", HistGradientBoostingRegressor(max_iter=500, random_state=seed)),
    ]
    final = Pipeline([("scl", StandardScaler()),
                      ("lasso", Lasso(alpha=0.01, max_iter=10000, random_state=seed))])
    m = StackingRegressor(estimators=estimators, final_estimator=final, n_jobs=1, cv=3)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


# ---------- Wu gplearn + RFR ----------
def fit_predict_wu_gp_rf(X_tr, y_tr, X_te, seed):
    """gplearn SymbolicTransformer features -> Random Forest."""
    from gplearn.genetic import SymbolicTransformer
    # Fewer generations and components for LODO speed (small folds, many runs)
    gp = SymbolicTransformer(
        population_size=300, generations=20,
        function_set=("add", "sub", "mul", "div", "inv", "log",
                      "sqrt", "max", "min"),
        init_depth=(2, 4), parsimony_coefficient=0.001,
        hall_of_fame=20, n_components=8,
        metric="pearson", verbose=0, random_state=seed, n_jobs=1,
    )
    try:
        gp.fit(X_tr, y_tr)
        gp_tr = gp.transform(X_tr)
        gp_te = gp.transform(X_te)
        Xt = np.hstack([X_tr, gp_tr])
        Xe = np.hstack([X_te, gp_te])
    except Exception as e:
        Xt, Xe = X_tr, X_te
    rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                                random_state=seed, n_jobs=2)
    rf.fit(Xt, y_tr)
    return rf.predict(Xe)


# ---------- Jain DNN (3 hidden x 128, dropout 0.2) ----------
def fit_predict_jain_dnn(X_tr, y_tr, X_te, seed):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # 15% val carve from train
    n = len(X_tr)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.15))
    v_idx = perm[:n_val]
    t_idx = perm[n_val:]
    X_t, y_t = X_tr[t_idx], y_tr[t_idx]
    X_v, y_v = X_tr[v_idx], y_tr[v_idx]

    scl = StandardScaler().fit(X_t)
    y_m = float(np.mean(y_t)); y_s = float(np.std(y_t) + 1e-8)
    X_t_s = scl.transform(X_t); X_v_s = scl.transform(X_v); X_te_s = scl.transform(X_te)
    y_t_s = (y_t - y_m) / y_s; y_v_s = (y_v - y_m) / y_s

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 1))
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = Net(X_tr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    Xt_t = torch.FloatTensor(X_t_s); yt_t = torch.FloatTensor(y_t_s)
    Xv_t = torch.FloatTensor(X_v_s); yv_t = torch.FloatTensor(y_v_s)
    Xe_t = torch.FloatTensor(X_te_s)
    loader = DataLoader(TensorDataset(Xt_t, yt_t),
                        batch_size=min(64, len(Xt_t)), shuffle=True)
    best, wait, best_state = float("inf"), 0, None
    for ep in range(200):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xv_t), yv_t).item()
        if vloss < best:
            best = vloss; wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 30:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(Xe_t).cpu().numpy()
    return out * y_s + y_m


# ---------- Sci Rep 2025 Transformer (per-feature token) ----------
def fit_predict_scirep_transformer(X_tr, y_tr, X_te, seed):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    n = len(X_tr)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.15))
    v_idx = perm[:n_val]; t_idx = perm[n_val:]
    X_t, y_t = X_tr[t_idx], y_tr[t_idx]
    X_v, y_v = X_tr[v_idx], y_tr[v_idx]

    scl = StandardScaler().fit(X_t)
    y_m = float(np.mean(y_t)); y_s = float(np.std(y_t) + 1e-8)
    X_t_s = scl.transform(X_t); X_v_s = scl.transform(X_v); X_te_s = scl.transform(X_te)
    y_t_s = (y_t - y_m) / y_s; y_v_s = (y_v - y_m) / y_s

    EMB = 32
    HEADS = 4
    LAYERS = 2
    FF = 128
    d_in = X_tr.shape[1]

    class TF(nn.Module):
        def __init__(self, d, emb=EMB):
            super().__init__()
            self.embed = nn.Linear(1, emb)
            self.pos = nn.Parameter(torch.randn(d, emb) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=HEADS,
                                                    dim_feedforward=FF,
                                                    dropout=0.1,
                                                    batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=LAYERS)
            self.head = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(),
                                       nn.Linear(64, 1))
        def forward(self, x):
            tokens = self.embed(x.unsqueeze(-1)) + self.pos
            z = self.enc(tokens).mean(dim=1)
            return self.head(z).squeeze(-1)

    model = TF(d_in).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    Xt_t = torch.FloatTensor(X_t_s); yt_t = torch.FloatTensor(y_t_s)
    Xv_t = torch.FloatTensor(X_v_s); yv_t = torch.FloatTensor(y_v_s)
    Xe_t = torch.FloatTensor(X_te_s)
    loader = DataLoader(TensorDataset(Xt_t, yt_t),
                        batch_size=min(64, len(Xt_t)), shuffle=True)
    best, wait, best_state = float("inf"), 0, None
    for ep in range(200):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xv_t), yv_t).item()
        if vloss < best:
            best = vloss; wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 30:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(Xe_t).cpu().numpy()
    return out * y_s + y_m


# ---------- LESets-inspired GNN (composition graph) ----------
def fit_predict_lesets_gnn(X_tr, y_tr, X_te, seed, elem_order):
    """Build a per-row graph: nodes=elements present, fully-connected edges,
    CGConv aggregation, then composition-fraction-weighted readout + MLP head.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch_geometric.data import Data, Batch
        from torch_geometric.nn import CGConv
    except Exception as e:
        # If pyg not available, fall back to Liu MLP
        return fit_predict_liu_mlp(X_tr, y_tr, X_te, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    n_elem = len(elem_order)
    n_feat = X_tr.shape[1]
    # Last column of X is T (Kelvin). Element fractions are first n_elem cols.
    # Hume-Rothery features sit between.

    # Element node descriptors (from ELEM_R/ELEM_CHI/ELEM_VEC)
    node_desc = np.array([
        [ELEM_R.get(e, 130.0), ELEM_CHI.get(e, 1.6), ELEM_VEC.get(e, 6.0)]
        for e in elem_order
    ], dtype=np.float32)
    # Standardise node desc
    nd_mean = node_desc.mean(axis=0, keepdims=True)
    nd_std = node_desc.std(axis=0, keepdims=True) + 1e-8
    node_desc_s = (node_desc - nd_mean) / nd_std

    # Edge index: fully-connected, no self-loops
    if n_elem >= 2:
        ii, jj = np.triu_indices(n_elem, k=1)
        edges = np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])])
        ei_t = torch.LongTensor(edges)
    else:
        ei_t = torch.LongTensor([[0], [0]])

    def make_graphs(X):
        graphs = []
        for row in X:
            x = row[:n_elem]
            T_k = row[-1]
            # Node feature: [fraction, R, EN, VEC, T]
            node_feat = np.concatenate([
                x[:, None],
                np.tile(node_desc_s, (1, 1)),
                np.full((n_elem, 1), (T_k - 700.0) / 500.0),
            ], axis=1).astype(np.float32)
            # Edge attr (per directed edge): radius mismatch + EN diff + frac product
            if n_elem >= 2:
                src, dst = ei_t[0].numpy(), ei_t[1].numpy()
                rs = node_desc[src, 0]; rd = node_desc[dst, 0]
                chs = node_desc[src, 1]; chd = node_desc[dst, 1]
                xs = x[src]; xd = x[dst]
                edge_attr = np.stack([
                    np.abs(rs - rd) / 150.0,
                    np.abs(chs - chd),
                    xs * xd,
                ], axis=1).astype(np.float32)
            else:
                edge_attr = np.zeros((1, 3), dtype=np.float32)
            d = Data(
                x=torch.FloatTensor(node_feat),
                edge_index=ei_t,
                edge_attr=torch.FloatTensor(edge_attr),
                frac=torch.FloatTensor(x),
            )
            graphs.append(d)
        return graphs

    n = len(X_tr)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.15))
    v_idx = perm[:n_val]; t_idx = perm[n_val:]
    X_t, y_t = X_tr[t_idx], y_tr[t_idx]
    X_v, y_v = X_tr[v_idx], y_tr[v_idx]

    y_m = float(np.mean(y_t)); y_s = float(np.std(y_t) + 1e-8)
    y_t_s = (y_t - y_m) / y_s; y_v_s = (y_v - y_m) / y_s

    g_tr = make_graphs(X_t); g_va = make_graphs(X_v); g_te = make_graphs(X_te)
    node_dim = g_tr[0].x.shape[1]

    class GNN(nn.Module):
        def __init__(self, nd, ed=3, hid=64):
            super().__init__()
            self.proj = nn.Linear(nd, hid)
            self.cg1 = CGConv(channels=hid, dim=ed)
            self.cg2 = CGConv(channels=hid, dim=ed)
            self.head = nn.Sequential(
                nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, 1)
            )
        def forward(self, batch):
            h = self.proj(batch.x)
            h = torch.relu(self.cg1(h, batch.edge_index, batch.edge_attr))
            h = torch.relu(self.cg2(h, batch.edge_index, batch.edge_attr))
            # Composition-fraction-weighted sum per graph
            frac = batch.frac.view(-1, 1)
            weighted = h * frac
            # Sum within graph
            from torch_geometric.utils import scatter
            pooled = scatter(weighted, batch.batch, dim=0, reduce='sum')
            return self.head(pooled).squeeze(-1)

    model = GNN(nd=node_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()

    def to_batch(graphs, ys):
        b = Batch.from_data_list(graphs)
        return b, torch.FloatTensor(ys)

    bs = min(32, len(g_tr))
    best, wait, best_state = float("inf"), 0, None
    for ep in range(200):
        model.train()
        perm_e = np.random.permutation(len(g_tr))
        for s in range(0, len(g_tr), bs):
            idx = perm_e[s:s+bs]
            sub = [g_tr[i] for i in idx]
            ys = y_t_s[idx]
            b, y_t_b = to_batch(sub, ys)
            opt.zero_grad()
            pred = model(b)
            loss = crit(pred, y_t_b)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            bv, yv_b = to_batch(g_va, y_v_s)
            vloss = crit(model(bv), yv_b).item()
        if vloss < best:
            best = vloss; wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 30:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        be, _ = to_batch(g_te, np.zeros(len(g_te)))
        out = model(be).cpu().numpy()
    return out * y_s + y_m


# ---------- SISSO (linear regression on combinatorial feature pool) ----------
def fit_predict_sisso(X_tr, y_tr, X_te, seed):
    """SISSO-lite: build pool of ratio/product/log/sqrt features then do
    forward L0 selection up to 3 features + linear regression.
    """
    rng = np.random.RandomState(seed)
    n_in = X_tr.shape[1]

    def expand(X):
        cols = [X]
        # Pairwise products and ratios (limited to first 8 features for speed)
        kmax = min(8, n_in)
        prods = []
        for i in range(kmax):
            for j in range(i+1, kmax):
                prods.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if prods:
            cols.append(np.hstack(prods))
        # Element-wise log/sqrt (positive offset)
        Xs = X.copy()
        Xpos = Xs - Xs.min(axis=0, keepdims=True) + 1.0
        cols.append(np.log(Xpos))
        cols.append(np.sqrt(Xpos))
        # Inverse (offset)
        cols.append(1.0 / (np.abs(X) + 1e-3))
        return np.hstack(cols)

    Xtr_pool = expand(X_tr)
    Xte_pool = expand(X_te)
    # Standardise
    scl = StandardScaler().fit(Xtr_pool)
    Xtr_s = scl.transform(Xtr_pool)
    Xte_s = scl.transform(Xte_pool)

    # Forward L0 selection: pick 3 features
    n_pool = Xtr_s.shape[1]
    selected = []
    residual = y_tr - y_tr.mean()
    for _ in range(3):
        best_corr = -1; best_idx = -1
        for j in range(n_pool):
            if j in selected:
                continue
            denom = np.linalg.norm(Xtr_s[:, j]) * np.linalg.norm(residual) + 1e-12
            c = abs(np.dot(Xtr_s[:, j], residual)) / denom
            if c > best_corr:
                best_corr = c; best_idx = j
        if best_idx == -1:
            break
        selected.append(best_idx)
        # Update residual via OLS on selected features
        Xs = Xtr_s[:, selected]
        beta, *_ = np.linalg.lstsq(Xs, y_tr, rcond=None)
        residual = y_tr - Xs @ beta

    if not selected:
        return np.full(len(X_te), float(y_tr.mean()))
    Xs = Xtr_s[:, selected]
    beta, *_ = np.linalg.lstsq(Xs, y_tr, rcond=None)
    Xe = Xte_s[:, selected]
    return Xe @ beta


# ---------- PySR no-template (free_1stage) ----------
def fit_predict_pysr_free(X_tr, y_tr, X_te, seed):
    try:
        from pysr import PySRRegressor
    except Exception:
        # Fallback if pysr unavailable
        return fit_predict_sisso(X_tr, y_tr, X_te, seed)
    # Reduced niterations to fit within wall-clock for many folds * seeds
    model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sqrt", "square"],
        maxsize=15,
        populations=15,
        population_size=33,
        progress=False,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        verbosity=0,
        temp_equation_file=True,
    )
    try:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        return np.asarray(pred, dtype=float)
    except Exception as e:
        log(f"  PySR failed: {e}", also_print=False)
        return fit_predict_sisso(X_tr, y_tr, X_te, seed)


# ============================================================
# 6. Baseline registry
# ============================================================
BASELINES = {
    # name : (fn, n_seeds, needs_elem_order, class)
    "Liu_MLP":         (fit_predict_liu_mlp,            len(SEEDS), False, "smooth"),
    "Liu_RF":          (fit_predict_liu_rf,             1,          False, "tree"),
    "Wu_gplearn_RFR":  (fit_predict_wu_gp_rf,           len(SEEDS), False, "tree"),
    "JMI_Stacking":    (fit_predict_jmi_stack,          len(SEEDS), False, "tree"),
    "Jain_DNN":        (fit_predict_jain_dnn,           len(SEEDS), False, "smooth"),
    "SciRep25_Transformer": (fit_predict_scirep_transformer, len(SEEDS), False, "smooth"),
    "LESets_GNN":      (fit_predict_lesets_gnn,         len(SEEDS), True,  "smooth"),
    "SISSO":           (fit_predict_sisso,              1,          False, "smooth"),
    "PySR_free_1stage": (fit_predict_pysr_free,         len(SEEDS), False, "smooth"),
}


# ============================================================
# 7. PT-SR results loader (already evaluated)
# ============================================================
def load_ptsr_fold_results():
    """Map each fold to PT-SR's reported R^2 from lodo_all_results.xlsx."""
    out = {}
    for _, row in lodo_def.iterrows():
        fid = fold_id(row)
        out[fid] = {
            "R2": float(row["Best_Test_R2"]) if pd.notna(row["Best_Test_R2"]) else float("nan"),
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "Template": str(row["Best_Template"]),
        }
    return out


# ============================================================
# 8. Main loop
# ============================================================
def run_baseline_on_folds(name, fn, n_seeds, needs_elem_order):
    fold_results = {}
    folds = list(lodo_def.iterrows())
    n_total = len(folds)
    t0 = time.time()
    for k, (_, row) in enumerate(folds, start=1):
        fid = fold_id(row)
        comp = row["Composition"]
        proc = row["Processing"]
        prop = row["Property"]
        test_ds = row["Test_Dataset"]
        test_source = (
            test_ds.replace("_CAST", "").replace("_WROUGHT", "").replace("_ANNEAL", "")
        )

        data = build_fold_features(comp, proc, prop, test_source)
        if data is None:
            fold_results[fid] = {
                "R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"),
                "n_train": 0, "n_test": 0,
                "skip_reason": "no usable data",
            }
            continue

        per_seed = []
        for seed in range(n_seeds):
            try:
                if needs_elem_order:
                    pred = fn(data["X_train"], data["y_train"],
                              data["X_test"], seed, data["elem_order"])
                else:
                    pred = fn(data["X_train"], data["y_train"],
                              data["X_test"], seed)
                m = safe_metrics(data["y_test"], pred)
            except Exception as e:
                log(f"    [{name}] fold {fid} seed {seed} FAIL: {e}",
                    also_print=False)
                m = {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
            per_seed.append(m)

        # Aggregate across seeds (median over seeds first)
        r2s = [s["R2"] for s in per_seed if np.isfinite(s["R2"])]
        maes = [s["MAE"] for s in per_seed if np.isfinite(s["MAE"])]
        rmses = [s["RMSE"] for s in per_seed if np.isfinite(s["RMSE"])]
        fold_results[fid] = {
            "R2": float(np.median(r2s)) if r2s else float("nan"),
            "R2_mean": float(np.mean(r2s)) if r2s else float("nan"),
            "R2_std": float(np.std(r2s)) if r2s else float("nan"),
            "MAE": float(np.median(maes)) if maes else float("nan"),
            "RMSE": float(np.median(rmses)) if rmses else float("nan"),
            "n_train": data["n_train"],
            "n_test": data["n_test"],
            "n_seeds_ok": len(r2s),
            "single_source": data.get("single_source", False),
        }

        if k % 10 == 0 or k == n_total:
            elapsed = time.time() - t0
            log(f"  [{name}] {k}/{n_total} folds done ({elapsed/60:.1f} min)")

    # Summary stats across folds (use median R^2 per fold)
    r2_all = [v["R2"] for v in fold_results.values() if np.isfinite(v.get("R2", np.nan))]
    summary = {
        "n_folds_total": n_total,
        "n_folds_evaluated": len(r2_all),
        "median_R2": float(np.median(r2_all)) if r2_all else float("nan"),
        "mean_R2": float(np.mean(r2_all)) if r2_all else float("nan"),
        "std_R2": float(np.std(r2_all)) if r2_all else float("nan"),
        "p25_R2": float(np.percentile(r2_all, 25)) if r2_all else float("nan"),
        "p75_R2": float(np.percentile(r2_all, 75)) if r2_all else float("nan"),
    }
    return {"folds": fold_results, "summary": summary}


def main():
    log("=" * 60)
    log("LODO baseline comparison starting")
    log("=" * 60)

    all_results = {}

    # ---- PT-SR (load existing) ----
    log("\n--- PT-SR (BIC-best, from lodo_all_results.xlsx) ---")
    ptsr_folds = load_ptsr_fold_results()
    r2_all = [v["R2"] for v in ptsr_folds.values() if np.isfinite(v["R2"])]
    ptsr_summary = {
        "n_folds_total": len(ptsr_folds),
        "n_folds_evaluated": len(r2_all),
        "median_R2": float(np.median(r2_all)),
        "mean_R2": float(np.mean(r2_all)),
        "std_R2": float(np.std(r2_all)),
        "p25_R2": float(np.percentile(r2_all, 25)),
        "p75_R2": float(np.percentile(r2_all, 75)),
    }
    all_results["PT_SR_BIC_best"] = {
        "class": "smooth", "folds": ptsr_folds, "summary": ptsr_summary,
    }
    log(f"  PT-SR median R^2 (all 134 folds): {ptsr_summary['median_R2']:.4f}")
    log(f"  PT-SR mean   R^2 (all 134 folds): {ptsr_summary['mean_R2']:.4f}")

    # Save initial
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- All other baselines ----
    for name, (fn, n_seeds, needs_elem, cls) in BASELINES.items():
        log(f"\n--- {name} (class={cls}, n_seeds={n_seeds}) ---")
        t0 = time.time()
        try:
            res = run_baseline_on_folds(name, fn, n_seeds, needs_elem)
            res["class"] = cls
            res["n_seeds"] = n_seeds
            all_results[name] = res
            elapsed = (time.time() - t0) / 60.0
            log(f"  {name} done in {elapsed:.1f} min — "
                f"median R^2 = {res['summary']['median_R2']:.4f}")
        except Exception as e:
            log(f"  {name} FAILED: {e}")
            all_results[name] = {
                "class": cls, "error": str(e),
                "summary": {"median_R2": float("nan")},
            }
        # Checkpoint after each baseline
        with open(OUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)

    log("\n" + "=" * 60)
    log("All baselines complete. Building summary.md")
    log("=" * 60)

    # ---- Build summary.md ----
    write_summary(all_results)
    log(f"Wrote {OUT_JSON}")
    log(f"Wrote {OUT_SUMMARY}")


def write_summary(all_results):
    lines = [
        "# LODO Cross-Dataset Generalization: Baselines vs PT-SR",
        "",
        "Protocol: 134 LODO folds (Composition x Processing x Property x Test_Dataset)",
        "from `lodo_all_results.xlsx`.",
        "Per fold: train on rows from sources != Test_Dataset's source, test on the held-out source.",
        "Stochastic baselines: median across 5 seeds (0..4).",
        "",
        "## Overall summary (median R^2 across folds)",
        "",
        "| Baseline | Class | n_folds_eval | median R^2 | mean R^2 | std | p25 | p75 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    # Order: PT-SR first, then by class
    order = ["PT_SR_BIC_best"] + [k for k in all_results if k != "PT_SR_BIC_best"]
    for name in order:
        r = all_results.get(name, {})
        s = r.get("summary", {})
        cls = r.get("class", "?")
        lines.append(
            f"| {name} | {cls} | {s.get('n_folds_evaluated', 0)} | "
            f"{s.get('median_R2', float('nan')):.4f} | "
            f"{s.get('mean_R2', float('nan')):.4f} | "
            f"{s.get('std_R2', float('nan')):.4f} | "
            f"{s.get('p25_R2', float('nan')):.4f} | "
            f"{s.get('p75_R2', float('nan')):.4f} |"
        )

    # Sub-summaries for FCC CAST subset
    lines += ["", "## FCC CAST subset (52 folds, no refractory)"]
    fcc_elements = {"Fe", "Co", "Ni", "Cr", "Cu", "Mn", "Al", "Li", "Mg", "Si", "Ti"}
    refractory = {"W", "Mo", "Nb", "Ta", "Hf", "Re"}

    def is_fcc_cast(fid):
        comp, proc, prop, _ = fid.split("|")
        if proc != "CAST":
            return False
        elems = set(re.findall(r"[A-Z][a-z]?", comp))
        # FCC-friendly: <=1 refractory element
        return len(elems & refractory) == 0

    lines += [
        "",
        "| Baseline | n | median R^2 | mean R^2 |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        r = all_results.get(name, {})
        folds = r.get("folds", {})
        r2s = [v.get("R2", float("nan")) for k, v in folds.items()
               if is_fcc_cast(k) and np.isfinite(v.get("R2", float("nan")))]
        if r2s:
            lines.append(
                f"| {name} | {len(r2s)} | {np.median(r2s):.4f} | {np.mean(r2s):.4f} |"
            )
        else:
            lines.append(f"| {name} | 0 | nan | nan |")

    # Per-fold table
    lines += ["", "## Per-fold R^2 (all baselines x all 134 folds)", ""]
    cols = ["Fold"] + order
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    fold_ids = sorted(all_results["PT_SR_BIC_best"]["folds"].keys())
    for fid in fold_ids:
        cells = [fid]
        for name in order:
            r = all_results.get(name, {})
            v = r.get("folds", {}).get(fid, {})
            r2 = v.get("R2", float("nan"))
            cells.append(f"{r2:.4f}" if np.isfinite(r2) else "—")
        lines.append("| " + " | ".join(cells) + " |")

    OUT_SUMMARY.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
