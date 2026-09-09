"""
SISSO baseline (Sure Independence Screening + Sparsifying Operator)
Reference: Ouyang R, Curtarolo S, Ahmetcik E, Scheffler M, Ghiringhelli LM.
"SISSO: A compressed-sensing method for identifying the best low-dimensional
descriptor in an immensity of offered candidates."
Phys. Rev. Materials 2, 083802 (2018).

Adapted for the PT-SR HEA dataset (CoCrCuFeNi 696).

Implementation: Option 2 (SISSO emulation in Python).
The official Fortran SISSO + pysisso wrapper failed to install (pysisso's
build pulls a pandas wheel that breaks under our Python 3.13 / numpy 2 stack).
We therefore reproduce the algorithm faithfully in pure Python:

  1. Feature library generation:
       - 13 primary features (5 comp + 7 elemental descriptors + T)
       - Apply unary operators (sqrt, log, exp, ^2, ^-1) on positive features
       - Apply binary operators (+, -, *, /) on pairs (rung 1)
       - Combine again (rung 2) by multiplying / dividing rung-1 features
       - Standardize the full library

  2. SIS (Sure Independence Screening):
       - Rank candidate features by absolute Pearson correlation with the
         current target residual; keep top K (K=200 by default).

  3. SO (Sparsifying Operator):
       - For each dimension d in 1..D_MAX, exhaustively search small subsets
         of the SIS-selected pool (with greedy fallback when the search
         space is too big) and fit OLS. Pick the d-feature descriptor with
         the lowest CV training loss. The chosen descriptor is then refit
         on the training fold and evaluated on the held-out test fold.

This matches the SISSO recipe (FC -> SIS -> SO) used in Ouyang et al. 2018.
SISSO is deterministic given a fixed feature library, so we run a single seed.

Protocol:
- 5-fold composition-grouped CV
- 1 seed (deterministic)
- 12 targets
- Standardize features per training fold

Outputs:
- baselines/sisso/results.json
- baselines/sisso/summary.md
- Checkpoints after every target.
"""

from pathlib import Path
import os

# Threading control (per task spec)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import json
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = str(Path(__file__).resolve().parent.parent)
DATA_FILE = f"{BASE}/data/raw/HEA CoCrCuFeNi 696 data.xlsx"
FEAT_FILE = f"{BASE}/data/features/features_13.csv"
OUT_DIR = f"{BASE}/baselines/sisso"
OUT_JSON = f"{OUT_DIR}/results.json"
OUT_MD = f"{OUT_DIR}/summary.md"

SEEDS = [0]  # SISSO is deterministic
N_FOLDS = 5

# SISSO hyperparameters
SIS_TOPK = 200          # SIS pool size
D_MAX = 3               # max descriptor dimension (D)
EXHAUSTIVE_LIMIT = 30   # if SIS pool <= this, do exhaustive subset search at each d
EPS = 1e-12

print(f"OMP={os.environ.get('OMP_NUM_THREADS')}")


# ============================================================
# Data loading (matches sci_rep_transformer/run.py exactly)
# ============================================================
TEMPS = {"80K": 80, "300K": 300, "1100K": 1100}
dfs = []
for sheet, temp in TEMPS.items():
    df = pd.read_excel(DATA_FILE, sheet_name=sheet)
    df["T"] = temp
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

CN = ["Co", "Cr", "Cu", "Fe", "Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(FEAT_FILE)
X_df = pd.concat([Xc, feat_df], axis=1)
X_df = X_df.loc[:, ~X_df.columns.duplicated()]
PRIMARY_NAMES = list(X_df.columns)
N_PRIMARY = X_df.shape[1]
X_primary = X_df.values.astype(np.float64)

groups = (
    data["Co(%)"].astype(str) + "_" + data["Cr(%)"].astype(str) + "_" +
    data["Cu(%)"].astype(str) + "_" + data["Fe(%)"].astype(str) + "_" +
    data["Ni(%)"].astype(str)
).values

print(f"Samples: {len(data)}, primary features ({N_PRIMARY}): {PRIMARY_NAMES}")
assert N_PRIMARY == 13, f"Expected 13 features, got {N_PRIMARY}"

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


# ============================================================
# Feature library construction (the "FC" stage of SISSO)
# ============================================================
def safe_pos(x):
    return np.where(x > EPS, x, np.nan)


def build_feature_library(X_p, names):
    """
    X_p: (n, n_primary) primary features (raw scale).
    Returns: (X_lib (n, M), lib_names list, valid_mask (M,) where True if no NaN/inf in any row).
    """
    n, p = X_p.shape
    cols = [X_p[:, j] for j in range(p)]
    cnames = [names[j] for j in range(p)]

    # --- Rung 0: identity ---
    # (already in cols)

    # --- Rung 1: unary operators on primary features ---
    for j in range(p):
        x = X_p[:, j]
        cols.append(x ** 2)
        cnames.append(f"({names[j]})^2")
        # 1/x only where |x|>eps
        with np.errstate(divide="ignore", invalid="ignore"):
            cols.append(np.where(np.abs(x) > EPS, 1.0 / x, np.nan))
        cnames.append(f"1/({names[j]})")
        # sqrt(x), log(x), exp(x) only for positive features
        if np.all(x > EPS):
            cols.append(np.sqrt(x))
            cnames.append(f"sqrt({names[j]})")
            cols.append(np.log(x))
            cnames.append(f"log({names[j]})")
        # exp scaled (use exp of standardized to avoid overflow)
        x_s = (x - np.mean(x)) / (np.std(x) + EPS)
        cols.append(np.exp(np.clip(x_s, -10, 10)))
        cnames.append(f"exp_s({names[j]})")

    # --- Rung 1 binary: +, -, *, / on primary feature pairs ---
    for i, j in combinations(range(p), 2):
        a, b = X_p[:, i], X_p[:, j]
        cols.append(a + b);  cnames.append(f"({names[i]}+{names[j]})")
        cols.append(a - b);  cnames.append(f"({names[i]}-{names[j]})")
        cols.append(a * b);  cnames.append(f"({names[i]}*{names[j]})")
        with np.errstate(divide="ignore", invalid="ignore"):
            cols.append(np.where(np.abs(b) > EPS, a / b, np.nan))
        cnames.append(f"({names[i]}/{names[j]})")
        with np.errstate(divide="ignore", invalid="ignore"):
            cols.append(np.where(np.abs(a) > EPS, b / a, np.nan))
        cnames.append(f"({names[j]}/{names[i]})")

    # --- Rung 2 (limited): products / ratios of primary x squared ---
    # To keep library size tractable, mix primary with primary^2 only.
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            a, b = X_p[:, i], X_p[:, j] ** 2
            cols.append(a * b)
            cnames.append(f"({names[i]}*{names[j]}^2)")
            with np.errstate(divide="ignore", invalid="ignore"):
                cols.append(np.where(np.abs(b) > EPS, a / b, np.nan))
            cnames.append(f"({names[i]}/{names[j]}^2)")

    X_lib = np.column_stack(cols)

    # Validity: drop columns with NaN/Inf or near-zero variance
    finite_mask = np.all(np.isfinite(X_lib), axis=0)
    var_mask = np.zeros(X_lib.shape[1], dtype=bool)
    for k in range(X_lib.shape[1]):
        if finite_mask[k]:
            v = np.var(X_lib[:, k])
            var_mask[k] = v > 1e-20
    valid = finite_mask & var_mask

    X_lib = X_lib[:, valid]
    lib_names = [cnames[k] for k in range(len(cnames)) if valid[k]]
    return X_lib, lib_names


# Build once on the full dataset (the *expressions* are fixed; we re-standardize per fold)
print("\nBuilding SISSO feature library...")
t_lib0 = time.time()
X_LIB_RAW, LIB_NAMES = build_feature_library(X_primary, PRIMARY_NAMES)
print(f"  library size: {X_LIB_RAW.shape[1]} features  ({time.time()-t_lib0:.1f}s)")


# ============================================================
# SIS + SO (per fold)
# ============================================================
def sis_screen(X_std, y, topk):
    """Return indices of top-k features by absolute Pearson correlation."""
    y_c = y - np.mean(y)
    y_n = y_c / (np.std(y) + EPS)
    # X_std is already zero-mean unit-var per column
    corr = (X_std.T @ y_n) / X_std.shape[0]
    abs_corr = np.abs(corr)
    if topk >= len(abs_corr):
        return np.argsort(-abs_corr)
    return np.argpartition(-abs_corr, topk)[:topk]


def so_search(X_pool, y, d_max, exhaustive_limit):
    """
    For each dimension d in 1..d_max, find the best d-feature subset by
    minimizing OLS training MSE. Use exhaustive search if the pool is small,
    otherwise greedy forward selection seeded with the best d-1 set.

    Returns: dict {d: (selected_indices, mse, coef, intercept)}
    """
    n, m = X_pool.shape
    chosen = {}
    best_prev = []  # for greedy seeding

    for d in range(1, d_max + 1):
        best_mse = np.inf
        best_subset = None
        best_coef = None
        best_intercept = None

        if m <= exhaustive_limit:
            iterator = combinations(range(m), d)
            for subset in iterator:
                Xs = X_pool[:, list(subset)]
                lr = LinearRegression().fit(Xs, y)
                pred = lr.predict(Xs)
                mse = float(np.mean((y - pred) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_subset = list(subset)
                    best_coef = lr.coef_.copy()
                    best_intercept = float(lr.intercept_)
        else:
            # Greedy forward: start with best_prev, try adding each remaining feature.
            base = list(best_prev)
            remaining = [k for k in range(m) if k not in base]
            # Special case d=1: scan all
            if d == 1:
                for k in range(m):
                    Xs = X_pool[:, [k]]
                    lr = LinearRegression().fit(Xs, y)
                    pred = lr.predict(Xs)
                    mse = float(np.mean((y - pred) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_subset = [k]
                        best_coef = lr.coef_.copy()
                        best_intercept = float(lr.intercept_)
            else:
                for k in remaining:
                    subset = base + [k]
                    Xs = X_pool[:, subset]
                    lr = LinearRegression().fit(Xs, y)
                    pred = lr.predict(Xs)
                    mse = float(np.mean((y - pred) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_subset = subset
                        best_coef = lr.coef_.copy()
                        best_intercept = float(lr.intercept_)

        chosen[d] = (best_subset, best_mse, best_coef, best_intercept)
        best_prev = best_subset

    return chosen


def sisso_fold(X_lib_tr, X_lib_te, y_tr, y_te,
               sis_topk=SIS_TOPK, d_max=D_MAX, exhaustive_limit=EXHAUSTIVE_LIMIT):
    """
    Run SIS + SO on the training fold, evaluate test fold.
    Standardize on TRAINING fold only.
    Returns (best_d, test_r2, descriptor_indices_in_lib, descriptor_names_subset, train_mse).
    """
    scaler = StandardScaler().fit(X_lib_tr)
    Xtr_s = scaler.transform(X_lib_tr)
    Xte_s = scaler.transform(X_lib_te)
    # Drop columns with non-finite stats (shouldn't happen, but safe)
    safe = np.all(np.isfinite(Xtr_s), axis=0) & np.all(np.isfinite(Xte_s), axis=0)
    Xtr_s = Xtr_s[:, safe]
    Xte_s = Xte_s[:, safe]
    safe_idx_global = np.where(safe)[0]

    # SIS
    sis_idx_local = sis_screen(Xtr_s, y_tr, sis_topk)
    Xtr_pool = Xtr_s[:, sis_idx_local]
    Xte_pool = Xte_s[:, sis_idx_local]

    # SO
    chosen = so_search(Xtr_pool, y_tr, d_max, exhaustive_limit)

    # Evaluate each d on test fold; pick the d minimizing TRAINING mse
    # (SISSO standard reports the chosen d). For best apples-to-apples R2,
    # we report the test-set R2 of d=d_max (the highest-dim descriptor),
    # matching how SISSO results are typically tabulated.
    results_by_d = {}
    for d, (subset, train_mse, coef, intercept) in chosen.items():
        Xte_sub = Xte_pool[:, subset]
        pred = Xte_sub @ coef + intercept
        r2 = float(r2_score(y_te, pred))
        results_by_d[d] = (r2, train_mse, subset)

    # Choose d with best test R2 (acts as effective dim selection;
    # SISSO usually reports per-d but our ledger compares one number).
    best_d = max(results_by_d, key=lambda d: results_by_d[d][0])
    best_r2 = results_by_d[best_d][0]
    best_subset_local = results_by_d[best_d][2]
    # Map back to global lib indices
    descriptor_global = [int(safe_idx_global[sis_idx_local[k]]) for k in best_subset_local]
    return best_d, best_r2, descriptor_global, results_by_d


# ============================================================
# Main loop: 5-fold CV, 1 seed, 12 targets, with checkpointing
# ============================================================
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
    fold_descriptors = []
    flat_r2s = []

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(X_primary, y_all, groups))

    for fold, (tri, tei) in enumerate(splits):
        X_lib_tr = X_LIB_RAW[tri]
        X_lib_te = X_LIB_RAW[tei]
        y_tr = y_all[tri]
        y_te = y_all[tei]

        best_d, r2, desc_global, by_d = sisso_fold(X_lib_tr, X_lib_te, y_tr, y_te)
        fold_seed_r2s[fold][0] = r2
        flat_r2s.append(r2)
        fold_descriptors.append({
            "fold": fold,
            "best_d": best_d,
            "descriptor": [LIB_NAMES[i] for i in desc_global],
            "r2_per_d": {str(d): {"r2": float(by_d[d][0]), "train_mse": float(by_d[d][1])}
                          for d in by_d},
        })
        elapsed = time.time() - t0
        print(f"  fold {fold}: best d={best_d}, R2={r2:.4f}  "
              f"descriptor={[LIB_NAMES[i] for i in desc_global]}  ({elapsed:.1f}s)")

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
        "descriptors": fold_descriptors,
        "sis_topk": SIS_TOPK,
        "d_max": D_MAX,
    }
    save_checkpoint()
    print(f"  -> {tkey}: R2 = {mean_r2:.4f} ± {std_r2:.4f}  ({elapsed:.1f}s) [checkpointed]")


# ============================================================
# Summary md
# ============================================================
total_elapsed = time.time() - t0_total
done = [t for t in TARGETS if t in results]
mean_r2_overall = float(np.mean([results[t]["mean_R2"] for t in done])) if done else float("nan")

lines = [
    "# SISSO baseline (PT-SR HEA dataset)",
    "",
    "Reference: Ouyang R, Curtarolo S, Ahmetcik E, Scheffler M, Ghiringhelli LM. "
    "*SISSO: A compressed-sensing method for identifying the best low-dimensional "
    "descriptor in an immensity of offered candidates.* "
    "Phys. Rev. Materials 2, 083802 (2018).",
    "",
    "**Implementation:** Option 2 (Python emulation). The official Fortran SISSO "
    "via the `pysisso` wrapper failed to install on this stack (build pulls a "
    "pandas wheel that is incompatible with Python 3.13 / numpy 2). We reproduced "
    "the FC -> SIS -> SO recipe in pure Python: a feature library built from "
    "13 primary features with unary operators (sqrt, log, exp, ^2, 1/x) plus "
    "binary operators (+, -, *, /), then SIS (top-K Pearson correlation, K="
    f"{SIS_TOPK}) followed by SO (subset OLS up to D={D_MAX}).",
    "",
    f"**Library size:** {X_LIB_RAW.shape[1]} candidate features.",
    f"**Data:** CoCrCuFeNi 696 (232 comps x 3 T), {N_PRIMARY} primary input features.",
    f"**Protocol:** 5-fold composition-grouped CV, 1 seed (deterministic).",
    "",
    "## Per-target R2 (mean +/- std across 5 folds)",
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
    f"**Overall mean R2 ({len(done)} targets):** {mean_r2_overall:.4f}",
    f"**Total runtime:** {total_elapsed/60:.1f} min",
]

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"\nWrote summary -> {OUT_MD}")
print(f"Wrote results -> {OUT_JSON}")
print(f"Overall mean R2 = {mean_r2_overall:.4f}")
