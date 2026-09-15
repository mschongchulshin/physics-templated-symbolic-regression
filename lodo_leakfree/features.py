"""
Step 2: Common feature builder used by ALL methods (PT-SR + 9 baselines).

For a fold (composition, processing, property, test_source):
  - Filter MPEA rows for elem_set + processing + non-null y
  - For each row, compute features:
      * element fractions (sorted alphabetic, length = #elements in comp system)
      * Hume-Rothery: S_mix, dH_mix, delta, VEC, dChi (+ r_avg, chi_avg)
      * T (Kelvin)
  - Split: rows from test_source -> test, others -> train
  - Returns dict with X_train, y_train, X_test, y_test, T_train, T_test,
    elem_order, formulas_train, formulas_test

NOTE: PT-SR also needs `inv_T = 1/T` and `S_mix` — they are part of features.
"""
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd

BASE = str(Path(__file__).resolve().parent.parent)
OUT = f"{BASE}/results/generated/lodo_v2"

# ---- Element properties (Miedema) - copied from baselines/lodo_comparison/run.py ----
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
    x = np.asarray(fractions, dtype=float)
    p = np.clip(x, 1e-12, 1.0)
    s_mix = -R_GAS * np.sum(p * np.log(p))
    h = 0.0
    for i, ei in enumerate(elements):
        for j in range(i + 1, len(elements)):
            h += 4.0 * _omega(ei, elements[j]) * x[i] * x[j]
    h_mix = h
    r = np.array([ELEM_R.get(e, 130.0) for e in elements])
    r_avg = float(np.dot(x, r))
    delta = float(np.sqrt(np.sum(x * (1.0 - r / r_avg) ** 2)))
    vec_arr = np.array([ELEM_VEC.get(e, 6.0) for e in elements], dtype=float)
    vec = float(np.dot(x, vec_arr))
    chi = np.array([ELEM_CHI.get(e, 1.6) for e in elements])
    chi_avg = float(np.dot(x, chi))
    dchi = float(np.sqrt(np.sum(x * (chi - chi_avg) ** 2)))
    return s_mix, h_mix, delta, vec, dchi, r_avg, chi_avg


def parse_formula(formula):
    if pd.isna(formula):
        return None
    matches = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", str(formula).strip())
    out = {}
    for el, amt in matches:
        if el:
            out[el] = float(amt) if amt else 1.0
    return out if out else None


PROP_COL = {
    "YS": "PROPERTY: YS (MPa)",
    "Elongation": "PROPERTY: Elongation (%)",
}


def load_mpea():
    """Load preprocessed MPEA cache (or original if not present)."""
    cache = f"{OUT}/mpea_preprocessed.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        return df
    # fallback
    df = pd.read_csv(f"{BASE}/data/LODO_experimental_dataset.csv")
    df["_parsed"] = df["FORMULA"].apply(parse_formula)
    df["_elem_set"] = df["_parsed"].apply(
        lambda p: "".join(sorted(p.keys())) if p else None
    )
    return df


def build_fold(comp, processing, prop, test_source, mpea):
    """Build train/test feature arrays for a fold.

    Returns dict with all features OR None if fold not buildable.
    """
    elem_order = sorted(set(re.findall(r"[A-Z][a-z]?", comp)))
    elem_set = "".join(elem_order)
    ycol = PROP_COL[prop]

    mask = (
        (mpea["_elem_set"] == elem_set)
        & (mpea["PROPERTY: Processing method"] == processing)
        & (mpea[ycol].notna())
    )
    sub = mpea[mask].copy()
    if len(sub) == 0:
        return None

    feats_full = []
    ys = []
    sources = []
    formulas = []
    Ts = []

    for _, row in sub.iterrows():
        parsed = parse_formula(row["FORMULA"])
        if parsed is None:
            continue
        amts = np.array([parsed.get(el, 0.0) for el in elem_order])
        s = amts.sum()
        if s == 0:
            continue
        x = amts / s
        s_mix, h_mix, delta, vec, dchi, r_avg, chi_avg = hume_rothery_features(elem_order, x)
        T_k = float(row["_T_K"])
        # feature order: [x_e1, x_e2, ..., x_eN, s_mix, h_mix, delta, vec, dchi, r_avg, chi_avg, T]
        feat = list(x) + [s_mix, h_mix, delta, vec, dchi, r_avg, chi_avg, T_k]
        feats_full.append(feat)
        ys.append(float(row[ycol]))
        sources.append(row["_source"])
        formulas.append(row["FORMULA"])
        Ts.append(T_k)

    if len(feats_full) == 0:
        return None

    X = np.array(feats_full, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    src = np.array(sources)
    formulas = np.array(formulas, dtype=object)
    Ts = np.array(Ts, dtype=np.float64)

    test_mask = src == test_source
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    n_elem = len(elem_order)
    feat_names = (
        [f"x_{e}" for e in elem_order]
        + ["S_mix", "H_mix", "delta", "VEC", "dChi", "r_avg", "chi_avg", "T"]
    )

    return {
        "elem_order": elem_order,
        "n_elem": n_elem,
        "feat_names": feat_names,
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "T_train": Ts[train_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "T_test": Ts[test_mask],
        "formulas_train": formulas[train_mask],
        "formulas_test": formulas[test_mask],
        "src_train": src[train_mask],
        "src_test": src[test_mask],
    }


if __name__ == "__main__":
    # smoke test
    mpea = load_mpea()
    folds = pd.read_csv(f"{OUT}/folds.csv")
    ok = folds[folds["status"].str.startswith("ok")]
    print(f"Testing on {len(ok)} ok folds...")
    for _, r in ok.head(3).iterrows():
        d = build_fold(r["composition"], r["processing"], r["property"], r["test_source"], mpea)
        if d is None:
            print(f"  {r['fold_id']}: FAILED")
            continue
        # sanity check
        assert d["X_train"].shape[1] == d["n_elem"] + 8
        assert d["X_train"].shape[0] == r["n_train"], f"n_train mismatch: {d['X_train'].shape[0]} vs {r['n_train']}"
        assert d["X_test"].shape[0] == r["n_test"], f"n_test mismatch: {d['X_test'].shape[0]} vs {r['n_test']}"
        print(f"  {r['fold_id']}: train={d['X_train'].shape}, test={d['X_test'].shape}, feat_dim={d['X_train'].shape[1]} -> OK")
    print("Smoke test OK")
