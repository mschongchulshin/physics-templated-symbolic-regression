"""
Step 1: Build verified LODO fold definitions from MPEA data.

For each (composition_system, processing, property) tuple, enumerate every
possible source as test set. Compute n_train, n_test from REAL MPEA row counts.
Mark folds with overlap (FORMULA, T) between train/test, and y_test_constant.

Output: lodo/folds.csv

Schema: composition, processing, property, test_source, train_sources,
        n_train, n_test, train_source_ids, test_source_ids,
        has_overlap, y_test_constant, status
"""
from pathlib import Path
import os
import re
import json
import numpy as np
import pandas as pd

BASE = str(Path(__file__).resolve().parent.parent)
OUT = f"{BASE}/results/generated/lodo_v2"
os.makedirs(OUT, exist_ok=True)

MPEA = pd.read_csv(f"{BASE}/data/LODO_experimental_dataset.csv")

# ---- Source mapping ----
def map_source(rid):
    if pd.isna(rid):
        return "Unknown"
    rid = int(rid)
    if rid <= 66:
        return "Borg"
    if rid <= 97:
        return "Couzinie"
    if rid <= 263:
        return "Iyer"
    return "Gorsse"


def parse_formula(formula):
    if pd.isna(formula):
        return None
    matches = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", str(formula).strip())
    out = {}
    for el, amt in matches:
        if el:
            out[el] = float(amt) if amt else 1.0
    return out if out else None


def to_kelvin(t_celsius):
    if pd.isna(t_celsius):
        return 298.15
    try:
        return float(t_celsius) + 273.15
    except Exception:
        return 298.15


MPEA["_source"] = MPEA["IDENTIFIER: Reference ID"].apply(map_source)
MPEA["_parsed"] = MPEA["FORMULA"].apply(parse_formula)
MPEA["_elem_set"] = MPEA["_parsed"].apply(
    lambda p: "".join(sorted(p.keys())) if p else None
)
MPEA["_T_K"] = MPEA["PROPERTY: Test temperature ($^\\circ$C)"].apply(to_kelvin)

PROP_COL = {
    "YS": "PROPERTY: YS (MPa)",
    "Elongation": "PROPERTY: Elongation (%)",
}

# the multi-principal-element families scanned for evaluable folds
COMPOSITIONS = [
    "AlCoCrCuFeNi", "AlCoCrFeMnNi", "AlCoCrFeMoNi", "AlCoCrFeNi",
    "AlCoCrFeNiTi", "AlCoCuFeNi", "AlCrFeMnNi", "AlCrFeMoNiTi", "AlCrFeNi",
    "AlCrMoNbTi", "AlCrNbTiV", "AlHfNbTaTiZr", "AlLiMgSi", "AlNbTiVZr",
    "CoCrCuFeNiV", "CoCrFeMnMoNi", "CoCrFeMnNi", "CoCrFeMnNiV",
    "CoCrFeMoNi", "CoCrFeNbNi", "CoCrFeNi", "CoCrNi", "CoCuFeNbNi",
    "CrFeNiTi", "HfMoNbSiTiV", "HfMoNbSiTiZr", "HfMoNbTaTiZr", "HfMoNbTiZr",
    "HfNbSiTiVZr", "HfNbTaTiZr", "HfNbTiVZr", "HfNbTiZr", "MoNbReTaW",
    "MoNbTaTiW", "MoNbTaVW", "MoNbTaW", "MoNbTiVZr", "NbTiVZr", "NbTiZr",
]
PROCESSINGS = ["CAST", "WROUGHT", "ANNEAL"]
PROPERTIES = ["YS", "Elongation"]

print(f"Compositions: {len(COMPOSITIONS)}")
print(f"Processings: {PROCESSINGS}")
print(f"Properties: {PROPERTIES}")
print(f"MPEA rows: {len(MPEA)}")

# ---- Build all candidate folds ----
records = []
for comp in COMPOSITIONS:
    elem_order = sorted(set(re.findall(r"[A-Z][a-z]?", comp)))
    elem_set = "".join(elem_order)

    for proc in PROCESSINGS:
        for prop in PROPERTIES:
            ycol = PROP_COL[prop]
            mask = (
                (MPEA["_elem_set"] == elem_set)
                & (MPEA["PROPERTY: Processing method"] == proc)
                & (MPEA[ycol].notna())
            )
            sub = MPEA[mask].copy()
            if len(sub) == 0:
                continue
            sources_present = sorted(sub["_source"].unique())
            if len(sources_present) < 2:
                # Cannot do LODO: only one source. Record as single_source skip.
                for src in sources_present:
                    records.append({
                        "composition": comp,
                        "processing": proc,
                        "property": prop,
                        "test_source": src,
                        "train_sources": "",
                        "n_train": 0,
                        "n_test": int((sub["_source"] == src).sum()),
                        "train_source_ids": "",
                        "test_source_ids": ",".join(
                            map(str, sorted(sub.loc[sub["_source"] == src, "IDENTIFIER: Reference ID"].astype(int).unique()))
                        ),
                        "has_overlap": False,
                        "y_test_constant": False,
                        "status": "skip_single_source",
                        "skip_reason": "only one source has data",
                    })
                continue
            for test_src in sources_present:
                train_src = [s for s in sources_present if s != test_src]
                tr_mask = sub["_source"].isin(train_src)
                te_mask = sub["_source"] == test_src
                n_tr = int(tr_mask.sum())
                n_te = int(te_mask.sum())

                tr_ids = sorted(
                    sub.loc[tr_mask, "IDENTIFIER: Reference ID"].astype(int).unique()
                )
                te_ids = sorted(
                    sub.loc[te_mask, "IDENTIFIER: Reference ID"].astype(int).unique()
                )

                # Check (FORMULA, T) overlap
                tr_keys = set(zip(sub.loc[tr_mask, "FORMULA"], sub.loc[tr_mask, "_T_K"]))
                te_keys = set(zip(sub.loc[te_mask, "FORMULA"], sub.loc[te_mask, "_T_K"]))
                overlap = bool(tr_keys & te_keys)

                # y_test constant?
                y_te = sub.loc[te_mask, ycol].values
                y_constant = bool(n_te >= 1 and np.std(y_te) < 1e-12)

                status = "ok"
                skip_reason = ""
                if n_tr < 3:
                    status = "skip_small_train"
                    skip_reason = f"n_train={n_tr} < 3"
                elif n_te < 2:
                    status = "skip_small_test"
                    skip_reason = f"n_test={n_te} < 2"
                elif y_constant:
                    status = "ok_constant_y"
                    skip_reason = "y_test is constant -> R2 NaN"

                records.append({
                    "composition": comp,
                    "processing": proc,
                    "property": prop,
                    "test_source": test_src,
                    "train_sources": ",".join(train_src),
                    "n_train": n_tr,
                    "n_test": n_te,
                    "train_source_ids": ",".join(map(str, tr_ids)),
                    "test_source_ids": ",".join(map(str, te_ids)),
                    "has_overlap": overlap,
                    "y_test_constant": y_constant,
                    "status": status,
                    "skip_reason": skip_reason,
                })

folds = pd.DataFrame(records)
folds["fold_id"] = folds.apply(
    lambda r: f"{r['composition']}|{r['processing']}|{r['property']}|{r['test_source']}",
    axis=1,
)
# Reorder
cols = [
    "fold_id", "composition", "processing", "property",
    "test_source", "train_sources",
    "n_train", "n_test",
    "train_source_ids", "test_source_ids",
    "has_overlap", "y_test_constant", "status", "skip_reason",
]
folds = folds[cols]

out_csv = f"{OUT}/folds.csv"
folds.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}: {len(folds)} fold records")
print(folds["status"].value_counts())
print(f"\nValid (ok / ok_constant_y): {(folds['status'].str.startswith('ok')).sum()}")
print(f"With overlap: {folds['has_overlap'].sum()}")
print(f"With constant y_test: {folds['y_test_constant'].sum()}")

print(f"MPEA cache -> {OUT}/mpea_preprocessed.csv")
