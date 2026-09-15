"""LODO folds for every property in the Borg MPEA dataset, with the same rules as the yield strength folds.

Properties: YS (verified rows, verified_V.pkl), and UTS, elongation, HV, density, modulus taken from the raw
dataset (data/LODO_experimental_dataset.csv). The raw values are NOT yet checked against the original papers, only YS is.

Rules, identical for every property:
  cast material only, one alloy system per pool (element set S, or S with one element at zero)
  exact repeats inside one paper (same composition, temperature, test type) are averaged into one row
  fold = (pool, held-out paper); training rows that duplicate a test condition (composition within 1 at.%
         total variation and temperature within 25 C) are removed, which is the leak the manuscript had
  test >= 3 rows, train >= 5 rows and >= 3 compositions after the leak rows are removed
Writes lodo_V_<prop>.pkl and lodo_folds_<prop>.pkl (yield strength keeps the names lodo_V.pkl, lodo_folds.pkl).
usage: python make_lodo_folds_all.py
"""
from pathlib import Path
import os, sys, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_lodo_folds as MLF

RAW = str(Path(__file__).resolve().parent.parent / "data"
          / "LODO_experimental_dataset.csv")
PROPS = {
    "UTS": ("PROPERTY: UTS (MPa)", True),
    "elongation": ("PROPERTY: Elongation (%)", True),
    "HV": ("PROPERTY: HV", False),
    "density": ("PROPERTY: Exp. Density (g/cm$^3$)", False),
    "modulus": ("PROPERTY: Exp. Young modulus (GPa)", False),
}


def parse(f):
    v = {}
    for e, n in re.findall(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)", str(f)):
        if e:
            v[e] = v.get(e, 0.0) + (float(n) if n else 1.0)
    s = sum(v.values())
    return {k: x / s for k, x in v.items() if x > 0}


def raw_frame(col, has_tt):
    D = pd.read_csv(RAW)
    D = D[(D["PROPERTY: Processing method"] == "CAST") & D[col].notna()].copy()
    D["vec"] = D.FORMULA.map(parse)
    D["T"] = D["PROPERTY: Test temperature ($^\\circ$C)"].fillna(25.0)
    D["test_type"] = D["PROPERTY: Type of test"].fillna("-") if has_tt else "-"
    D = D.rename(columns={col: "YS_MPa", "IDENTIFIER: Reference ID": "ref_id", "FORMULA": "formula",
                          "REFERENCE: doi": "doi", "REFERENCE: year": "year"})
    D["processing"] = "CAST"
    D["lab_name"] = "ref" + D.ref_id.astype(str) + " (" + D.year.astype(str) + ")"
    D = D[["ref_id", "doi", "year", "formula", "processing", "test_type", "T", "YS_MPa", "vec", "lab_name"]].reset_index(drop=True)
    D["ck"] = D.vec.apply(lambda v: tuple(sorted((k, round(100 * x)) for k, x in v.items())))
    D["T_i"] = D["T"].round().astype(int)
    g = D.groupby([D.lab_name, D.ck.map(str), D.T_i, D.test_type], as_index=False)
    D = g.apply(lambda d: d.assign(YS_MPa=d.YS_MPa.mean()).iloc[:1], include_groups=True).reset_index(drop=True)
    D["els"] = D.vec.apply(frozenset)
    return D


if __name__ == "__main__":
    summary = []
    for name, (col, has_tt) in PROPS.items():
        V = raw_frame(col, has_tt)
        F = MLF.folds(V)
        use = [f for f in F if f["usable"]]
        V.to_pickle(f"{HERE}/lodo_V_{name}.pkl"); pd.to_pickle(use, f"{HERE}/lodo_folds_{name}.pkl")
        summary.append(dict(property=name, rows=len(V), papers=V.lab_name.nunique(), candidate_folds=len(F), usable_folds=len(use)))
        print(f"\n=== {name}: {len(V)} cast rows, {len(use)} usable folds of {len(F)}")
        if use:
            T = pd.DataFrame([{k: f[k] for k in ["fold_id", "n_train", "n_test", "train_comps", "test_comps",
                                                 "leak_rows_removed", "T_train", "T_test", "y_train", "y_test", "y_test_sd"]} for f in use])
            pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 48)
            print(T.assign(fold_id=T.fold_id.str.replace("|CAST|", "|", regex=False)).to_string(index=False))
    print("\n" + pd.DataFrame(summary).to_string(index=False))
