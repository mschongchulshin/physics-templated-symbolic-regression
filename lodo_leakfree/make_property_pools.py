"""Repeat-free CAST pools for properties other than yield strength, using the rules of the YS dataset.

Source: the raw Borg MPEA dataset (data/LODO_experimental_dataset.csv), CAST rows only. Values are NOT yet checked against the
original papers, unlike the YS dataset. This is a screening step, the property that works gets verified after.

Rules, identical to make_norepeat.py:
1. repeats: every (composition rounded to 1 at.%, test temperature, test type) group with two or more rows is dropped
2. zigzag series: within one paper, one temperature and one test type, compositions ordered by the element that
   varies most; if the property changes direction two or more times the whole series is dropped
3. pools: one alloy system S (rows whose element set is S, or S with one element at zero), at least 6 rows,
   4 compositions and 3 full-system rows, nested subsystems dropped

Writes norepeat_<property>.pkl with the same columns as norepeat_V.pkl (target column named YS_MPa so the
existing LOCO code runs unchanged) and prints the pool table for every property.
usage: python make_property_pools.py
"""
from pathlib import Path
import os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = str(Path(__file__).resolve().parent.parent / "data"
          / "LODO_experimental_dataset.csv")
PROPS = {
    "HV": ("PROPERTY: HV", False),                                  # hardness, no test type, room temperature
    "UTS": ("PROPERTY: UTS (MPa)", True),
    "elongation": ("PROPERTY: Elongation (%)", True),
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


def build(name, col, has_test_type):
    D = pd.read_csv(RAW)
    D = D[(D["PROPERTY: Processing method"] == "CAST") & D[col].notna()].copy()
    D["vec"] = D.FORMULA.map(parse)
    D["T"] = D["PROPERTY: Test temperature ($^\\circ$C)"].fillna(25.0)
    D["test_type"] = D["PROPERTY: Type of test"].fillna("-") if has_test_type else "-"
    D = D.rename(columns={col: "YS_MPa", "IDENTIFIER: Reference ID": "ref_id", "FORMULA": "formula",
                          "REFERENCE: doi": "doi", "REFERENCE: year": "year"})
    D["processing"] = "CAST"
    D["lab_name"] = "ref" + D.ref_id.astype(str) + " (" + D.year.astype(str) + ")"
    D = D[["ref_id", "doi", "year", "formula", "processing", "test_type", "T", "YS_MPa", "vec", "lab_name"]].reset_index(drop=True)

    ck = D.vec.apply(lambda v: str(tuple(sorted((k, round(100 * x)) for k, x in v.items()))))
    n = D.groupby([ck, D["T"].round().astype(int), D.test_type]).YS_MPa.transform("size")
    R = D[n == 1].copy()
    zig = []
    for (ref, tt, T), g in R.groupby(["ref_id", "test_type", R["T"].round()]):
        if len(g) < 3:
            continue
        els = sorted(set().union(*g.vec.apply(set)))
        M = np.array([[v.get(e, 0.0) for e in els] for v in g.vec])
        o = np.argsort(M[:, M.std(0).argmax()])
        d = np.sign(np.diff(g.YS_MPa.values[o]))
        if len(d) > 1 and int((d[1:] * d[:-1] < 0).sum()) >= 2:
            zig += g.index.tolist()
    R = R.drop(index=zig).reset_index(drop=True)
    R.to_pickle(f"{HERE}/norepeat_{name}.pkl")

    R["els"] = R.vec.apply(lambda v: frozenset(v)); R["ckey"] = R.vec.apply(lambda v: tuple(sorted((k, round(100 * x)) for k, x in v.items())))
    out = []
    for tt, G0 in R.groupby("test_type"):
        for S in sorted(set(G0.els), key=lambda s: (-len(s), "".join(sorted(s)))):
            if len(S) < 3:
                continue
            G = G0[G0.els.apply(lambda e: e <= S and len(S - e) <= 1)]
            if len(G) < 6 or G.ckey.nunique() < 4 or int((G.els == S).sum()) < 3:
                continue
            if any(o["tt"] == tt and S < o["S"] for o in out):
                continue
            out.append(dict(tt=tt, S=S, property=name, system="".join(sorted(S)), test=tt, rows=len(G),
                            compositions=G.ckey.nunique(), papers=G.ref_id.nunique(),
                            value=f"{G.YS_MPa.min():.0f}-{G.YS_MPa.max():.0f}", SD=round(float(G.YS_MPa.std())),
                            T=f"{int(G['T'].min())}-{int(G['T'].max())}"))
    print(f"\n=== {name}: CAST rows {len(D)} -> repeat-free {len(R)} ({len(zig)} zigzag rows dropped), pools {len(out)}")
    if out:
        print(pd.DataFrame(out).drop(columns=["tt", "S"]).to_string(index=False))
    return out


if __name__ == "__main__":
    for name, (col, tt) in PROPS.items():
        build(name, col, tt)
