import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MIN_TEST, MIN_TRAIN, MIN_TRAIN_COMPS = 3, 5, 3


SPREAD = 0.30


def conflicting(V, verbose=False):
    bad = []
    for (ck, tt), g in V.groupby([V.ck.map(str), V.test_type]):
        g = g.sort_values("T_i")
        cluster, prev = [], None
        for i, r in g.iterrows():
            if prev is not None and r.T_i - prev > 25:
                bad += check(V, cluster, verbose); cluster = []
            cluster.append(i); prev = r.T_i
        bad += check(V, cluster, verbose)
    return bad


def check(V, idx, verbose):
    if len(idx) < 2:
        return []
    g = V.loc[idx]
    if g.lab_name.nunique() < 2:
        return []
    lo, hi, mean = g.YS_MPa.min(), g.YS_MPa.max(), g.YS_MPa.mean()
    if mean <= 0 or (hi - lo) / mean <= SPREAD:
        return []
    if verbose:
        print(f"   dropped, sources disagree: {g.formula.iloc[0]} "
              f"{int(g.T_i.iloc[0])}C {g.test_type.iloc[0]} "
              f"{lo:.0f}-{hi:.0f} ({', '.join(sorted(g.lab_name.unique()))})")
    return list(idx)


def load(verbose=False):
    V = pd.read_pickle(f"{HERE}/verified_V.pkl").reset_index(drop=True)
    V = V[V.processing == "CAST"].copy()
    V["ck"] = V.vec.apply(lambda v: tuple(sorted((k, round(100 * x)) for k, x in v.items())))
    V["T_i"] = V["T"].round().astype(int)
    g = V.groupby([V.lab_name, V.ck.map(str), V.T_i, V.test_type], as_index=False)
    V = g.apply(lambda d: d.assign(YS_MPa=d.YS_MPa.mean()).iloc[:1], include_groups=True).reset_index(drop=True)
    V["els"] = V.vec.apply(frozenset)
    bad = conflicting(V, verbose)
    if verbose:
        print(f"rows dropped where sources disagree: {len(bad)} / {len(V)}")
    return V.drop(index=bad).reset_index(drop=True)


def dist(a, b):
    ks = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in ks) * 100


def folds(V):
    out = []
    for (proc, tt), G0 in V.groupby(["processing", "test_type"]):
        for S in sorted(set(G0.els), key=lambda s: (-len(s), "".join(sorted(s)))):
            if len(S) < 3:
                continue
            G = G0[G0.els.apply(lambda e: e <= S and len(S - e) <= 1)]
            if G.lab_name.nunique() < 2:
                continue
            pool = f"{''.join(sorted(S))}|{proc}|{tt}"
            if any(o["proc"] == proc and o["tt"] == tt and S < o["S"] for o in out):
                continue
            for lab in sorted(G.lab_name.unique()):
                te = G[G.lab_name == lab]; tr = G[G.lab_name != lab]
                leak = []

                why = []
                if len(te) < MIN_TEST: why.append(f"test<{MIN_TEST}")
                if len(tr) < MIN_TRAIN: why.append(f"train<{MIN_TRAIN}")
                if tr.ck.nunique() < MIN_TRAIN_COMPS: why.append(f"train_comps<{MIN_TRAIN_COMPS}")
                if tr.lab_name.nunique() < 2: why.append("train_papers<2")
                out.append(dict(fold_id=f"{pool}||{lab}", pool=pool, S=S, proc=proc, tt=tt, held_out=lab,
                                train=tr.index.tolist(), test=te.index.tolist(), n_train=len(tr), n_test=len(te),
                                leak_rows_removed=len(leak), train_comps=tr.ck.nunique(), test_comps=te.ck.nunique(),
                                train_papers="; ".join(f"{k} {v}" for k, v in tr.lab_name.value_counts().sort_index().items()),
                                T_train=f"{tr['T_i'].min()}-{tr['T_i'].max()}" if len(tr) else "-", T_test=f"{te['T_i'].min()}-{te['T_i'].max()}",
                                y_train=f"{tr.YS_MPa.min():.0f}-{tr.YS_MPa.max():.0f}" if len(tr) else "-",
                                y_test=f"{te.YS_MPa.min():.0f}-{te.YS_MPa.max():.0f}", y_test_sd=round(float(np.nan_to_num(te.YS_MPa.std()))),
                                usable=not why, fail=", ".join(why)))
    return out


if __name__ == "__main__":
    V = load()
    F = folds(V)
    V.to_pickle(f"{HERE}/lodo_V.pkl")
    pd.to_pickle([f for f in F if f["usable"]], f"{HERE}/lodo_folds.pkl")
    D = pd.DataFrame(F)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 55)
    print(f"cast verified rows after averaging within-paper repeats: {len(V)}")
    print(D[["fold_id", "n_train", "n_test", "train_comps", "test_comps", "leak_rows_removed", "T_train", "T_test", "y_train", "y_test", "y_test_sd", "usable", "fail"]]
          .assign(fold_id=D.fold_id.str.replace("|CAST|", "|", regex=False)).to_string(index=False))
    print("\nusable folds", int(D.usable.sum()), "of", len(D))
    for f in F:
        if f["usable"]:
            print(f"\n{f['fold_id']}\n  TEST  {f['held_out']} {f['n_test']} rows "
                  f"({f['test_comps']} compositions, {f['T_test']}C, "
                  f"{f['y_test']} MPa, SD {f['y_test_sd']})"
                  f"\n  TRAIN {f['train_papers']} papers = {f['n_train']} rows "
                  f"({f['train_comps']} compositions, {f['T_train']}C, "
                  f"{f['y_train']} MPa), removed for leakage "
                  f"{f['leak_rows_removed']} rows")
