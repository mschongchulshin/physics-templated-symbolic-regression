"""LODO folds on the verified experimental data: hold out one paper of one alloy system.

Source: leakfree_V.pkl, the 138 verified rows (47 papers checked against the original papers), cast material only.
Repeats are not deleted globally here, that emptied the pools. Instead the leak is removed inside each fold:

  fold = (alloy system S, processing, test type, held-out paper)
  train = the same pool without that paper, MINUS every training row that duplicates a test condition
          (composition within 1 at.% total variation and test temperature within 25 C), which is exactly the
          leak the manuscript's evaluation had
  conditions, fixed before any model runs:
      test  >= 3 rows
      train >= 5 rows and >= 3 compositions after the leak rows are removed
  rows whose element set is S, or S with one element at zero, belong to the pool
  exact repeats inside one paper (same composition, temperature, test type) are averaged into one row first

Writes lodo_folds.pkl and prints the train/test table.
usage: python make_lodo_folds.py
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MIN_TEST, MIN_TRAIN, MIN_TRAIN_COMPS = 3, 5, 3


SPREAD = 0.30      # independent measurements of one condition may differ by at most 30 % of their mean


def conflicting(V, verbose=False):
    """Row ids of conditions measured by more than one paper where the values disagree badly.

    Rows of the same composition (1 at.%) and the same test type whose temperatures are within 25 C form one
    condition. If two papers measured it and the values span more than SPREAD of their mean, the whole condition
    is dropped as unreliable. Agreeing repeats are kept, they are real reproduced measurements and dropping them
    would only make the task artificially harder.
    """
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
        print(f"   불일치 제거: {g.formula.iloc[0]} {int(g.T_i.iloc[0])}C {g.test_type.iloc[0]} "
              f"값 {lo:.0f}~{hi:.0f} ({', '.join(sorted(g.lab_name.unique()))})")
    return list(idx)


def load(verbose=False):
    V = pd.read_pickle(f"{HERE}/leakfree_V.pkl").reset_index(drop=True)
    V = V[V.processing == "CAST"].copy()
    V["ck"] = V.vec.apply(lambda v: tuple(sorted((k, round(100 * x)) for k, x in v.items())))
    V["T_i"] = V["T"].round().astype(int)
    # average exact repeats inside one paper
    g = V.groupby([V.lab_name, V.ck.map(str), V.T_i, V.test_type], as_index=False)
    V = g.apply(lambda d: d.assign(YS_MPa=d.YS_MPa.mean()).iloc[:1], include_groups=True).reset_index(drop=True)
    V["els"] = V.vec.apply(frozenset)
    bad = conflicting(V, verbose)
    if verbose:
        print(f"불일치 조건으로 제거한 행 {len(bad)} / {len(V)}")
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
                leak = []      # nothing is dropped per fold any more, see conflicting() for the data quality rule

                why = []
                if len(te) < MIN_TEST: why.append(f"test<{MIN_TEST}")
                if len(tr) < MIN_TRAIN: why.append(f"train<{MIN_TRAIN}")
                if tr.ck.nunique() < MIN_TRAIN_COMPS: why.append(f"train_comps<{MIN_TRAIN_COMPS}")
                # selection inside the fold holds one training paper out, so a single training paper makes the
                # fold unscorable: every model and template gets an undefined inner error
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
            print(f"\n{f['fold_id']}\n  TEST  {f['held_out']} {f['n_test']}점 ({f['test_comps']}조성, {f['T_test']}C, {f['y_test']} MPa, SD {f['y_test_sd']})"
                  f"\n  TRAIN {f['train_papers']} = {f['n_train']}점 ({f['train_comps']}조성, {f['T_train']}C, {f['y_train']} MPa), 누수로 제거 {f['leak_rows_removed']}행")
