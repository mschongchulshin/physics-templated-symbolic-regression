from pathlib import Path
import os, sys, re, time, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, f"{REPO}/lodo")
from features import hume_rothery_features

DATA = os.environ.get("LOCO_DATA", "verified_V.pkl")
TAG = os.environ.get("LOCO_TAG", "")
MIN_ROWS = int(os.environ.get("LOCO_MIN_ROWS", "8"))
V = pd.read_pickle(f"{HERE}/{DATA}").reset_index(drop=True)
V["els"] = V.vec.apply(lambda v: frozenset(v))
V["ckey"] = V.vec.apply(lambda v: tuple(sorted((k, round(100 * x)) for k, x in v.items())))


def pools():
    out = []
    for (proc, tt), G0 in V.groupby(["processing", "test_type"]):
        for S in sorted(set(G0.els), key=lambda s: (-len(s), "".join(sorted(s)))):
            if len(S) < 3:
                continue
            G = G0[G0.els.apply(lambda e: e <= S and len(S - e) <= 1)]
            if len(G) < MIN_ROWS or G.ckey.nunique() < 4 or int((G.els == S).sum()) < 3:
                continue
            if any(p["proc"] == proc and p["tt"] == tt and S < p["S"] for p in out):
                continue
            out.append(dict(pool=f"{''.join(sorted(S))}|{proc}|{tt}", S=S, proc=proc, tt=tt, idx=G.index.tolist()))
    return out


def features(rows, elem):
    X = []
    for _, r in rows.iterrows():
        x = np.array([r.vec.get(e, 0.0) for e in elem]); x = x / x.sum()
        X.append(list(x) + list(hume_rothery_features(elem, x)) + [float(r["T"]) + 273.15])
    return np.array(X, float)


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("lb", f"{REPO}/baselines/lodo_comparison/run.py")
    B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    MODELS = [("Liu_RF", B.fit_predict_liu_rf, False), ("Liu_MLP", B.fit_predict_liu_mlp, False),
              ("Wu_gplearn_RFR", B.fit_predict_wu_gp_rf, False), ("JMI_Stacking", B.fit_predict_jmi_stack, False),
              ("Jain_DNN", B.fit_predict_jain_dnn, False), ("SciRep25_Transformer", B.fit_predict_scirep_transformer, False),
              ("LESets_GNN", B.fit_predict_lesets_gnn, True), ("SISSO", B.fit_predict_sisso, False),
              ("PySR_no_template", B.fit_predict_pysr_free, False)]
    P = pools()
    info = []
    preds = []
    for p in P:
        rows = V.loc[p["idx"]]; elem = sorted(p["S"])
        X = features(rows, elem); y = rows.YS_MPa.values; keys = rows.ckey.map(str).values
        info.append(dict(pool=p["pool"], n=len(rows), compositions=rows.ckey.nunique(), labs=rows.lab_name.nunique(),
                         papers="; ".join(f"{a} ({b})" for a, b in rows.lab_name.value_counts().sort_index().items()),
                         YS_range=f"{y.min():.0f}-{y.max():.0f}", YS_SD=float(y.std()), T_range=f"{int(rows['T'].min())}-{int(rows['T'].max())}"))
        for name, fn, ne in MODELS:
            t0 = time.time()
            for ck in pd.unique(keys):
                te = keys == ck; tr = ~te
                try:
                    pr = fn(X[tr], y[tr], X[te], 0, elem) if ne else fn(X[tr], y[tr], X[te], 0)
                except Exception as e:
                    pr = np.full(te.sum(), np.nan)
                for i, v in zip(np.where(te)[0], pr):
                    preds.append(dict(pool=p["pool"], model=name, row=int(rows.index[i]), paper=rows.lab_name.iloc[i],
                                      formula=rows.formula.iloc[i], T_C=float(rows["T"].iloc[i]), y_true=float(y[i]), y_pred=float(v)))
            print(time.strftime("[%H:%M:%S]"), p["pool"], name, f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(preds).to_csv(f"{HERE}/loco_predictions_baselines{TAG}.csv", index=False)
    pd.DataFrame(info).to_csv(f"{HERE}/loco_pools{TAG}.csv", index=False)
    D = pd.DataFrame(preds)
    def score(g):
        g = g.dropna(subset=["y_pred"]); yt, yp = g.y_true.values, g.y_pred.values
        return pd.Series(dict(n=len(g), R2=1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2), MAE=np.mean(np.abs(yt - yp))))
    S = D.groupby(["pool", "model"]).apply(score).reset_index()
    S.to_csv(f"{HERE}/loco_scores_baselines{TAG}.csv", index=False)
    print(pd.DataFrame(info).to_string(index=False))
    print(S.pivot(index="model", columns="pool", values="R2").round(2).to_string())
