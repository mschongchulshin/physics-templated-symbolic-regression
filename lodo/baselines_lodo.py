from pathlib import Path
import os, sys, json, time, glob
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROP = os.environ.get("LODO_PROP", "YS")
SUF = "" if PROP == "YS" else f"_{PROP}"
OUT = f"{HERE}/results_baselines_lodo{SUF}"; os.makedirs(OUT, exist_ok=True)
V = pd.read_pickle(f"{HERE}/lodo_V{SUF}.pkl")
FOLDS = {f["fold_id"]: f for f in pd.read_pickle(f"{HERE}/lodo_folds{SUF}.pkl")}


def feats(idx, elem):
    from features import hume_rothery_features
    X = []
    for _, r in V.loc[idx].iterrows():
        x = np.array([r.vec.get(e, 0.0) for e in elem]); x = x / x.sum()
        X.append(list(x) + list(hume_rothery_features(elem, x)) + [float(r["T"]) + 273.15])
    return np.array(X, float)


def run_task(job):
    name, fid = job
    fn = f"{OUT}/{name}__{fid.replace('|', '_').replace(' ', '_').replace('/', '-')}.json"
    if os.path.exists(fn):
        return fn
    import optuna, warnings
    import tune_baselines as TB
    warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
    f = FOLDS[fid]; elem = sorted(f["S"])
    Xtr, ytr = feats(f["train"], elem), V.loc[f["train"], "YS_MPa"].values
    Xte, yte = feats(f["test"], elem), V.loc[f["test"], "YS_MPa"].values
    labs = V.loc[f["train"], "lab_name"].values
    splits = [(np.where(labs != l)[0], np.where(labs == l)[0]) for l in pd.unique(labs)]
    splits = [(a, b) for a, b in splits if len(a) >= 4 and len(b) >= 1]
    trials = []

    def objective(t):
        p = TB.space(name, t)
        pr = np.empty(len(ytr)); pr[:] = np.nan
        for a, b in splits:
            pr[b] = TB.predict(name, p, Xtr[a], ytr[a], Xtr[b], 0, elem)
        m = ~np.isnan(pr)
        mse = float(np.mean((ytr[m] - pr[m]) ** 2)) if m.any() and np.all(np.isfinite(pr[m])) else 1e30
        pte = np.asarray(TB.predict(name, p, Xtr, ytr, Xte, 0, elem), float)
        trials.append(dict(params=p, inner_mse=mse, pred_test=[float(v) for v in np.nan_to_num(pte, nan=1e9)]))
        return mse

    t0 = time.time()
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=TB.N_TRIALS[name], catch=(Exception,))
    json.dump(dict(model=name, fold=fid, y_test=yte.tolist(), test_rows=f["test"], trials=trials,
                   time_s=time.time() - t0), open(fn, "w"))
    print(time.strftime("[%H:%M:%S]"), name, fid[:50], f"{len(trials)} trials {time.time() - t0:.0f}s", flush=True)
    return fn


def report():
    import tune_baselines as TB
    recs = [json.load(open(f)) for f in glob.glob(f"{OUT}/*.json")]
    print(f"baseline LODO tasks done {len(recs)} / {len(TB.N_TRIALS) * len(FOLDS)}")
    if not recs:
        return
    rows = []
    for r in recs:
        T = [t for t in r["trials"] if np.isfinite(t["inner_mse"]) and t["inner_mse"] < 1e29]
        if not T:
            continue
        yt = np.array(r["y_test"]); yp = np.array(min(T, key=lambda t: t["inner_mse"])["pred_test"], float)
        rows.append(dict(model=r["model"], fold=r["fold"], tuned_R2=round(float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)), 3)))
    S = pd.DataFrame(rows)
    S.to_csv(f"{HERE}/lodo_scores_baselines.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.pivot(index="model", columns="fold", values="tuned_R2").to_string())


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(); sys.exit()
    import tune_baselines as TB
    from multiprocessing import Pool
    J = [(name, fid) for name in TB.N_TRIALS for fid in FOLDS]
    print(f"{len(J)} tasks", flush=True)
    with Pool(int(sys.argv[2])) as pool:
        for _ in pool.imap_unordered(run_task, J):
            pass
    report()
