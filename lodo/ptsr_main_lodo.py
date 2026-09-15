from pathlib import Path
import os, sys, time, json, re, subprocess
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("JULIA_NUM_THREADS", "1")
os.environ.setdefault("JULIA_NUM_GC_THREADS", "1")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = str(Path(__file__).resolve().parent.parent)
CONV = os.environ.get("PTSR_CONV") == "1"
STEP = int(os.environ.get("PTSR_STEP", "100")); MAXIT = int(os.environ.get("PTSR_MAXIT", "3000"))
OUT = f"{HERE}/results_main" + ("_conv" if CONV else ""); TMP = f"{HERE}/_pysr_main_tmp"
LAST_ITERS = []


def fit_conv(m, X, y):
    if not CONV:
        m.fit(X, y); LAST_ITERS.append(int(m.get_params()["niterations"])); return m
    m.set_params(niterations=STEP, warm_start=True)
    prev, stall, total = None, 0, 0
    while total < MAXIT:
        m.fit(X, y); total += STEP
        loss = float(m.equations_["loss"].min())
        print(time.strftime("[%H:%M:%S]"), f"  chunk it={total} loss={loss:.5g}", flush=True)
        stall = stall + 1 if (prev is not None and prev - loss <= 0.01 * abs(prev)) else 0
        prev = loss
        if stall >= 2:
            break
    LAST_ITERS.append(total)
    return m
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
TEMPLATES = ["additive", "multiplicative", "power_law", "thermal_softening", "arrhenius",
             "entropy_weighted", "calphad", "free_2stage", "free_1stage"]


def jobs(seeds):
    M = pd.read_pickle(f"{HERE}/verified_M.pkl")
    return [(f, t, s) for f in sorted(M.fold_id.unique()) for t in TEMPLATES for s in seeds]


def build(fid):
    sys.path.insert(0, f"{REPO}/lodo")
    from features import hume_rothery_features
    V = pd.read_pickle(f"{HERE}/verified_V.pkl"); M = pd.read_pickle(f"{HERE}/verified_M.pkl")
    elem = sorted(re.findall(r"[A-Z][a-z]?", fid.split("|")[1]))
    cols = [f"x_{e}" for e in elem] + ["S_mix", "H_mix", "delta", "VEC", "dChi", "r_avg", "chi_avg"]
    out = {}
    for role in ["train", "test"]:
        sub = V.loc[M[(M.fold_id == fid) & (M.role == role)].row]
        X, y, T = [], [], []
        for _, r in sub.iterrows():
            x = np.array([r.vec.get(e, 0.0) for e in elem]); x = x / x.sum()
            X.append(list(x) + list(hume_rothery_features(elem, x)))
            y.append(float(r.YS_MPa)); T.append(float(r["T"]) + 273.15)
        out[role] = (pd.DataFrame(X, columns=cols), np.array(y), np.array(T))
    return out


def r2(y, p):
    p = np.asarray(p, float)
    if not np.all(np.isfinite(p)) or np.var(y) < 1e-12: return float("nan")
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def run_job(fid, tn, seed):
    from pysr import PySRRegressor
    d = build(fid)
    Xtr, ytr, Ttr = d["train"]; Xte, yte, Tte = d["test"]
    rt = np.abs(Ttr - 298.15) <= 10
    tag = re.sub(r"[^A-Za-z0-9]", "", fid)[:40] + f"_{tn}_s{seed}"

    POPS = int(os.environ.get("PTSR_POPS", "40")); POPSIZE = int(os.environ.get("PTSR_POPSIZE", "60"))
    NITER = int(os.environ.get("PTSR_NITER", "0"))

    def pysr(name, **kw):
        base = dict(binary_operators=["+", "-", "*", "/"], unary_operators=["square", "cube", "sqrt", "log", "exp"],
                    populations=POPS, population_size=POPSIZE, parsimony=0.005, turbo=True, bumper=True,
                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                    deterministic=False, procs=1, verbosity=0, progress=False, random_state=seed,
                    temp_equation_file=True, tempdir=f"{TMP}/{tag}_{name}")
        base.update(kw); os.makedirs(base["tempdir"], exist_ok=True)
        return PySRRegressor(**base)

    def cx(m):
        try: return int(m.get_best()["complexity"])
        except Exception: return 0

    G = 0.05 * float(np.median(np.abs(ytr)))
    CAP = 2.0 * float(ytr.max())
    clipf = lambda p: np.clip(np.nan_to_num(np.asarray(p, float), nan=0.0, posinf=CAP, neginf=0.0), 0.0, CAP)
    gfloor = lambda g: np.where(np.abs(g) < G, np.where(g < 0, -G, G), g)
    rec = []
    LAST_ITERS.clear()

    def add(variant, ptr, pte, k, s1, s2):
        mse = float(np.mean((ytr - np.asarray(ptr, float)) ** 2)) if np.all(np.isfinite(ptr)) else float("inf")
        n = len(ytr)
        bic = n * np.log(mse) + k * np.log(n) if np.isfinite(mse) and mse > 0 else float("inf")
        mae = float(np.mean(np.abs(yte - np.asarray(pte, float)))) if np.all(np.isfinite(pte)) else float("nan")
        rec.append(dict(fold_id=fid, template=tn, seed=seed, variant=variant, n_train=n, n_test=len(yte),
                        train_r2=r2(ytr, ptr), test_r2=r2(yte, pte), test_mae=mae, bic=bic, complexity=k,
                        S1=s1, S2=s2, error="", iterations=json.dumps(list(LAST_ITERS)), training="converged" if CONV else "standard", pred_test=json.dumps([float(v) for v in np.asarray(pte, float)])))

    if tn == "free_1stage":
        XT = Xtr.assign(T=Ttr); XTe = Xte.assign(T=Tte)
        m = pysr("f", niterations=200, maxsize=25); fit_conv(m, XT, ytr)
        ptr, pte = m.predict(XT), m.predict(XTe); eq = str(m.get_best()["equation"])
        add("original", ptr, pte, cx(m), eq, ""); add("guarded", clipf(ptr), clipf(pte), cx(m), eq, "")
        return rec

    gm = pysr("g", niterations=200, maxsize=20, parsimony=0.005, unary_operators=["square", "cube", "sqrt", "log"])
    fit_conv(gm, Xtr[rt], ytr[rt])
    gtr, gte = gm.predict(Xtr), gm.predict(Xte); s1 = str(gm.get_best()["equation"]); k1 = cx(gm)
    Tdf, Tdfe = pd.DataFrame({"T": Ttr}), pd.DataFrame({"T": Tte})

    if tn == "additive":
        h = pysr("h", niterations=200, maxsize=15, parsimony=0.005); fit_conv(h, Tdf, ytr - gtr)
        ptr, pte = gtr + h.predict(Tdf), gte + h.predict(Tdfe); s2 = str(h.get_best()["equation"]); k = k1 + cx(h)
        add("original", ptr, pte, k, s1, s2); add("guarded", clipf(ptr), clipf(pte), k, s1, s2)
    elif tn == "calphad":
        res = np.nan_to_num((ytr - gtr) / (-Ttr), nan=0, posinf=0, neginf=0)
        h = pysr("h", niterations=200, maxsize=15, parsimony=0.01); fit_conv(h, Xtr, res)
        ptr, pte = gtr - Ttr * h.predict(Xtr), gte - Tte * h.predict(Xte); s2 = str(h.get_best()["equation"]); k = k1 + cx(h)
        add("original", ptr, pte, k, s1, s2); add("guarded", clipf(ptr), clipf(pte), k, s1, s2)
    elif tn == "free_2stage":
        gT, gTe = pd.DataFrame({"g": gtr, "T": Ttr}), pd.DataFrame({"g": gte, "T": Tte})
        f = pysr("h", niterations=300, maxsize=25, parsimony=0.003); fit_conv(f, gT, ytr)
        ptr, pte = f.predict(gT), f.predict(gTe); s2 = str(f.get_best()["equation"]); k = k1 + cx(f)
        add("original", ptr, pte, k, s1, s2); add("guarded", clipf(ptr), clipf(pte), k, s1, s2)
    elif tn in ("multiplicative", "power_law", "thermal_softening"):
        use_comp = tn == "thermal_softening"
        H, He = (Xtr.assign(T=Ttr), Xte.assign(T=Tte)) if use_comp else (Tdf, Tdfe)
        pars = 0.01 if use_comp else 0.005
        for variant in ["original", "guarded"]:
            if variant == "original":
                gs, gse = np.where(np.abs(gtr) < 1e-10, 1e-10, gtr), gte
                ratio = np.nan_to_num(ytr / gs, nan=1, posinf=1, neginf=1)
            else:
                gs, gse = gfloor(gtr), gfloor(gte)
                ratio = np.clip(ytr / gs, 0.05, 20)
            h = pysr(f"h_{variant}", niterations=200, maxsize=15, parsimony=pars); fit_conv(h, H, ratio)
            ptr, pte = gs * h.predict(H), gse * h.predict(He)
            if variant == "guarded": ptr, pte = clipf(ptr), clipf(pte)
            add(variant, ptr, pte, k1 + cx(h), s1, str(h.get_best()["equation"]))
    elif tn in ("arrhenius", "entropy_weighted"):
        if tn == "arrhenius":
            H, He = Xtr.assign(inv_T=1 / Ttr), Xte.assign(inv_T=1 / Tte)
        else:
            H, He = pd.DataFrame({"S_mix": Xtr.S_mix, "inv_T": 1 / Ttr}), pd.DataFrame({"S_mix": Xte.S_mix, "inv_T": 1 / Tte})
        for variant in ["original", "guarded"]:
            floor = 1e-10 if variant == "original" else G
            gs, gse = np.clip(np.abs(gtr), floor, None), np.clip(np.abs(gte), floor, None)
            lr = np.nan_to_num(np.log(np.clip(np.abs(ytr), 1e-10, None)) - np.log(gs), nan=0, posinf=0, neginf=0)
            if variant == "guarded": lr = np.clip(lr, -5, 5)
            h = pysr(f"h_{variant}", niterations=200, maxsize=15, parsimony=0.01); fit_conv(h, H, lr)
            ptr, pte = gs * np.exp(np.clip(h.predict(H), -20, 20)), gse * np.exp(np.clip(h.predict(He), -20, 20))
            if variant == "guarded": ptr, pte = clipf(ptr), clipf(pte)
            add(variant, ptr, pte, k1 + cx(h), s1, str(h.get_best()["equation"]))
    return rec


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "launch":
        n, seeds = int(sys.argv[2]), sys.argv[3]
        procs = [subprocess.Popen([sys.executable, __file__, "worker", str(k), str(n), seeds],
                                  stdout=open(f"{OUT}/worker_{k}.log", "a"), stderr=subprocess.STDOUT) for k in range(n)]
        codes = [p.wait() for p in procs]
        print("workers finished", codes)
    else:
        k, n = int(sys.argv[2]), int(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]
        csv = f"{OUT}/shard_{k}.csv"
        done = set()
        if os.path.exists(csv):
            old = pd.read_csv(csv); done = set(zip(old.fold_id, old.template, old.seed))
        for i, (fid, tn, seed) in enumerate(jobs(seeds)):
            if i % n != k or (fid, tn, seed) in done: continue
            t0 = time.time()
            try:
                rec = run_job(fid, tn, seed)
            except Exception as e:
                rec = [dict(fold_id=fid, template=tn, seed=seed, variant=v, error=f"{type(e).__name__}: {e}") for v in ("original", "guarded")]
            for r in rec: r["time_s"] = time.time() - t0
            pd.DataFrame(rec).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            print(time.strftime("[%H:%M:%S]"), fid, tn, seed, [(r.get("variant"), r.get("test_r2")) for r in rec], flush=True)
