"""Nested hyperparameter tuning of the baselines on the repeat-free LOCO pools (norepeat_V.pkl).

Outer loop: leave one composition out (same folds as loco_benchmark.py and loco_ptsr.py).
Inner loop: GroupKFold over the training compositions only (min(5, n compositions) splits). Optuna TPE picks the
hyperparameters with the lowest inner MSE. The test composition is never used for the choice.
For every trial the model is also refit on the whole outer training set and its test prediction is stored, so an
ORACLE (best trial by test error, per held-out composition) can be reported as an upper bound. The oracle looks at
the test data and is not a valid result.

Models follow baselines/lodo_comparison/run.py with the fixed settings turned into search ranges. Two fixes enter as
options the search may pick: target scaling for Liu MLP and an intercept for SISSO-lite. PySR without a template is
not tuned here, the PT-SR run's free_1stage template is that model with the full budget and many seeds.
usage: python tune_baselines.py run N_PROCS    |    python tune_baselines.py report
"""
from pathlib import Path
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOCO_DATA", "norepeat_V.pkl"); os.environ.setdefault("LOCO_MIN_ROWS", "6")
TAG = os.environ.get("LOCO_TAG", "_norepeat")   # one tag per dataset so runs never mix
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/tune_baselines{TAG}"; os.makedirs(OUT, exist_ok=True)
REPO = str(Path(__file__).resolve().parent.parent)
N_TRIALS = {"Liu_RF": 40, "SISSO": 30, "Liu_MLP": 30, "JMI_Stacking": 25, "Jain_DNN": 20,
            "SciRep25_Transformer": 15, "LESets_GNN": 15, "Wu_gplearn_RFR": 15,
            # Cranmer PySR with no physics template. One fit costs seconds to minutes, and the
            # objective fits it once per inner split plus once for the test set, so the trial
            # budget is deliberately small.
            "PySR_no_template": 18}
B = None


def repo():
    global B
    if B is None:
        import importlib.util, contextlib, io
        spec = importlib.util.spec_from_file_location("lb", f"{REPO}/baselines/lodo_comparison/run.py")
        B = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(B)
    return B


def space(name, t):
    if name == "Liu_RF":
        return dict(n=t.suggest_int("n", 100, 500, step=100), depth=t.suggest_categorical("depth", [2, 3, 5, 8, 15, 0]),
                    leaf=t.suggest_int("leaf", 1, 4), mf=t.suggest_float("mf", 0.2, 1.0))
    if name == "Liu_MLP":
        return dict(w=t.suggest_categorical("w", [8, 16, 32, 64, 128]), L=t.suggest_int("L", 1, 3),
                    alpha=t.suggest_float("alpha", 1e-5, 10, log=True), lr=t.suggest_float("lr", 1e-4, 3e-2, log=True),
                    it=t.suggest_categorical("it", [500, 2000, 5000]), yscale=t.suggest_categorical("yscale", [True, False]))
    if name == "JMI_Stacking":
        return dict(ne=t.suggest_int("ne", 100, 500, step=100), ed=t.suggest_categorical("ed", [3, 5, 0]),
                    hi=t.suggest_categorical("hi", [50, 200, 500]), hlr=t.suggest_float("hlr", 0.01, 0.3, log=True),
                    hleaf=t.suggest_int("hleaf", 1, 5), la=t.suggest_float("la", 1e-4, 100, log=True))
    if name == "Wu_gplearn_RFR":
        return dict(pop=t.suggest_categorical("pop", [300, 1000]), gen=t.suggest_categorical("gen", [10, 20, 40]),
                    pc=t.suggest_float("pc", 1e-4, 1e-1, log=True), nc=t.suggest_categorical("nc", [2, 4, 8]),
                    depth=t.suggest_categorical("depth", [3, 5, 15]))
    if name in ("Jain_DNN", "SciRep25_Transformer", "LESets_GNN"):
        p = dict(lr=t.suggest_float("lr", 1e-4, 3e-2, log=True), wd=t.suggest_float("wd", 1e-6, 1e-1, log=True),
                 drop=t.suggest_float("drop", 0.0, 0.3), val=t.suggest_categorical("val", [True, False]),
                 ep=t.suggest_categorical("ep", [200, 500, 1000]))
        if name == "Jain_DNN":
            p.update(w=t.suggest_categorical("w", [16, 32, 64, 128]), L=t.suggest_int("L", 1, 3))
        elif name == "SciRep25_Transformer":
            p.update(emb=t.suggest_categorical("emb", [16, 32]), heads=t.suggest_categorical("heads", [2, 4]),
                     L=t.suggest_int("L", 1, 2))
        else:
            p.update(hid=t.suggest_categorical("hid", [16, 32, 64]))
        return p
    if name == "SISSO":
        return dict(k=t.suggest_int("k", 1, 3), icpt=t.suggest_categorical("icpt", [True, False]))
    if name == "PySR_no_template":
        # the published baseline fixes the operator set and the search budget, so the candidate pool
        # is built from independent random restarts instead. The seed is taken from the trial number
        # so the restarts are distinct, matching how PT-SR draws one candidate per template x variant.
        return dict(seed=t.number)


def train_torch(model, fwd_tr, fwd_va, fwd_te, y_t, y_v, p, seed):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["wd"])
    crit = torch.nn.MSELoss(); yt = torch.FloatTensor(y_t); yv = torch.FloatTensor(y_v)
    best, wait, state = float("inf"), 0, None
    for ep in range(p["ep"]):
        model.train(); opt.zero_grad(); loss = crit(fwd_tr(), yt); loss.backward(); opt.step()
        if not p["val"]:
            continue
        model.eval()
        with torch.no_grad():
            v = crit(fwd_va(), yv).item()
        if v < best:
            best, wait, state = v, 0, {k: x.clone() for k, x in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 30: break
    if state is not None: model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return fwd_te().numpy()


def predict(name, p, Xtr, ytr, Xte, seed, elem):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if name == "Liu_RF":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=p["n"], max_depth=p["depth"] or None, min_samples_leaf=p["leaf"],
                                  max_features=p["mf"], random_state=seed, n_jobs=1)
        return m.fit(Xtr, ytr).predict(Xte)
    if name == "Liu_MLP":
        from sklearn.neural_network import MLPRegressor
        from sklearn.compose import TransformedTargetRegressor
        m = Pipeline([("scl", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(p["w"],) * p["L"], alpha=p["alpha"],
                      learning_rate_init=p["lr"], max_iter=p["it"], batch_size=64, random_state=seed))])
        if p["yscale"]: m = TransformedTargetRegressor(regressor=m, transformer=StandardScaler())
        return m.fit(Xtr, ytr).predict(Xte)
    if name == "JMI_Stacking":
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
        from sklearn.linear_model import Lasso
        est = [("ert", ExtraTreesRegressor(n_estimators=p["ne"], max_depth=p["ed"] or None, random_state=seed, n_jobs=1)),
               ("hgb", HistGradientBoostingRegressor(max_iter=p["hi"], learning_rate=p["hlr"], min_samples_leaf=p["hleaf"], random_state=seed))]
        fin = Pipeline([("scl", StandardScaler()), ("lasso", Lasso(alpha=p["la"], max_iter=10000))])
        m = StackingRegressor(estimators=est, final_estimator=fin, cv=min(3, len(ytr)))
        return m.fit(Xtr, ytr).predict(Xte)
    if name == "Wu_gplearn_RFR":
        from gplearn.genetic import SymbolicTransformer
        from sklearn.ensemble import RandomForestRegressor
        gp = SymbolicTransformer(population_size=p["pop"], generations=p["gen"], function_set=("add", "sub", "mul", "div", "inv", "log", "sqrt", "max", "min"),
                                 init_depth=(2, 4), parsimony_coefficient=p["pc"], hall_of_fame=max(20, 2 * p["nc"]), n_components=p["nc"],
                                 metric="pearson", verbose=0, random_state=seed, n_jobs=1)
        try:
            gp.fit(Xtr, ytr); Xt, Xe = np.hstack([Xtr, gp.transform(Xtr)]), np.hstack([Xte, gp.transform(Xte)])
        except Exception:
            Xt, Xe = Xtr, Xte
        rf = RandomForestRegressor(n_estimators=300, max_depth=p["depth"], random_state=seed, n_jobs=1)
        return rf.fit(Xt, ytr).predict(Xe)
    if name == "PySR_no_template":
        # Cranmer et al. (2023) PySR run with no template, same operator set and population
        # settings as baselines/lodo_comparison/run.py
        from pysr import PySRRegressor
        m = PySRRegressor(niterations=30, binary_operators=["+", "-", "*", "/"],
                          unary_operators=["log", "exp", "sqrt", "square"], maxsize=15,
                          populations=15, population_size=33, progress=False,
                          random_state=p["seed"], deterministic=True, parallelism="serial",
                          verbosity=0, temp_equation_file=True)
        try:
            m.fit(Xtr, ytr)
            return np.asarray(m.predict(Xte), float)
        except Exception:
            return np.full(len(Xte), np.nan)
    if name == "SISSO":
        n_in = Xtr.shape[1]
        def expand(X, ref):
            k = min(8, n_in)
            prods = [(X[:, i] * X[:, j])[:, None] for i in range(k) for j in range(i + 1, k)]
            pos = X - ref.min(axis=0, keepdims=True) + 1.0
            return np.hstack([X] + ([np.hstack(prods)] if prods else []) + [np.log(np.clip(pos, 1e-6, None)), np.sqrt(np.clip(pos, 0, None)), 1.0 / (np.abs(X) + 1e-3)])
        scl = StandardScaler().fit(expand(Xtr, Xtr)); A = np.nan_to_num(scl.transform(expand(Xtr, Xtr))); E = np.nan_to_num(scl.transform(expand(Xte, Xtr)))
        sel, res = [], ytr - ytr.mean()
        design = lambda M, s: np.hstack([M[:, s], np.ones((len(M), 1))]) if p["icpt"] else M[:, s]
        for _ in range(p["k"]):
            c = np.abs(A.T @ res) / (np.linalg.norm(A, axis=0) * np.linalg.norm(res) + 1e-12); c[sel] = -1
            sel.append(int(c.argmax())); beta, *_ = np.linalg.lstsq(design(A, sel), ytr, rcond=None); res = ytr - design(A, sel) @ beta
        return design(E, sel) @ beta
    # torch models
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtr); perm = np.random.RandomState(seed).permutation(n)
    if p["val"] and n >= 4:
        nv = max(1, int(n * 0.15)); vi, ti = perm[:nv], perm[nv:]
    else:
        p = dict(p, val=False); vi, ti = perm[:1], perm
    Xt, Xv = Xtr[ti], Xtr[vi]; y_t, y_v = ytr[ti], ytr[vi]
    ym, ys = float(y_t.mean()), float(y_t.std() + 1e-8)
    scl = StandardScaler().fit(Xt)
    if name in ("Jain_DNN", "SciRep25_Transformer"):
        T = {k: torch.FloatTensor(scl.transform(v)) for k, v in (("t", Xt), ("v", Xv), ("e", Xte))}
        d = Xtr.shape[1]
        if name == "Jain_DNN":
            layers, h = [], d
            for _ in range(p["L"]): layers += [nn.Linear(h, p["w"]), nn.ReLU(), nn.Dropout(p["drop"])]; h = p["w"]
            net = nn.Sequential(*layers, nn.Linear(h, 1)); f = lambda x: net(x).squeeze(-1); model = net
        else:
            class TF(nn.Module):
                def __init__(s):
                    super().__init__(); s.embed = nn.Linear(1, p["emb"]); s.pos = nn.Parameter(torch.randn(d, p["emb"]) * 0.02)
                    s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(p["emb"], p["heads"], 4 * p["emb"], p["drop"], batch_first=True), p["L"])
                    s.head = nn.Sequential(nn.Linear(p["emb"], 64), nn.ReLU(), nn.Linear(64, 1))
                def forward(s, x): return s.head(s.enc(s.embed(x.unsqueeze(-1)) + s.pos).mean(1)).squeeze(-1)
            model = TF(); f = model
        pr = train_torch(model, lambda: f(T["t"]), lambda: f(T["v"]), lambda: f(T["e"]), (y_t - ym) / ys, (y_v - ym) / ys, p, seed)
        return pr * ys + ym
    # LESets GNN, graph construction as in run.py
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import CGConv
    from torch_geometric.utils import scatter
    R = repo(); ne = len(elem)
    nd = np.array([[R.ELEM_R.get(e, 130.0), R.ELEM_CHI.get(e, 1.6), R.ELEM_VEC.get(e, 6.0)] for e in elem], np.float32)
    nds = (nd - nd.mean(0, keepdims=True)) / (nd.std(0, keepdims=True) + 1e-8)
    ii, jj = np.triu_indices(ne, 1); ei = torch.LongTensor(np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])]))
    def graphs(X):
        out = []
        for row in X:
            x = row[:ne]; s, t = ei[0].numpy(), ei[1].numpy()
            nf = np.concatenate([x[:, None], nds, np.full((ne, 1), (row[-1] - 700.0) / 500.0)], 1).astype(np.float32)
            ea = np.stack([np.abs(nd[s, 0] - nd[t, 0]) / 150.0, np.abs(nd[s, 1] - nd[t, 1]), x[s] * x[t]], 1).astype(np.float32)
            out.append(Data(x=torch.FloatTensor(nf), edge_index=ei, edge_attr=torch.FloatTensor(ea), frac=torch.FloatTensor(x)))
        return Batch.from_data_list(out)
    class GNN(nn.Module):
        def __init__(s, h=p["hid"]):
            super().__init__(); s.proj = nn.Linear(5, h); s.c1 = CGConv(h, dim=3); s.c2 = CGConv(h, dim=3)
            s.drop = nn.Dropout(p["drop"]); s.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(s, b):
            h = torch.relu(s.c1(s.proj(b.x), b.edge_index, b.edge_attr)); h = s.drop(torch.relu(s.c2(h, b.edge_index, b.edge_attr)))
            return s.head(scatter(h * b.frac.view(-1, 1), b.batch, dim=0, reduce="sum")).squeeze(-1)
    model = GNN(); gt, gv, ge = graphs(Xt), graphs(Xv), graphs(Xte)
    pr = train_torch(model, lambda: model(gt), lambda: model(gv), lambda: model(ge), (y_t - ym) / ys, (y_v - ym) / ys, p, seed)
    return pr * ys + ym


def tasks():
    import loco_benchmark as LB
    out = []
    for name in N_TRIALS:
        for pl in LB.pools():
            rows = LB.V.loc[pl["idx"]]
            for ck in pd.unique(rows.ckey.map(str)):
                out.append((name, pl["pool"], ck))
    return out


def run_task(job):
    name, pool, ck = job
    import hashlib
    fn = f"{OUT}/{name}__{hashlib.md5((pool + ck).encode()).hexdigest()[:12]}.json"
    if os.path.exists(fn):
        return fn
    import optuna, warnings
    from sklearn.model_selection import GroupKFold
    warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
    import loco_benchmark as LB
    pl = {q["pool"]: q for q in LB.pools()}[pool]
    rows = LB.V.loc[pl["idx"]]; elem = sorted(pl["S"]); X = LB.features(rows, elem); y = rows.YS_MPa.values
    keys = rows.ckey.map(str).values; te = keys == ck; tr = ~te
    Xtr, ytr, Xte, gtr = X[tr], y[tr], X[te], keys[tr]
    splits = list(GroupKFold(n_splits=min(5, len(set(gtr)))).split(Xtr, ytr, gtr))
    trials = []
    t0 = time.time()

    def objective(t):
        p = space(name, t)
        pr = np.empty(len(ytr))
        for a, b in splits:
            pr[b] = predict(name, p, Xtr[a], ytr[a], Xtr[b], 0, elem)
        mse = float(np.mean((ytr - pr) ** 2)) if np.all(np.isfinite(pr)) else 1e30
        pte = np.asarray(predict(name, p, Xtr, ytr, Xte, 0, elem), float)
        trials.append(dict(params=p, inner_mse=mse, pred_test=[float(v) for v in np.nan_to_num(pte, nan=1e9)]))
        return mse

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=N_TRIALS[name], catch=(Exception,))
    json.dump(dict(model=name, pool=pool, ck=ck, test_rows=rows.index[te].tolist(), y_test=y[te].tolist(),
                   trials=trials, time_s=time.time() - t0), open(fn, "w"))
    print(time.strftime("[%H:%M:%S]"), name, pool, f"{len(trials)} trials {time.time() - t0:.0f}s", flush=True)
    return fn


def report(verbose=True):
    import glob
    recs = [json.load(open(f)) for f in glob.glob(f"{OUT}/*.json")]
    if not recs:
        print("tuning: no finished folds yet"); return None
    D = pd.DataFrame(recs)
    nf = {"AlCrFeMnNi|CAST|C": 9, "HfMoNbTaTiZr|CAST|C": 6, "AlCoCrFeMoNi|CAST|C": 6, "AlCoCrFeMnNi|CAST|T": 10}
    print(f"tuning: {len(D)} / {len(N_TRIALS) * sum(nf.values())} model-fold tasks done")
    out = []
    for (m, pool), G in D.groupby(["model", "pool"]):
        if len(G) < nf[pool]:
            continue
        yt, yf, yo = [], [], []
        for _, r in G.iterrows():
            T = [t for t in r.trials if np.isfinite(t["inner_mse"]) and t["inner_mse"] < 1e29]
            if not T: continue
            y = np.array(r.y_test); yt += list(y)
            yf += min(T, key=lambda t: t["inner_mse"])["pred_test"]
            yo += min(T, key=lambda t: np.mean(np.abs(np.array(t["pred_test"]) - y)))["pred_test"]
        yt, yf, yo = map(np.array, (yt, yf, yo)); ss = np.sum((yt - yt.mean()) ** 2)
        out.append(dict(model=m, pool=pool, tuned_R2=1 - np.sum((yt - yf) ** 2) / ss, ORACLE_R2=1 - np.sum((yt - yo) ** 2) / ss))
    if not out:
        return None
    S = pd.DataFrame(out)
    S.to_csv(f"{HERE}/tune_scores{TAG}.csv", index=False)
    if verbose:
        print(S.pivot(index="model", columns="pool", values="tuned_R2").round(2).to_string())
    return S


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        from multiprocessing import Pool
        J = tasks()
        print(f"{len(J)} tasks", flush=True)
        with Pool(int(sys.argv[2])) as pool:
            for _ in pool.imap_unordered(run_task, J):
                pass
        report()
