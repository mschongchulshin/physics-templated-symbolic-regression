"""
SR Sensitivity worker: run with `python3 sr_sens_worker.py <worker_id> <n_workers>`
Each worker handles jobs where job_index % n_workers == worker_id
"""
import sys, json, numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.metrics import r2_score
from pysr import PySRRegressor

WORKER_ID = int(sys.argv[1])
N_WORKERS = int(sys.argv[2])
OUT_CSV = Path(f"/tmp/sr_sens_w{WORKER_ID}.csv")

BASE = Path("/Users/hongchulshin/Desktop/696/github_repo/physics-template-SR-HEA")
EQ_JSON = BASE / "equations/all_equations_660.json"

data = pd.read_csv(BASE / "data/raw/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})
feat = pd.read_csv(BASE / "data/features/features_13.csv")

CN = ["Co","Cr","Cu","Fe","Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=float)
T_all = data["T"].values.astype(float)
inv_T = 1.0 / T_all
feat_cols = list(feat.columns)
xfrac = Xc.values / 100.0
S_mix = -8.314 * np.sum(xfrac * np.where(xfrac > 0, np.log(xfrac), 0), axis=1)

with open(EQ_JSON) as f:
    eq_list = json.load(f)

TARGETS = {
    "Youngs_modulus": "Young's modulus(Gpa)",
    "UTS": "UTS(Gpa)",
    "Disloc_0pct": "Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc_20pct": "Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC_0pct": "FCC 0%(%)", "HCP_0pct": "HCP 0%(%)",
    "BCC_0pct": "BCC 0%(%)", "Other_0pct": "Other 0%(%)",
    "FCC_20pct": "FCC 20%(%)", "HCP_20pct": "HCP 20%(%)",
    "BCC_20pct": "BCC 20%(%)", "Other_20pct": "Other 20%(%)",
}

TWO_STAGE = {
    "additive":         {"parsimony": 0.005, "niter": 200, "maxsize": 15},
    "multiplicative":   {"parsimony": 0.005, "niter": 200, "maxsize": 15},
    "thermal_softening":{"parsimony": 0.01,  "niter": 200, "maxsize": 15},
    "arrhenius":        {"parsimony": 0.01,  "niter": 200, "maxsize": 15},
    "power_law":        {"parsimony": 0.005, "niter": 200, "maxsize": 15},
    "entropy_weighted": {"parsimony": 0.01,  "niter": 200, "maxsize": 15},
    "calphad":          {"parsimony": 0.01,  "niter": 200, "maxsize": 15},
    "free_2stage":      {"parsimony": 0.003, "niter": 300, "maxsize": 25},
}
SINGLE_STAGE = {
    "free_1stage":  {"parsimony": 0.005, "niter": 200, "maxsize": 25},
    "freeform":     {"parsimony": 0.005, "niter": 200, "maxsize": 25},
    "6feat_control":{"parsimony": 0.005, "niter": 200, "maxsize": 25},
}

PARSIMONY_VALS = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
NITER_VALS = [50, 100, 150, 200, 300, 400]
SEED = 0

def get_eq(template, target):
    for e in eq_list:
        if e["template"] == template and e["target"] == target and e["seed"] == 0:
            return e
    return None

def safe_eval(eq_str, row_dict):
    ns = {"log": np.log, "sqrt": np.sqrt, "exp": np.exp, "abs": np.abs,
          "square": lambda x: x**2, "cube": lambda x: x**3,
          "clip": lambda x,a,b=None: np.maximum(x,a) if b is None else np.clip(x,a,b)}
    ns.update(row_dict)
    try:
        return float(eval(eq_str, {"__builtins__": {}}, ns))
    except:
        return np.nan

def compute_g_pred(template, target):
    eq_info = get_eq(template, target)
    if eq_info is None: return None
    eq_str = eq_info.get("stage1_equation","")
    if not eq_str: return None
    preds = []
    for i in range(len(data)):
        row = {c: float(Xc.iloc[i][c]) for c in CN}
        for fc in feat_cols: row[fc] = float(feat.iloc[i][fc])
        row.update({"T": T_all[i], "inv_T": inv_T[i], "S_mix": S_mix[i]})
        preds.append(safe_eval(eq_str, row))
    return np.array(preds)

def get_stage2_features(template, g_pred=None):
    if template == "additive":
        return pd.DataFrame({"T": T_all})
    elif template in ("multiplicative","power_law"):
        return pd.DataFrame({"T": T_all})
    elif template == "thermal_softening":
        return pd.concat([Xc.reset_index(drop=True), pd.DataFrame({"T": T_all})], axis=1)
    elif template == "arrhenius":
        return pd.concat([Xc.reset_index(drop=True), feat.reset_index(drop=True),
                          pd.DataFrame({"inv_T": inv_T})], axis=1)
    elif template == "calphad":
        return pd.concat([Xc.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    elif template == "entropy_weighted":
        return pd.DataFrame({"S_mix": S_mix, "inv_T": inv_T})
    elif template == "free_2stage":
        return pd.concat([Xc.reset_index(drop=True), feat.reset_index(drop=True),
                          pd.DataFrame({"T": T_all, "g": g_pred})], axis=1)
    elif template == "6feat_control":
        return pd.concat([Xc.reset_index(drop=True), pd.DataFrame({"T": T_all})], axis=1)
    else:
        return pd.concat([Xc.reset_index(drop=True), feat.reset_index(drop=True),
                          pd.DataFrame({"T": T_all})], axis=1)

def get_stage2_target(template, y, g_pred):
    g_safe = np.clip(np.abs(g_pred), 1e-10, None)
    y_safe = np.clip(np.abs(y), 1e-10, None)
    if template == "additive": return y - g_pred
    elif template in ("multiplicative","power_law","thermal_softening","entropy_weighted"):
        return np.nan_to_num(y/g_safe, nan=1., posinf=1., neginf=1.)
    elif template == "arrhenius":
        return np.nan_to_num(np.log(y_safe)-np.log(g_safe), nan=0., posinf=0., neginf=0.)
    elif template == "calphad":
        return np.nan_to_num((y-g_pred)/(-T_all), nan=0., posinf=0., neginf=0.)
    return y

def eval_full_r2(template, y, X_feat, model, g_pred):
    try:
        h = model.predict(X_feat)
    except:
        return np.nan
    g_safe = np.clip(np.abs(g_pred), 1e-10, None) if g_pred is not None else None
    if template == "additive": yp = g_pred + h
    elif template in ("multiplicative","power_law","thermal_softening","entropy_weighted"):
        yp = g_safe * h
    elif template == "arrhenius": yp = g_safe * np.exp(np.clip(h,-20,20))
    elif template == "calphad": yp = g_pred - T_all * h
    else: yp = h
    return float(r2_score(y, yp))

def make_pysr(niterations, maxsize, parsimony, tmpdir):
    return PySRRegressor(
        binary_operators=["+","-","*","/"],
        unary_operators=["square","cube","sqrt","log","exp"],
        populations=40, population_size=60,
        niterations=niterations, maxsize=maxsize, parsimony=parsimony,
        turbo=True, bumper=True,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        deterministic=False, procs=1, verbosity=0,
        random_state=SEED, temp_equation_file=True, tempdir=tmpdir,
    )

# ── Build job list ─────────────────────────────────────────────────────────
all_templates = list(TWO_STAGE.keys()) + list(SINGLE_STAGE.keys())
jobs = []
for tmpl in all_templates:
    cfg = TWO_STAGE.get(tmpl, SINGLE_STAGE.get(tmpl))
    for tkey in TARGETS:
        for p in PARSIMONY_VALS:
            jobs.append(("parsimony", tmpl, tkey, p, cfg["niter"], cfg["maxsize"]))
        for n in NITER_VALS:
            jobs.append(("niterations", tmpl, tkey, cfg["parsimony"], n, cfg["maxsize"]))

# Filter to this worker's jobs
my_jobs = [j for i, j in enumerate(jobs) if i % N_WORKERS == WORKER_ID]

# Skip already done
done_keys = set()
if OUT_CSV.exists():
    done_df = pd.read_csv(OUT_CSV)
    done_keys = set(zip(done_df["experiment"], done_df["model"], done_df["target"],
                        done_df["parsimony"].astype(str), done_df["niterations"].astype(str)))

print(f"Worker {WORKER_ID}: {len(my_jobs)} jobs to run", flush=True)

for ji, (exp, tmpl, tkey, parsimony, niter, maxsize) in enumerate(my_jobs):
    done_key = (exp, tmpl, tkey, str(parsimony), str(niter))
    if done_key in done_keys:
        print(f"  [{ji+1}/{len(my_jobs)}] SKIP {exp} {tmpl}/{tkey}", flush=True)
        continue

    print(f"  [{ji+1}/{len(my_jobs)}] {exp} {tmpl}/{tkey} p={parsimony} n={niter}", flush=True)
    y = data[TARGETS[tkey]].values.astype(float)
    is_two = tmpl in TWO_STAGE

    if is_two:
        g_pred = compute_g_pred(tmpl, tkey)
        if g_pred is None:
            rows.append({"experiment": exp, "model": tmpl, "target": tkey,
                         "parsimony": parsimony, "niterations": niter, "seed": SEED,
                         "train_r2_best": np.nan, "stage2_r2": np.nan})
            continue
        s2_target = get_stage2_target(tmpl, y, g_pred)
        X_feat = get_stage2_features(tmpl, g_pred)
    else:
        g_pred = None
        s2_target = y
        X_feat = get_stage2_features(tmpl)

    tmpdir = f"/tmp/sr_sens/w{WORKER_ID}_{tmpl}_{tkey}_{exp[0]}{parsimony}_{niter}"
    os.makedirs(tmpdir, exist_ok=True)
    try:
        m = make_pysr(niter, maxsize, parsimony, tmpdir)
        m.fit(X_feat, s2_target)
        s2_r2 = float(r2_score(s2_target, m.predict(X_feat)))
        full_r2 = eval_full_r2(tmpl, y, X_feat, m, g_pred) if is_two else float(r2_score(y, m.predict(X_feat)))
    except Exception as e:
        s2_r2, full_r2 = np.nan, np.nan
        print(f"    ERROR: {e}", flush=True)

    new_row = pd.DataFrame([{"experiment": exp, "model": tmpl, "target": tkey,
                 "parsimony": parsimony, "niterations": niter, "seed": SEED,
                 "train_r2_best": full_r2, "stage2_r2": s2_r2}])
    existing = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined = combined.drop_duplicates(subset=["model","target","parsimony","niterations","experiment"])
    combined.to_csv(OUT_CSV, index=False)

print(f"Worker {WORKER_ID} DONE", flush=True)
