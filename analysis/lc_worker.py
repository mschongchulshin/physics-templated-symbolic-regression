import os
import tempfile
"""
Learning Curve v3 - Fixed val set, no data leak:
- Fixed val: 20% compositions (46 comps = 138 pts), constant across all fractions
- Train: 25/50/75/100% of remaining 80% train pool (182 comps)
- All fractions evaluated on same val set → comparable Test R²
- 100% included (train all 182 pool comps, val fixed 46)
Run: python3 lc_worker_v3.py <worker_id> <n_workers>
"""
import sys, json, numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.metrics import r2_score
from pysr import PySRRegressor

WORKER_ID = int(sys.argv[1])
N_WORKERS = int(sys.argv[2])

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "results" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / f"lc_v3_w{WORKER_ID}.csv"
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

comp_key = data[[f"{n}(%)" for n in CN]].apply(lambda r: "_".join(r.astype(str)), axis=1).values
unique_comps = list(dict.fromkeys(comp_key))
n_total_comps = len(unique_comps)  # 228

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
TWO_STAGE_CFG = {
    "additive":          {"parsimony": 0.005, "maxsize": 15},
    "multiplicative":    {"parsimony": 0.005, "maxsize": 15},
    "thermal_softening": {"parsimony": 0.01,  "maxsize": 15},
    "arrhenius":         {"parsimony": 0.01,  "maxsize": 15},
    "power_law":         {"parsimony": 0.005, "maxsize": 15},
    "entropy_weighted":  {"parsimony": 0.01,  "maxsize": 15},
    "calphad":           {"parsimony": 0.01,  "maxsize": 15},
    "free_2stage":       {"parsimony": 0.003, "maxsize": 25},
}
SINGLE_STAGE_CFG = {
    "free_1stage":   {"parsimony": 0.005, "maxsize": 25},
    "freeform":      {"parsimony": 0.005, "maxsize": 25},
    "6feat_control": {"parsimony": 0.005, "maxsize": 25},
}
ALL_CFG = {**TWO_STAGE_CFG, **SINGLE_STAGE_CFG}
TWO_STAGE = set(TWO_STAGE_CFG.keys())
NITER = 200
SEED = 0

# Fixed train/val split (seed=0, 80/20 composition split)
rng = np.random.default_rng(SEED)
all_comps_shuffled = np.array(unique_comps.copy())
rng.shuffle(all_comps_shuffled)

n_val_comps   = int(round(n_total_comps * 0.20))  # 46
n_train_pool  = n_total_comps - n_val_comps         # 182

val_comps_set   = set(all_comps_shuffled[-n_val_comps:])
train_pool_arr  = all_comps_shuffled[:n_train_pool]  # 182 comps

VAL_IDX = np.where(np.array([c in val_comps_set for c in comp_key]))[0]

# 25/50/75/100% of train pool
DATA_FRACTIONS = [0.25, 0.50, 0.75, 1.00]

def get_eq(template, target):
    for e in eq_list:
        if e["template"] == template and e["target"] == target and e["seed"] == 0:
            return e
    return None

def safe_eval(eq_str, idx):
    ns = {"log": np.log, "sqrt": np.sqrt, "exp": np.exp, "abs": np.abs,
          "square": lambda x: x**2, "cube": lambda x: x**3}
    for c in CN: ns[c] = float(Xc.iloc[idx][c])
    for fc in feat_cols: ns[fc] = float(feat.iloc[idx][fc])
    ns["T"] = T_all[idx]; ns["inv_T"] = inv_T[idx]; ns["S_mix"] = S_mix[idx]
    try:
        return float(eval(eq_str, {"__builtins__": {}}, ns))
    except:
        return np.nan

def compute_g_pred(template, target, indices):
    eq_info = get_eq(template, target)
    if eq_info is None: return None
    eq_str = eq_info.get("stage1_equation", "")
    if not eq_str: return None
    return np.array([safe_eval(eq_str, i) for i in indices])

def get_X_feat(template, indices, g_pred=None):
    Xc_s = Xc.iloc[indices].reset_index(drop=True)
    feat_s = feat.iloc[indices].reset_index(drop=True)
    T_s = T_all[indices]; inv_T_s = inv_T[indices]; S_mix_s = S_mix[indices]
    if template == "additive": return pd.DataFrame({"T": T_s})
    elif template in ("multiplicative","power_law"): return pd.DataFrame({"T": T_s})
    elif template == "thermal_softening":
        return pd.concat([Xc_s, pd.DataFrame({"T": T_s})], axis=1)
    elif template == "arrhenius":
        return pd.concat([Xc_s, feat_s, pd.DataFrame({"inv_T": inv_T_s})], axis=1)
    elif template == "calphad":
        return pd.concat([Xc_s, feat_s], axis=1)
    elif template == "entropy_weighted":
        return pd.DataFrame({"S_mix": S_mix_s, "inv_T": inv_T_s})
    elif template == "free_2stage":
        return pd.concat([Xc_s, feat_s, pd.DataFrame({"T": T_s, "g": g_pred})], axis=1)
    elif template == "6feat_control":
        return pd.concat([Xc_s, pd.DataFrame({"T": T_s})], axis=1)
    else:
        return pd.concat([Xc_s, feat_s, pd.DataFrame({"T": T_s})], axis=1)

def get_s2_target(template, y, g_pred, T_sub):
    g_safe = np.clip(np.abs(g_pred), 1e-10, None)
    y_safe = np.clip(np.abs(y), 1e-10, None)
    if template == "additive": return y - g_pred
    elif template in ("multiplicative","power_law","thermal_softening","entropy_weighted"):
        return np.nan_to_num(y/g_safe, nan=1., posinf=1., neginf=1.)
    elif template == "arrhenius":
        return np.nan_to_num(np.log(y_safe)-np.log(g_safe), nan=0., posinf=0., neginf=0.)
    elif template == "calphad":
        return np.nan_to_num((y-g_pred)/(-T_sub), nan=0., posinf=0., neginf=0.)
    return y

def reconstruct_y(template, h, g_pred, T_sub):
    g_safe = np.clip(np.abs(g_pred), 1e-10, None) if g_pred is not None else None
    if template == "additive": return g_pred + h
    elif template in ("multiplicative","power_law","thermal_softening","entropy_weighted"):
        return g_safe * h
    elif template == "arrhenius": return g_safe * np.exp(np.clip(h,-20,20))
    elif template == "calphad": return g_pred - T_sub * h
    return h

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

# Build jobs
jobs = []
for frac in DATA_FRACTIONS:
    n_tr_comps = int(round(n_train_pool * frac))
    tr_comps_set = set(train_pool_arr[:n_tr_comps])
    tr_idx = np.where(np.array([c in tr_comps_set for c in comp_key]))[0]
    n_tr_pts = len(tr_idx)
    for tmpl in list(TWO_STAGE_CFG.keys()) + list(SINGLE_STAGE_CFG.keys()):
        for tkey in TARGETS:
            jobs.append((frac, n_tr_comps, n_tr_pts, tmpl, tkey, tr_idx))

my_jobs = [j for i, j in enumerate(jobs) if i % N_WORKERS == WORKER_ID]

done_keys = set()
if OUT_CSV.exists():
    done_df = pd.read_csv(OUT_CSV)
    done_keys = set(zip(done_df["data_fraction"].round(2).astype(str),
                        done_df["model"], done_df["target"]))

print(f"Worker {WORKER_ID}: {len(my_jobs)} jobs, val={len(VAL_IDX)} pts", flush=True)

for ji, (frac, n_tr_comps, n_tr_pts, tmpl, tkey, tr_idx) in enumerate(my_jobs):
    done_key = (str(round(frac,2)), tmpl, tkey)
    if done_key in done_keys:
        print(f"  [{ji+1}/{len(my_jobs)}] SKIP frac={frac} {tmpl}/{tkey}", flush=True)
        continue
    print(f"  [{ji+1}/{len(my_jobs)}] frac={frac:.2f} n_tr={n_tr_pts} {tmpl}/{tkey}", flush=True)

    y_full = data[TARGETS[tkey]].values.astype(float)
    y_tr  = y_full[tr_idx];   T_tr  = T_all[tr_idx]
    y_val = y_full[VAL_IDX];  T_val = T_all[VAL_IDX]
    is_two = tmpl in TWO_STAGE

    if is_two:
        g_tr  = compute_g_pred(tmpl, tkey, tr_idx)
        g_val = compute_g_pred(tmpl, tkey, VAL_IDX)
        if g_tr is None or g_val is None:
            print("    no stage1, skip", flush=True); continue
        s2_target = get_s2_target(tmpl, y_tr, g_tr, T_tr)
        X_tr  = get_X_feat(tmpl, tr_idx,  g_tr)
        X_val = get_X_feat(tmpl, VAL_IDX, g_val)
    else:
        g_tr = g_val = None
        s2_target = y_tr
        X_tr  = get_X_feat(tmpl, tr_idx)
        X_val = get_X_feat(tmpl, VAL_IDX)

    tmpdir = os.path.join(tempfile.gettempdir(),
                          f"sr_lc_v3/w{WORKER_ID}_{tmpl}_{tkey}_f{int(frac*100)}")
    os.makedirs(tmpdir, exist_ok=True)
    try:
        cfg = ALL_CFG[tmpl]
        m = make_pysr(NITER, cfg["maxsize"], cfg["parsimony"], tmpdir)
        m.fit(X_tr, s2_target)
        h_val = m.predict(X_val)
        y_pred_val = reconstruct_y(tmpl, h_val, g_val, T_val)
        test_r2 = float(r2_score(y_val, y_pred_val))
    except Exception as e:
        test_r2 = np.nan
        print(f"    ERROR: {e}", flush=True)

    print(f"    test R²={test_r2:.4f}", flush=True)
    new_row = pd.DataFrame([{
        "data_fraction": frac, "n_train_compositions": n_tr_comps,
        "n_train_datapoints": n_tr_pts, "n_val_datapoints": len(VAL_IDX),
        "model": tmpl, "target": tkey, "seed": SEED, "test_r2": test_r2,
    }])
    existing = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined = combined.drop_duplicates(subset=["data_fraction","model","target"])
    combined.to_csv(OUT_CSV, index=False)

print(f"Worker {WORKER_ID} DONE", flush=True)
