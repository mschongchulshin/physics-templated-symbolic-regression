"""
Free-form SR — full-data fit (no CV), 13 features, 5 seeds
Usage: python3 src/run_sr_freeform.py <seed_start> <seed_end>
Saves: equations/freeform_13feat/{target}_seed{N}.json + sr_models/
"""
import pandas as pd, numpy as np, json, os, time, threading, sys, pickle
from pathlib import Path
from pysr import PySRRegressor
from sklearn.metrics import r2_score

SEED_START = int(sys.argv[1])
SEED_END   = int(sys.argv[2])
RUN_TIMEOUT = 600

_watchdog_timer = None
def start_watchdog():
    global _watchdog_timer
    cancel_watchdog()
    _watchdog_timer = threading.Timer(RUN_TIMEOUT, lambda: os._exit(99))
    _watchdog_timer.daemon = True
    _watchdog_timer.start()
def cancel_watchdog():
    global _watchdog_timer
    if _watchdog_timer: _watchdog_timer.cancel(); _watchdog_timer = None

REPO    = Path(__file__).resolve().parent.parent
EQ_DIR  = REPO / "equations" / "freeform"
RES_DIR = REPO / "results"
TMP_DIR = REPO / "tmp"
os.makedirs(EQ_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

data = pd.read_csv(REPO / "data/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})

CN = ["Co","Cr","Cu","Fe","Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(REPO / "data/features_13.csv")
X_full = pd.concat([Xc, feat_df], axis=1)
X_full = X_full.loc[:, ~X_full.columns.duplicated()]
feature_names = list(X_full.columns)
print(f"Features ({X_full.shape[1]}): {feature_names}")

ALL_TARGETS = {
    "Youngs_modulus": "Young's modulus(Gpa)",
    "UTS": "UTS(Gpa)",
    "Disloc_0pct": "Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc_20pct": "Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC_0pct": "FCC 0%(%)", "HCP_0pct": "HCP 0%(%)",
    "BCC_0pct": "BCC 0%(%)", "FCC_20pct": "FCC 20%(%)",
    "HCP_20pct": "HCP 20%(%)", "BCC_20pct": "BCC 20%(%)",
    "Other_0pct": "Other 0%(%)", "Other_20pct": "Other 20%(%)",
}

CP_FILE = RES_DIR / f"done_freeform_fullfit_s{SEED_START}to{SEED_END-1}.json"
done = set(json.load(open(CP_FILE))) if CP_FILE.exists() else set()
def save_done():
    with open(CP_FILE, "w") as f: json.dump(list(done), f)

def extract_pareto(model, X_fit, y_fit):
    pareto = []
    if model.equations_ is None:
        return pareto
    for idx, row in model.equations_.iterrows():
        entry = {"complexity": int(row["complexity"]), "equation": str(row["equation"]), "loss": float(row["loss"])}
        try:
            eq_pred = model.predict(X_fit, index=int(idx))
            entry["train_r2"]  = float(r2_score(y_fit, eq_pred))
            entry["train_mae"] = float(np.mean(np.abs(y_fit - eq_pred)))
        except Exception:
            pass
        pareto.append(entry)
    return pareto

total = len(ALL_TARGETS) * (SEED_END - SEED_START)
rc = 0; nc = 0
print(f"Seeds {SEED_START}-{SEED_END-1}, {len(done)} cached. Total: {total}")

for tkey, tcol in ALL_TARGETS.items():
    y = data[tcol].values.astype(np.float64)
    for seed in range(SEED_START, SEED_END):
        rk = f"{tkey}__freeform__s{seed}"; rc += 1
        if rk in done:
            continue
        nc += 1
        print(f"[{rc}/{total}] {tkey} freeform s{seed}...", end=" ", flush=True)
        t0 = time.time()
        start_watchdog()
        try:
            tmpdir = str(TMP_DIR / f"ff_free_{tkey}_s{seed}")
            os.makedirs(tmpdir, exist_ok=True)
            model = PySRRegressor(
                niterations=200, binary_operators=["+","-","*","/"],
                unary_operators=["square","cube","sqrt","log","exp"],
                populations=40, population_size=60, maxsize=25,
                parsimony=0.005, turbo=True, bumper=True,
                elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                temp_equation_file=True, tempdir=tmpdir,
                random_state=seed, deterministic=False, procs=1, verbosity=0)
            model.fit(X_full, y)
            pred = model.predict(X_full)
            cancel_watchdog()

            train_r2  = float(r2_score(y, pred))
            train_mae = float(np.mean(np.abs(y - pred)))

            best_row = model.get_best()
            result = {
                "template": "freeform", "target": tkey, "seed": seed,
                "feature_set": "13feat",
                "feature_mapping": {f"x{i}": c for i, c in enumerate(feature_names)},
                "stage1_equation": None,
                "stage2_equation": None,
                "full_equation":   str(model.sympy()),
                "equation_sympy":  str(model.sympy()),
                "equation_latex":  str(model.latex()),
                "pysr_raw_string": str(best_row["equation"]),
                "train_r2":   train_r2,
                "train_mae":  train_mae,
                "complexity": int(best_row["complexity"]),
                "pareto_front": extract_pareto(model, X_full, y),
            }
            print(f"train_R²={train_r2:.4f} [{time.time()-t0:.0f}s]")

            with open(EQ_DIR / f"{tkey}_seed{seed}.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

            try:
                with open(TMP_DIR / f"freeform_{tkey}_seed{seed}.pkl", "wb") as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"  [pkl warn] {e}")

            done.add(rk); save_done()
        except Exception as e:
            cancel_watchdog()
            print(f"FAIL [{time.time()-t0:.0f}s]: {e}")
            with open(EQ_DIR / f"{tkey}_seed{seed}_FAILED.json", "w") as f:
                json.dump({"error": str(e), "target": tkey, "seed": seed}, f)
            done.add(rk); save_done()

print(f"\nDone! {nc} new runs.")
