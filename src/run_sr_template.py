"""
9-template SR — full-data fit (no CV), 5 seeds
Saves structured JSON + pkl per run
Usage: python3 run_9templates_fullfit.py <seed_start> <seed_end>
"""
import pandas as pd, numpy as np, json, os, time, threading, sys, pickle
from pathlib import Path
from pysr import PySRRegressor
from sklearn.metrics import r2_score

SEED_START     = int(sys.argv[1])
SEED_END       = int(sys.argv[2])
TEMPLATE_FILTER = sys.argv[3] if len(sys.argv) > 3 else None  # optional single template
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
EQ_DIR  = REPO / "equations"
RES_DIR = REPO / "results"
TMP_DIR = REPO / "tmp"
os.makedirs(RES_DIR, exist_ok=True)

data = pd.read_csv(REPO / "data/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})
Ta   = data["T"].values.astype(np.float64)

CN = ["Co","Cr","Cu","Fe","Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(REPO / "data/features_13.csv")
X_full = pd.concat([Xc, feat_df], axis=1)
X_full = X_full.loc[:, ~X_full.columns.duplicated()]
print(f"Features ({X_full.shape[1]}): {list(X_full.columns)}")

R_gas = 8.314
comp_fracs = Xc.values / 100.0
S_mix = np.zeros(len(data))
for i in range(len(data)):
    s = 0
    for j in range(5):
        if comp_fracs[i, j] > 0:
            s -= comp_fracs[i, j] * np.log(comp_fracs[i, j])
    S_mix[i] = R_gas * s

X_no_T = X_full.drop(columns=["T"], errors="ignore")

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

ALL_TEMPLATES = [
    "additive", "multiplicative", "thermal_softening", "arrhenius",
    "power_law", "free_2stage", "free_1stage", "entropy_weighted", "calphad",
]
TEMPLATES = [TEMPLATE_FILTER] if TEMPLATE_FILTER else ALL_TEMPLATES

# Progress checkpoint (simple done-set, fast to check)
tmpl_tag = f"_{TEMPLATE_FILTER}" if TEMPLATE_FILTER else ""
CP_FILE = RES_DIR / f"done_fullfit{tmpl_tag}_s{SEED_START}to{SEED_END-1}.json"
done = set(json.load(open(CP_FILE))) if CP_FILE.exists() else set()
def save_done():
    with open(CP_FILE, "w") as f: json.dump(list(done), f)


def feat_mapping(cols):
    return {f"x{i}": c for i, c in enumerate(cols)}


def extract_pareto(model, X_fit, y_fit):
    """Extract pareto front from model.equations_ with r2/mae per equation."""
    pareto = []
    if model.equations_ is None:
        return pareto
    for idx, row in model.equations_.iterrows():
        entry = {
            "complexity": int(row["complexity"]),
            "equation": str(row["equation"]),
            "loss": float(row["loss"]),
        }
        try:
            eq_pred = model.predict(X_fit, index=int(idx))
            entry["train_r2"] = float(r2_score(y_fit, eq_pred))
            entry["train_mae"] = float(np.mean(np.abs(y_fit - eq_pred)))
        except Exception:
            pass
        pareto.append(entry)
    return pareto


def best_info(model, X_fit, y_fit):
    """Return dict with best equation info."""
    best_pred = model.predict(X_fit)
    info = {
        "train_r2":  float(r2_score(y_fit, best_pred)),
        "train_mae": float(np.mean(np.abs(y_fit - best_pred))),
    }
    try:
        best_row = model.get_best()
        info["complexity"]      = int(best_row["complexity"])
        info["equation_sympy"]  = str(model.sympy())
        info["equation_latex"]  = str(model.latex())
        info["pysr_raw_string"] = str(best_row["equation"])
    except Exception:
        pass
    return info


def save_pkl(model, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"  [pkl warn] {e}")


total = len(ALL_TARGETS) * len(TEMPLATES) * (SEED_END - SEED_START)
rc = 0; nc = 0
print(f"Seeds {SEED_START}-{SEED_END-1}, {len(done)} cached. Total: {total}")

for tkey, tcol in ALL_TARGETS.items():
    y = data[tcol].values.astype(np.float64)
    for tn in TEMPLATES:
        tmpl_eq_dir = EQ_DIR / tn
        os.makedirs(tmpl_eq_dir, exist_ok=True)

        for seed in range(SEED_START, SEED_END):
            rk = f"{tkey}__{tn}__s{seed}"; rc += 1
            if rk in done:
                continue
            nc += 1
            print(f"[{rc}/{total}] {tkey} {tn} s{seed}...", end=" ", flush=True)
            t0 = time.time()
            start_watchdog()
            try:
                tmpdir = str(TMP_DIR / f"ff_{tkey}_{tn}_s{seed}")
                os.makedirs(tmpdir, exist_ok=True)
                m300 = Ta == 300

                result = {
                    "template": tn, "target": tkey, "seed": seed,
                    "feature_set": "13feat",
                    "feature_mapping": feat_mapping(list(X_full.columns)),
                    "stage1_equation": None, "stage2_equation": None,
                    "full_equation": None,
                    "equation_sympy": None, "equation_latex": None,
                    "pysr_raw_string": None,
                    "train_r2": None, "train_mae": None,
                    "complexity": None, "pareto_front": [],
                }

                def pysr(**kwargs):
                    defaults = dict(
                        binary_operators=["+","-","*","/"],
                        unary_operators=["square","cube","sqrt","log","exp"],
                        populations=40, population_size=60,
                        parsimony=0.005, turbo=True, bumper=True,
                        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                        deterministic=False, procs=1, verbosity=0,
                        random_state=seed,
                    )
                    defaults.update(kwargs)
                    return PySRRegressor(**defaults)

                if tn == "free_1stage":
                    model = pysr(niterations=200, maxsize=25, temp_equation_file=True, tempdir=tmpdir)
                    model.fit(X_full, y)
                    pred = model.predict(X_full)
                    b = best_info(model, X_full, y)
                    result.update(b)
                    result["full_equation"]    = b.get("equation_sympy")
                    result["pareto_front"]     = extract_pareto(model, X_full, y)
                    result["feature_mapping"]  = feat_mapping(list(X_full.columns))
                    save_pkl(model, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}.pkl")

                elif tn == "entropy_weighted":
                    td1 = str(TMP_DIR / f"ff_{tkey}_{tn}_g_s{seed}"); os.makedirs(td1, exist_ok=True)
                    gm = pysr(niterations=200, maxsize=20, parsimony=0.005,
                              unary_operators=["square","cube","sqrt","log"],
                              temp_equation_file=True, tempdir=td1)
                    gm.fit(X_no_T[m300], y[m300])
                    g_all = gm.predict(X_no_T)
                    gs = np.clip(np.abs(g_all), 1e-10, None)
                    ys = np.clip(np.abs(y), 1e-10, None)
                    lr = np.nan_to_num(np.log(ys) - np.log(gs), nan=0, posinf=0, neginf=0)
                    td2 = str(TMP_DIR / f"ff_{tkey}_{tn}_h_s{seed}"); os.makedirs(td2, exist_ok=True)
                    Xent = pd.DataFrame({"S_mix": S_mix, "inv_T": 1.0/Ta})
                    em = pysr(niterations=200, maxsize=15, parsimony=0.01,
                              temp_equation_file=True, tempdir=td2)
                    em.fit(Xent, lr)
                    pred = gs * np.exp(np.clip(em.predict(Xent), -20, 20))
                    b_g = best_info(gm, X_no_T[m300], y[m300])
                    b_h = best_info(em, Xent, lr)
                    result["stage1_equation"] = b_g.get("equation_sympy")
                    result["stage2_equation"] = b_h.get("equation_sympy")
                    result["full_equation"]   = f"exp(clip({b_h.get('equation_sympy')}, -20, 20)) * clip(abs({b_g.get('equation_sympy')}), 1e-10)"
                    result["train_r2"]  = float(r2_score(y, pred))
                    result["train_mae"] = float(np.mean(np.abs(y - pred)))
                    result["pareto_front"] = {"g_stage": extract_pareto(gm, X_no_T[m300], y[m300]),
                                              "h_stage": extract_pareto(em, Xent, lr)}
                    result["feature_mapping"] = {"g_stage": feat_mapping(list(X_no_T.columns)),
                                                 "h_stage": feat_mapping(["S_mix","inv_T"])}
                    save_pkl(gm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage1.pkl")
                    save_pkl(em, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage2.pkl")

                elif tn == "calphad":
                    td1 = str(TMP_DIR / f"ff_{tkey}_{tn}_g_s{seed}"); os.makedirs(td1, exist_ok=True)
                    gm = pysr(niterations=200, maxsize=20, parsimony=0.005,
                              unary_operators=["square","cube","sqrt","log"],
                              temp_equation_file=True, tempdir=td1)
                    gm.fit(X_no_T[m300], y[m300])
                    g_all = gm.predict(X_no_T)
                    residual = np.nan_to_num((y - g_all) / (-Ta), nan=0, posinf=0, neginf=0)
                    td2 = str(TMP_DIR / f"ff_{tkey}_{tn}_h_s{seed}"); os.makedirs(td2, exist_ok=True)
                    hm = pysr(niterations=200, maxsize=15, parsimony=0.01,
                              temp_equation_file=True, tempdir=td2)
                    hm.fit(X_no_T, residual)
                    pred = g_all - Ta * hm.predict(X_no_T)
                    b_g = best_info(gm, X_no_T[m300], y[m300])
                    b_h = best_info(hm, X_no_T, residual)
                    result["stage1_equation"] = b_g.get("equation_sympy")
                    result["stage2_equation"] = b_h.get("equation_sympy")
                    result["full_equation"]   = f"({b_g.get('equation_sympy')}) - T * ({b_h.get('equation_sympy')})"
                    result["train_r2"]  = float(r2_score(y, pred))
                    result["train_mae"] = float(np.mean(np.abs(y - pred)))
                    result["pareto_front"] = {"g_stage": extract_pareto(gm, X_no_T[m300], y[m300]),
                                              "h_stage": extract_pareto(hm, X_no_T, residual)}
                    result["feature_mapping"] = feat_mapping(list(X_no_T.columns))
                    save_pkl(gm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage1.pkl")
                    save_pkl(hm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage2.pkl")

                else:
                    # additive, multiplicative, power_law, thermal_softening, arrhenius, free_2stage
                    td1 = str(TMP_DIR / f"ff_{tkey}_{tn}_g_s{seed}"); os.makedirs(td1, exist_ok=True)
                    gm = pysr(niterations=200, maxsize=20, parsimony=0.005,
                              unary_operators=["square","cube","sqrt","log"],
                              temp_equation_file=True, tempdir=td1)
                    gm.fit(X_no_T[m300], y[m300])
                    g_all = gm.predict(X_no_T)
                    td2 = str(TMP_DIR / f"ff_{tkey}_{tn}_h_s{seed}"); os.makedirs(td2, exist_ok=True)

                    if tn == "additive":
                        res = y - g_all
                        Tdf = pd.DataFrame(Ta.reshape(-1, 1), columns=["T"])
                        hm = pysr(niterations=200, maxsize=15, parsimony=0.005,
                                  temp_equation_file=True, tempdir=td2)
                        hm.fit(Tdf, res)
                        pred = g_all + hm.predict(Tdf)
                        b_g = best_info(gm, X_no_T[m300], y[m300])
                        b_h = best_info(hm, Tdf, res)
                        result["stage1_equation"] = b_g.get("equation_sympy")
                        result["stage2_equation"] = b_h.get("equation_sympy")
                        result["full_equation"]   = f"({b_g.get('equation_sympy')}) + ({b_h.get('equation_sympy')})"
                        result["train_r2"]  = float(r2_score(y, pred))
                        result["train_mae"] = float(np.mean(np.abs(y - pred)))
                        result["pareto_front"] = {"g_stage": extract_pareto(gm, X_no_T[m300], y[m300]),
                                                  "h_stage": extract_pareto(hm, Tdf, res)}
                        result["feature_mapping"] = {"g_stage": feat_mapping(list(X_no_T.columns)),
                                                     "h_stage": {"x0": "T"}}
                        save_pkl(gm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage1.pkl")
                        save_pkl(hm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage2.pkl")

                    elif tn in ["multiplicative", "power_law"]:
                        gs = np.where(np.abs(g_all) < 1e-10, 1e-10, g_all)
                        ratio = np.nan_to_num(y / gs, nan=1, posinf=1, neginf=1)
                        Tdf = pd.DataFrame(Ta.reshape(-1, 1), columns=["T"])
                        hm = pysr(niterations=200, maxsize=15, parsimony=0.005,
                                  temp_equation_file=True, tempdir=td2)
                        hm.fit(Tdf, ratio)
                        pred = g_all * hm.predict(Tdf)
                        b_g = best_info(gm, X_no_T[m300], y[m300])
                        b_h = best_info(hm, Tdf, ratio)
                        result["stage1_equation"] = b_g.get("equation_sympy")
                        result["stage2_equation"] = b_h.get("equation_sympy")
                        result["full_equation"]   = f"({b_g.get('equation_sympy')}) * ({b_h.get('equation_sympy')})"
                        result["train_r2"]  = float(r2_score(y, pred))
                        result["train_mae"] = float(np.mean(np.abs(y - pred)))
                        result["pareto_front"] = {"g_stage": extract_pareto(gm, X_no_T[m300], y[m300]),
                                                  "h_stage": extract_pareto(hm, Tdf, ratio)}
                        result["feature_mapping"] = {"g_stage": feat_mapping(list(X_no_T.columns)),
                                                     "h_stage": {"x0": "T"}}
                        save_pkl(gm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage1.pkl")
                        save_pkl(hm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage2.pkl")

                    elif tn == "thermal_softening":
                        gs = np.where(np.abs(g_all) < 1e-10, 1e-10, g_all)
                        ratio = np.nan_to_num(y / gs, nan=1, posinf=1, neginf=1)
                        XcT = pd.concat([X_no_T.reset_index(drop=True), pd.DataFrame({"T": Ta})], axis=1)
                        hm = pysr(niterations=200, maxsize=15, parsimony=0.01,
                                  temp_equation_file=True, tempdir=td2)
                        hm.fit(XcT, ratio)
                        pred = g_all * hm.predict(XcT)
                        b_g = best_info(gm, X_no_T[m300], y[m300])
                        b_h = best_info(hm, XcT, ratio)
                        result["stage1_equation"] = b_g.get("equation_sympy")
                        result["stage2_equation"] = b_h.get("equation_sympy")
                        result["full_equation"]   = f"({b_g.get('equation_sympy')}) * ({b_h.get('equation_sympy')})"
                        result["train_r2"]  = float(r2_score(y, pred))
                        result["train_mae"] = float(np.mean(np.abs(y - pred)))
                        result["pareto_front"] = {"g_stage": extract_pareto(gm, X_no_T[m300], y[m300]),
                                                  "h_stage": extract_pareto(hm, XcT, ratio)}
                        result["feature_mapping"] = {"g_stage": feat_mapping(list(X_no_T.columns)),
                                                     "h_stage": feat_mapping(list(XcT.columns))}
                        save_pkl(gm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage1.pkl")
                        save_pkl(hm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage2.pkl")

                    elif tn == "arrhenius":
                        gs = np.clip(np.abs(g_all), 1e-10, None)
                        ys = np.clip(np.abs(y), 1e-10, None)
                        lr = np.nan_to_num(np.log(ys) - np.log(gs), nan=0, posinf=0, neginf=0)
                        Xar = pd.concat([X_no_T.reset_index(drop=True), pd.DataFrame({"inv_T": 1.0/Ta})], axis=1)
                        em = pysr(niterations=200, maxsize=15, parsimony=0.01,
                                  temp_equation_file=True, tempdir=td2)
                        em.fit(Xar, lr)
                        pred = gs * np.exp(np.clip(em.predict(Xar), -20, 20))
                        b_g = best_info(gm, X_no_T[m300], y[m300])
                        b_h = best_info(em, Xar, lr)
                        result["stage1_equation"] = b_g.get("equation_sympy")
                        result["stage2_equation"] = b_h.get("equation_sympy")
                        result["full_equation"]   = f"clip(abs({b_g.get('equation_sympy')}),1e-10) * exp(clip({b_h.get('equation_sympy')},-20,20))"
                        result["train_r2"]  = float(r2_score(y, pred))
                        result["train_mae"] = float(np.mean(np.abs(y - pred)))
                        result["pareto_front"] = {"g_stage": extract_pareto(gm, X_no_T[m300], y[m300]),
                                                  "h_stage": extract_pareto(em, Xar, lr)}
                        result["feature_mapping"] = {"g_stage": feat_mapping(list(X_no_T.columns)),
                                                     "h_stage": feat_mapping(list(Xar.columns))}
                        save_pkl(gm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage1.pkl")
                        save_pkl(em, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage2.pkl")

                    elif tn == "free_2stage":
                        gT = pd.DataFrame({"g": g_all, "T": Ta})
                        fm = pysr(niterations=300, maxsize=25, parsimony=0.003,
                                  temp_equation_file=True, tempdir=td2)
                        fm.fit(gT, y)
                        pred = fm.predict(gT)
                        b_g = best_info(gm, X_no_T[m300], y[m300])
                        b_f = best_info(fm, gT, y)
                        result["stage1_equation"] = b_g.get("equation_sympy")
                        result["stage2_equation"] = b_f.get("equation_sympy")
                        result["full_equation"]   = b_f.get("equation_sympy")
                        result["train_r2"]  = float(r2_score(y, pred))
                        result["train_mae"] = float(np.mean(np.abs(y - pred)))
                        result["pareto_front"] = {"g_stage": extract_pareto(gm, X_no_T[m300], y[m300]),
                                                  "f_stage": extract_pareto(fm, gT, y)}
                        result["feature_mapping"] = {"g_stage": feat_mapping(list(X_no_T.columns)),
                                                     "f_stage": {"x0": "g(comp)", "x1": "T"}}
                        save_pkl(gm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage1.pkl")
                        save_pkl(fm, f"{MODEL_DIR}/{tn}_{tkey}_seed{seed}_stage2.pkl")

                cancel_watchdog()
                elapsed = time.time() - t0
                print(f"train_R²={result['train_r2']:.4f} [{elapsed:.0f}s]")

                out_path = tmpl_eq_dir / f"{tkey}_seed{seed}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)

                done.add(rk)
                save_done()

            except Exception as e:
                cancel_watchdog()
                print(f"FAIL [{time.time()-t0:.0f}s]: {e}")
                err_path = tmpl_eq_dir / f"{tkey}_seed{seed}_FAILED.json"
                with open(err_path, "w") as f:
                    json.dump({"error": str(e), "template": tn, "target": tkey, "seed": seed}, f)
                done.add(rk)
                save_done()

print(f"\nDone! {nc} new runs.")
