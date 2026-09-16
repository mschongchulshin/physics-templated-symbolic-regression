import pandas as pd, numpy as np, json, os, tempfile, time, threading, sys
from pathlib import Path
from pysr import PySRRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

SEED_START = int(sys.argv[1])
SEED_END = int(sys.argv[2])
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

REPO = Path(__file__).resolve().parent.parent
data = pd.read_csv(REPO / "data/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})
data["comp_id"] = (data["Co(%)"].astype(str)+"_"+data["Cr(%)"].astype(str)+"_"+data["Cu(%)"].astype(str)+"_"+data["Fe(%)"].astype(str)+"_"+data["Ni(%)"].astype(str))
groups = data["comp_id"].values
Ta = data["T"].values.astype(np.float64)

CN = ["Co","Cr","Cu","Fe","Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(REPO / "data/features_13.csv")
X_full = pd.concat([Xc, feat_df], axis=1)
X_full = X_full.loc[:, ~X_full.columns.duplicated()]

R_gas = 8.314
comp_fracs = Xc.values / 100.0
S_mix = np.zeros(len(data))
for i in range(len(data)):
    s = 0
    for j in range(5):
        if comp_fracs[i,j] > 0:
            s -= comp_fracs[i,j] * np.log(comp_fracs[i,j])
    S_mix[i] = R_gas * s

X_no_T = X_full.drop(columns=["T"], errors='ignore')

ALL_TARGETS = {
    "Youngs_modulus":"Young's modulus(Gpa)",
    "UTS":"UTS(Gpa)",
    "Disloc_0pct":"Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc_20pct":"Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC_0pct":"FCC 0%(%)", "HCP_0pct":"HCP 0%(%)",
    "BCC_0pct":"BCC 0%(%)", "FCC_20pct":"FCC 20%(%)",
    "HCP_20pct":"HCP 20%(%)", "BCC_20pct":"BCC 20%(%)",
    "Other_0pct":"Other 0%(%)", "Other_20pct":"Other 20%(%)"
}
TEMPLATES = ['additive','multiplicative','thermal_softening','arrhenius','power_law','free_2stage','free_1stage','entropy_weighted','calphad']

RES_DIR = (Path(__file__).resolve().parent.parent) / "results"
os.makedirs(RES_DIR, exist_ok=True)
CP_FILE = RES_DIR / f"checkpoint_shuffled_s{SEED_START}to{SEED_END-1}.json"
completed = json.load(open(CP_FILE)) if os.path.exists(CP_FILE) else {}
def save_cp():
    with open(CP_FILE,"w") as f: json.dump(completed,f,indent=2,default=str)

gkf = GroupKFold(n_splits=5)
total = len(ALL_TARGETS)*len(TEMPLATES)*5*(SEED_END-SEED_START)
print(f"Shuffled-y permutation test. Seeds {SEED_START}-{SEED_END-1}. Total: {total}")

for tkey, tcol in ALL_TARGETS.items():
    y_real = data[tcol].values.astype(np.float64)
    for tn in TEMPLATES:
        for fold, (tri, tei) in enumerate(gkf.split(X_full, y_real, groups)):
            for seed in range(SEED_START, SEED_END):
                rk = f"{tkey}__{tn}__f{fold}__s{seed}__shuffled"
                if rk in completed:
                    continue
                rng = np.random.RandomState(seed * 1000 + fold)
                y_shuffled = y_real.copy()
                y_shuffled[tri] = rng.permutation(y_real[tri])

                print(f"[shuffled] {tkey} {tn} f{fold}s{seed}...", end=" ", flush=True)
                t0 = time.time()
                start_watchdog()
                try:
                    m300 = Ta[tri] == 300
                    tmpdir = os.path.join(tempfile.gettempdir(),
                                          f"shuf_{tkey}_{tn}_f{fold}_s{seed}")
                    os.makedirs(tmpdir, exist_ok=True)

                    if tn == 'free_1stage':
                        from pysr import PySRRegressor as PSR
                        m = PSR(niterations=200, binary_operators=["+","-","*","/"],
                                unary_operators=["square","cube","sqrt","log","exp"],
                                populations=40, population_size=60, maxsize=25,
                                parsimony=0.005, turbo=True, bumper=True,
                                elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                temp_equation_file=True, tempdir=tmpdir,
                                random_state=seed, deterministic=False, procs=1, verbosity=0)
                        m.fit(X_full.iloc[tri], y_shuffled[tri])
                        pred = m.predict(X_full.iloc[tei])
                    else:
                        td1 = tmpdir + "/g"; os.makedirs(td1, exist_ok=True)
                        gm = PySRRegressor(niterations=200, binary_operators=["+","-","*","/"],
                                unary_operators=["square","cube","sqrt","log"],
                                populations=40, population_size=60, maxsize=20,
                                parsimony=0.005, turbo=True, bumper=True,
                                elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                temp_equation_file=True, tempdir=td1,
                                random_state=seed, deterministic=False, procs=1, verbosity=0)
                        gm.fit(X_no_T.iloc[tri][m300], y_shuffled[tri][m300])
                        gtr = gm.predict(X_no_T.iloc[tri])
                        gte = gm.predict(X_no_T.iloc[tei])
                        td2 = tmpdir + "/h"; os.makedirs(td2, exist_ok=True)

                        if tn == 'arrhenius':
                            gs = np.clip(np.abs(gtr), 1e-10, None)
                            ys = np.clip(np.abs(y_shuffled[tri]), 1e-10, None)
                            lr = np.nan_to_num(np.log(ys)-np.log(gs), nan=0, posinf=0, neginf=0)
                            invT = 1.0/Ta[tri]
                            Xar = pd.concat([X_no_T.iloc[tri].reset_index(drop=True), pd.DataFrame({"inv_T": invT})], axis=1)
                            hm = PySRRegressor(niterations=200, binary_operators=["+","-","*","/"],
                                    unary_operators=["square","cube","sqrt","log","exp"],
                                    populations=40, population_size=60, maxsize=15, parsimony=0.01,
                                    turbo=True, bumper=True,
                                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                    temp_equation_file=True, tempdir=td2,
                                    random_state=seed, deterministic=False, procs=1, verbosity=0)
                            hm.fit(Xar, lr)
                            invTe = 1.0/Ta[tei]
                            Xare = pd.concat([X_no_T.iloc[tei].reset_index(drop=True), pd.DataFrame({"inv_T": invTe})], axis=1)
                            pred = np.clip(np.abs(gte), 1e-10, None) * np.exp(np.clip(hm.predict(Xare), -20, 20))
                        elif tn == 'additive':
                            res = y_shuffled[tri] - gtr
                            Tdf = pd.DataFrame(Ta[tri].reshape(-1,1), columns=["T"])
                            hm = PySRRegressor(niterations=200, binary_operators=["+","-","*","/"],
                                    unary_operators=["square","cube","sqrt","log","exp"],
                                    populations=40, population_size=60, maxsize=15, parsimony=0.005,
                                    turbo=True, bumper=True,
                                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                    temp_equation_file=True, tempdir=td2,
                                    random_state=seed, deterministic=False, procs=1, verbosity=0)
                            hm.fit(Tdf, res)
                            pred = gte + hm.predict(pd.DataFrame(Ta[tei].reshape(-1,1), columns=["T"]))
                        elif tn in ['multiplicative','power_law']:
                            gs2 = np.where(np.abs(gtr)<1e-10, 1e-10, gtr)
                            ratio = np.nan_to_num(y_shuffled[tri]/gs2, nan=1, posinf=1, neginf=1)
                            Tdf = pd.DataFrame(Ta[tri].reshape(-1,1), columns=["T"])
                            hm = PySRRegressor(niterations=200, binary_operators=["+","-","*","/"],
                                    unary_operators=["square","cube","sqrt","log","exp"],
                                    populations=40, population_size=60, maxsize=15, parsimony=0.005,
                                    turbo=True, bumper=True,
                                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                    temp_equation_file=True, tempdir=td2,
                                    random_state=seed, deterministic=False, procs=1, verbosity=0)
                            hm.fit(Tdf, ratio)
                            pred = gte * hm.predict(pd.DataFrame(Ta[tei].reshape(-1,1), columns=["T"]))
                        elif tn == 'thermal_softening':
                            gs2 = np.where(np.abs(gtr)<1e-10, 1e-10, gtr)
                            ratio = np.nan_to_num(y_shuffled[tri]/gs2, nan=1, posinf=1, neginf=1)
                            XcT = pd.concat([X_no_T.iloc[tri].reset_index(drop=True), pd.DataFrame({"T": Ta[tri]})], axis=1)
                            hm = PySRRegressor(niterations=200, binary_operators=["+","-","*","/"],
                                    unary_operators=["square","cube","sqrt","log","exp"],
                                    populations=40, population_size=60, maxsize=15, parsimony=0.01,
                                    turbo=True, bumper=True,
                                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                    temp_equation_file=True, tempdir=td2,
                                    random_state=seed, deterministic=False, procs=1, verbosity=0)
                            hm.fit(XcT, ratio)
                            XcTe = pd.concat([X_no_T.iloc[tei].reset_index(drop=True), pd.DataFrame({"T": Ta[tei]})], axis=1)
                            pred = gte * hm.predict(XcTe)
                        elif tn == 'calphad':
                            residual = (y_shuffled[tri] - gtr) / (-Ta[tri])
                            residual = np.nan_to_num(residual, nan=0, posinf=0, neginf=0)
                            hm = PySRRegressor(niterations=200, binary_operators=["+","-","*","/"],
                                    unary_operators=["square","cube","sqrt","log","exp"],
                                    populations=40, population_size=60, maxsize=15, parsimony=0.01,
                                    turbo=True, bumper=True,
                                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                    temp_equation_file=True, tempdir=td2,
                                    random_state=seed, deterministic=False, procs=1, verbosity=0)
                            hm.fit(X_no_T.iloc[tri], residual)
                            pred = gte - Ta[tei] * hm.predict(X_no_T.iloc[tei])
                        elif tn == 'entropy_weighted':
                            gs = np.clip(np.abs(gtr), 1e-10, None)
                            ys = np.clip(np.abs(y_shuffled[tri]), 1e-10, None)
                            lr = np.nan_to_num(np.log(ys)-np.log(gs), nan=0, posinf=0, neginf=0)
                            invT = 1.0/Ta[tri]
                            Xent = pd.DataFrame({"S_mix": S_mix[tri], "inv_T": invT})
                            hm = PySRRegressor(niterations=200, binary_operators=["+","-","*","/"],
                                    unary_operators=["square","cube","sqrt","log","exp"],
                                    populations=40, population_size=60, maxsize=15, parsimony=0.01,
                                    turbo=True, bumper=True,
                                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                    temp_equation_file=True, tempdir=td2,
                                    random_state=seed, deterministic=False, procs=1, verbosity=0)
                            hm.fit(Xent, lr)
                            invTe = 1.0/Ta[tei]
                            Xent_te = pd.DataFrame({"S_mix": S_mix[tei], "inv_T": invTe})
                            pred = np.clip(np.abs(gte), 1e-10, None) * np.exp(np.clip(hm.predict(Xent_te), -20, 20))
                        elif tn == 'free_2stage':
                            gT = pd.DataFrame({"g": gtr, "T": Ta[tri]})
                            hm = PySRRegressor(niterations=300, binary_operators=["+","-","*","/"],
                                    unary_operators=["square","cube","sqrt","log","exp"],
                                    populations=40, population_size=60, maxsize=25, parsimony=0.003,
                                    turbo=True, bumper=True,
                                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                                    temp_equation_file=True, tempdir=td2,
                                    random_state=seed, deterministic=False, procs=1, verbosity=0)
                            hm.fit(gT, y_shuffled[tri])
                            pred = hm.predict(pd.DataFrame({"g": gte, "T": Ta[tei]}))

                    cancel_watchdog()
                    r2 = float(r2_score(y_real[tei], pred))
                    print(f"R²={r2:.4f} [{time.time()-t0:.0f}s]")
                    completed[rk] = {"test_r2": r2, "time": time.time()-t0}
                    save_cp()
                except Exception as e:
                    cancel_watchdog()
                    print(f"FAIL [{time.time()-t0:.0f}s]")
                    completed[rk] = {"test_r2": -1, "error": str(e)}
                    save_cp()

print(f"\nDone! Total: {len(completed)}")
