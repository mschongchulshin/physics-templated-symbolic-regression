"""
ML Optuna CV re-run
Loads best HP from ml_optuna_results_13feat.json (per target × model)
Runs 5-fold GroupKFold × 5 seeds CV with those HP
Output: ml_optuna_cv_results_13feat.json (same format as ML_Baselines)
"""
import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBRegressor

REPO      = Path(__file__).resolve().parent.parent
SEEDS     = [0, 1, 2, 3, 4]
FOLDS     = [0, 1, 2, 3, 4]
OUT       = REPO / "results" / "ml_optuna_cv_results.json"
OPT_FILE  = REPO / "results" / "ml_optuna_results.json"

data = pd.read_csv(REPO / "data/raw/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})

CN = ["Co","Cr","Cu","Fe","Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(REPO / "data/features/features_13.csv")
X = pd.concat([Xc, feat_df], axis=1)
X = X.loc[:, ~X.columns.duplicated()]
print(f"Features ({X.shape[1]}): {list(X.columns)}")

groups = (data["Co(%)"].astype(str)+"_"+data["Cr(%)"].astype(str)+"_"+
          data["Cu(%)"].astype(str)+"_"+data["Fe(%)"].astype(str)+"_"+
          data["Ni(%)"].astype(str)).values

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

gkf = GroupKFold(n_splits=5)
opt_results = json.load(open(OPT_FILE))
TUNABLE_MODELS = ["RandomForest", "GradientBoosting", "XGBoost", "SVR", "Ridge", "Lasso", "MLP"]


def make_model(mname, params, seed):
    if mname == "RandomForest":
        return RandomForestRegressor(**{k: v for k,v in params.items()}, random_state=seed, n_jobs=-1)
    elif mname == "GradientBoosting":
        return GradientBoostingRegressor(**{k: v for k,v in params.items()}, random_state=seed)
    elif mname == "XGBoost":
        return XGBRegressor(**{k: v for k,v in params.items()}, random_state=seed, n_jobs=-1, verbosity=0)
    elif mname == "SVR":
        p = params
        return Pipeline([("scaler", StandardScaler()),
                         ("model", SVR(C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"]))])
    elif mname == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=params["alpha"]))])
    elif mname == "Lasso":
        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=params["alpha"]))])
    elif mname == "MLP":
        n1, n2, n3 = params["n1"], params["n2"], params["n3"]
        layers = (n1, n2) if n3 == 0 else (n1, n2, n3)
        return Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(
            hidden_layer_sizes=layers, alpha=params["alpha"],
            learning_rate_init=params["lr"],
            max_iter=1000, early_stopping=True, random_state=seed))])


results = {}
total = len(ALL_TARGETS) * len(TUNABLE_MODELS)
rc = 0

for tkey, tcol in ALL_TARGETS.items():
    y = data[tcol].values.astype(np.float64)
    results[tkey] = {}

    for mname in TUNABLE_MODELS:
        rc += 1
        best_params = opt_results[tkey][mname]["best_params"]
        print(f"[{rc}/{total}] {tkey} {mname} params={best_params}", flush=True)

        seed_fold_r2s = []
        for seed in SEEDS:
            fold_r2s = []
            for tri, tei in gkf.split(X, y, groups):
                m = make_model(mname, best_params, seed)
                m.fit(X.iloc[tri], y[tri])
                fold_r2s.append(float(r2_score(y[tei], m.predict(X.iloc[tei]))))
            seed_fold_r2s.append(fold_r2s)

        all_r2s = [r for folds in seed_fold_r2s for r in folds]
        results[tkey][mname] = {
            "mean_r2": float(np.mean(all_r2s)),
            "std_r2":  float(np.std(all_r2s)),
            "seed_fold_r2s": seed_fold_r2s,
            "best_params": best_params,
        }
        print(f"  CV R²: {np.mean(all_r2s):.4f} ± {np.std(all_r2s):.4f}")

os.makedirs(Path(str(OUT)).parent, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
