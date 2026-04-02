"""
ML baselines with Optuna hyperparameter optimization
13-feature set (Set A), fullfit (no CV)
Optuna: internal 15% val split for HP search
Final: train on ALL data, report train R² × 5 seeds
Output: results/ml_optuna_results.json
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
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO      = Path(__file__).resolve().parent.parent
SEEDS     = [0, 1, 2, 3, 4]
N_TRIALS  = 50
OUT       = REPO / "results" / "ml_optuna_results.json"

data = pd.read_csv(REPO / "data/raw/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})

CN = ["Co","Cr","Cu","Fe","Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(REPO / "data/features/features_13.csv")
X = pd.concat([Xc, feat_df], axis=1)
X = X.loc[:, ~X.columns.duplicated()]
print(f"Features ({X.shape[1]}): {list(X.columns)}")

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

# Internal val split indices (fixed, last 15%)
n_val = max(1, int(len(X) * 0.15))
tr_idx = np.arange(len(X) - n_val)
va_idx = np.arange(len(X) - n_val, len(X))


def make_objective(model_name, X_data, y_data):
    X_tr, y_tr = X_data.iloc[tr_idx], y_data[tr_idx]
    X_va, y_va = X_data.iloc[va_idx], y_data[va_idx]
    def objective(trial):
        if model_name == "RandomForest":
            m = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 800),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=0, n_jobs=-1)
        elif model_name == "GradientBoosting":
            m = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 800),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                random_state=0)
        elif model_name == "XGBoost":
            m = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 800),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=0, n_jobs=-1, verbosity=0)
        elif model_name == "SVR":
            m = Pipeline([("scaler", StandardScaler()), ("model", SVR(
                C=trial.suggest_float("C", 0.1, 1000, log=True),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
                epsilon=trial.suggest_float("epsilon", 0.001, 1.0, log=True)))])
        elif model_name == "Ridge":
            m = Pipeline([("scaler", StandardScaler()),
                          ("model", Ridge(alpha=trial.suggest_float("alpha", 0.001, 100, log=True)))])
        elif model_name == "Lasso":
            m = Pipeline([("scaler", StandardScaler()),
                          ("model", Lasso(alpha=trial.suggest_float("alpha", 0.0001, 1.0, log=True)))])
        elif model_name == "MLP":
            n1 = trial.suggest_categorical("n1", [64, 128, 256])
            n2 = trial.suggest_categorical("n2", [32, 64, 128])
            n3 = trial.suggest_categorical("n3", [0, 32, 64])
            layers = (n1, n2) if n3 == 0 else (n1, n2, n3)
            m = Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(
                hidden_layer_sizes=layers,
                alpha=trial.suggest_float("alpha", 1e-5, 0.1, log=True),
                learning_rate_init=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                max_iter=1000, early_stopping=True, random_state=0))])
        else:
            raise ValueError(model_name)
        mc = clone(m)
        mc.fit(X_tr, y_tr)
        return float(r2_score(y_va, mc.predict(X_va)))
    return objective


def make_model(mname, params, seed):
    if mname == "RandomForest":
        return RandomForestRegressor(**{k: v for k,v in params.items()}, random_state=seed, n_jobs=-1)
    elif mname == "GradientBoosting":
        return GradientBoostingRegressor(**{k: v for k,v in params.items()}, random_state=seed)
    elif mname == "XGBoost":
        return XGBRegressor(**{k: v for k,v in params.items()}, random_state=seed, n_jobs=-1, verbosity=0)
    elif mname == "SVR":
        p = {k: v for k,v in params.items()}
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
            max_iter=1000, early_stopping=False, random_state=seed))])


TUNABLE_MODELS = ["RandomForest", "GradientBoosting", "XGBoost", "SVR", "Ridge", "Lasso", "MLP"]

results = {}
total = len(ALL_TARGETS) * len(TUNABLE_MODELS)
rc = 0

for tkey, tcol in ALL_TARGETS.items():
    y = data[tcol].values.astype(np.float64)
    results[tkey] = {}

    for mname in TUNABLE_MODELS:
        rc += 1
        print(f"[{rc}/{total}] {tkey} {mname} — optuna {N_TRIALS} trials...", flush=True)

        # Optuna search on internal val split
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(mname, X, y),
                       n_trials=N_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        best_val_r2 = study.best_value
        print(f"  best val R²={best_val_r2:.4f}, params={best_params}")

        # Final: train on ALL data, 5 seeds, report train R²
        seed_r2s = []
        for seed in SEEDS:
            m = make_model(mname, best_params, seed)
            m.fit(X, y)
            seed_r2s.append(float(r2_score(y, m.predict(X))))

        results[tkey][mname] = {
            "mean_train_r2": float(np.mean(seed_r2s)),
            "std_train_r2":  float(np.std(seed_r2s)),
            "seed_train_r2s": seed_r2s,
            "best_params": best_params,
            "optuna_best_val_r2": best_val_r2,
        }
        print(f"  Train R²: {np.mean(seed_r2s):.4f} ± {np.std(seed_r2s):.4f}")

os.makedirs(Path(str(OUT)).parent, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
