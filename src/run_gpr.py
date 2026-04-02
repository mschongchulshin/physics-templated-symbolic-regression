"""
GPR baseline with 13-feature set (Set A), 5 seeds × 5-fold CV
Outputs: results/gpr_results.json
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import json, os

REPO  = Path(__file__).resolve().parent.parent
SEEDS = [0, 1, 2, 3, 4]

data = pd.read_csv(REPO / "data/raw/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})

CN = ["Co", "Cr", "Cu", "Fe", "Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=np.float64)
feat_df = pd.read_csv(REPO / "data/features/features_13.csv")
X = pd.concat([Xc, feat_df], axis=1)
X = X.loc[:, ~X.columns.duplicated()]
print(f"Features ({X.shape[1]}): {list(X.columns)}")

groups = (data["Co(%)"].astype(str) + "_" + data["Cr(%)"].astype(str) + "_" +
          data["Cu(%)"].astype(str) + "_" + data["Fe(%)"].astype(str) + "_" +
          data["Ni(%)"].astype(str)).values

ALL_TARGETS = {
    "Youngs_modulus": "Young's modulus(Gpa)",
    "UTS": "UTS(Gpa)",
    "Disloc_0pct": "Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc_20pct": "Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC_0pct": "FCC 0%(%)", "HCP_0pct": "HCP 0%(%)",
    "BCC_0pct": "BCC 0%(%)", "FCC_20pct": "FCC 20%(%)",
    "HCP_20pct": "HCP 20%(%)", "BCC_20pct": "BCC 20%(%)",
    "Other_0pct": "Other 0%(%)", "Other_20pct": "Other 20%(%)"
}

gkf = GroupKFold(n_splits=5)
results = {}

for tkey, tcol in ALL_TARGETS.items():
    y = data[tcol].values.astype(np.float64)
    seed_fold_r2s = []
    for seed in SEEDS:
        fold_r2s = []
        for fold, (tri, tei) in enumerate(gkf.split(X, y, groups)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X.iloc[tri])
            X_te = scaler.transform(X.iloc[tei])
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=seed)
            gpr.fit(X_tr, y[tri])
            pred = gpr.predict(X_te)
            fold_r2s.append(float(r2_score(y[tei], pred)))
        seed_fold_r2s.append(fold_r2s)
    all_r2s = [r for folds in seed_fold_r2s for r in folds]
    results[tkey] = {
        "mean_r2": float(np.mean(all_r2s)),
        "std_r2": float(np.std(all_r2s)),
        "seed_fold_r2s": seed_fold_r2s
    }
    print(f"  {tkey}: {np.mean(all_r2s):.4f} ± {np.std(all_r2s):.4f}")

OUT = REPO / "results" / "gpr_results.json"
os.makedirs(OUT.parent, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT}")
