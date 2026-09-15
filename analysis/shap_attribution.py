import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor
import shap
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "results" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)
ELE=["Co","Cr","Cu","Fe","Ni"]
feat=pd.read_csv(BASE / "data" / "features_13.csv")
stat=[c for c in feat.columns if c!="T"]
f300=feat[feat["T"]==300].reset_index(drop=True)
d=pd.read_csv(BASE / "data" / "CoCrCuFeNi_684.csv")
d=d[d["T_K"]==300].drop(columns=["T_K"]).reset_index(drop=True)
assert len(f300)==len(d)
X=pd.concat([d[[f"{e}(%)" for e in ELE]].rename(columns={f"{e}(%)":e for e in ELE}),f300[stat]],axis=1)
y=d["UTS(Gpa)"].values
gbr=GradientBoostingRegressor(random_state=0).fit(X,y)
sv=shap.TreeExplainer(gbr).shap_values(X)

mean_signed=pd.Series(sv.mean(0),index=X.columns)
mean_abs=pd.Series(np.abs(sv).mean(0),index=X.columns)
out=pd.DataFrame({"feature":X.columns,
                  "mean_SHAP_signed":mean_signed.values,
                  "mean_abs_SHAP":mean_abs.values})
out=out.sort_values("mean_abs_SHAP",ascending=False).reset_index(drop=True)
out["rank"]=range(1,len(out)+1)
out.to_csv(OUTDIR / "shap_uts_300K_signed_mean.csv",index=False)

M=pd.DataFrame(sv,columns=[f"SHAP_{c}" for c in X.columns])
MX=pd.concat([X.reset_index(drop=True).add_prefix("x_"),M],axis=1)
MX.to_csv(OUTDIR / "shap_uts_300K_matrix.csv",index=False)

print("GBR train R2=%.4f"%gbr.score(X,y))
print(out.to_string(index=False))
print("\nWROTE shap_uts_300K_signed_mean.csv  and  shap_uts_300K_matrix.csv")
