import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import torch, torch.nn as nn
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "results" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(BASE / "data/CoCrCuFeNi_684.csv").rename(columns={"T_K": "T"})
feat = pd.read_csv(BASE / "data/features_13.csv")

CN = ["Co","Cr","Cu","Fe","Ni"]
Xc = pd.DataFrame({n: data[f"{n}(%)"] for n in CN}, dtype=float)
X_full = pd.concat([Xc, feat], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
X_full["T"] = data["T"].values
N_FEAT = X_full.shape[1]

groups = (data["Co(%)"].astype(str)+"_"+data["Cr(%)"].astype(str)+"_"+
          data["Cu(%)"].astype(str)+"_"+data["Fe(%)"].astype(str)+"_"+data["Ni(%)"].astype(str)).values

unique_g = list(dict.fromkeys(groups))
np.random.seed(0)
np.random.shuffle(unique_g)
val_g = set(unique_g[:int(0.2*len(unique_g))])
val_mask = np.array([g in val_g for g in groups])

ALL_TARGETS = {
    "Young's modulus": "Young's modulus(Gpa)",
    "UTS": "UTS(Gpa)",
    "Disloc 0%": "Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc 20%": "Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC 0%": "FCC 0%(%)",
    "HCP 0%": "HCP 0%(%)",
    "BCC 0%": "BCC 0%(%)",
    "Other 0%": "Other 0%(%)",
    "FCC 20%": "FCC 20%(%)",
    "HCP 20%": "HCP 20%(%)",
    "BCC 20%": "BCC 20%(%)",
    "Other 20%": "Other 20%(%)",
}

SEED = 0
N_ML = 50
N_DL = 30

# ── ML ──────────────────────────────────────────────────────────────────────
ML_MODELS = ["RandomForest","GradientBoosting","XGBoost","SVR","Ridge","Lasso","MLP"]
ML_COLORS = ["#2196F3","#FF9800","#9C27B0","#4CAF50","#F44336","#00BCD4","#795548"]

def make_ml(name, trial):
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_est",100,800),
            max_depth=trial.suggest_int("max_d",3,20),
            min_samples_leaf=trial.suggest_int("msl",1,5), random_state=SEED, n_jobs=-1)
    elif name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_est",100,800),
            max_depth=trial.suggest_int("max_d",2,8),
            learning_rate=trial.suggest_float("lr",0.01,0.3,log=True),
            subsample=trial.suggest_float("sub",0.6,1.0), random_state=SEED)
    elif name == "XGBoost":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_est",100,800),
            max_depth=trial.suggest_int("max_d",2,8),
            learning_rate=trial.suggest_float("lr",0.01,0.3,log=True),
            subsample=trial.suggest_float("sub",0.6,1.0),
            colsample_bytree=trial.suggest_float("cbt",0.6,1.0),
            random_state=SEED, verbosity=0, n_jobs=-1)
    elif name == "SVR":
        return Pipeline([("sc",StandardScaler()),("m",SVR(
            C=trial.suggest_float("C",0.1,1000,log=True),
            gamma=trial.suggest_categorical("gamma",["scale","auto"]),
            epsilon=trial.suggest_float("eps",0.001,1.0,log=True)))])
    elif name == "Ridge":
        return Pipeline([("sc",StandardScaler()),("m",
            Ridge(alpha=trial.suggest_float("alpha",0.001,100,log=True)))])
    elif name == "Lasso":
        return Pipeline([("sc",StandardScaler()),("m",
            Lasso(alpha=trial.suggest_float("alpha",0.0001,1.0,log=True),max_iter=5000))])
    elif name == "MLP":
        n1=trial.suggest_categorical("n1",[64,128,256])
        n2=trial.suggest_categorical("n2",[32,64,128])
        return Pipeline([("sc",StandardScaler()),("m",MLPRegressor(
            hidden_layer_sizes=(n1,n2),
            alpha=trial.suggest_float("alpha",1e-5,0.1,log=True),
            learning_rate_init=trial.suggest_float("lr",1e-4,1e-2,log=True),
            max_iter=2000,random_state=SEED,early_stopping=True,n_iter_no_change=20))])

# ── DL ──────────────────────────────────────────────────────────────────────
class DeepMLP(nn.Module):
    def __init__(self,n,h1,h2,h3,dr):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n,h1),nn.BatchNorm1d(h1),nn.ReLU(),nn.Dropout(dr),
            nn.Linear(h1,h2),nn.BatchNorm1d(h2),nn.ReLU(),nn.Dropout(dr),
            nn.Linear(h2,h3),nn.ReLU(),nn.Linear(h3,1))
    def forward(self,x): return self.net(x).squeeze(-1)

class AttentionMLP(nn.Module):
    def __init__(self,n,ed,nh):
        super().__init__()
        self.proj=nn.Linear(n,ed)
        self.attn=nn.MultiheadAttention(ed,nh,batch_first=True)
        self.out=nn.Sequential(nn.Linear(ed,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        h=self.proj(x).unsqueeze(1); h,_=self.attn(h,h,h)
        return self.out(h.squeeze(1)).squeeze(-1)

class TabularTransformer(nn.Module):
    def __init__(self,n,dm,nh,nl):
        super().__init__()
        self.proj=nn.Linear(n,dm)
        enc=nn.TransformerEncoderLayer(dm,nh,dim_feedforward=dm*4,batch_first=True,dropout=0.1)
        self.tf=nn.TransformerEncoder(enc,nl)
        self.out=nn.Sequential(nn.Linear(dm,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        return self.out(self.tf(self.proj(x).unsqueeze(1)).squeeze(1)).squeeze(-1)

def train_dl(model, Xt, yt, Xv, yv, lr, wd, epochs=500, patience=60):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=15,factor=0.5)
    best,cnt=-1e9,0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        ((model(Xt)-yt)**2).mean().backward(); opt.step()
        model.eval()
        with torch.no_grad(): vp=model(Xv).numpy()
        vr2=float(r2_score(yv.numpy(),vp))
        sch.step(-vr2)
        if vr2>best: best,cnt=vr2,0
        else:
            cnt+=1
            if cnt>=patience: break
    return best

DL_MODELS = ["DeepMLP","AttentionMLP","TabularTransformer"]
DL_COLORS = ["#E91E63","#FF5722","#607D8B"]

def make_dl(name, trial):
    if name=="DeepMLP":
        return DeepMLP(N_FEAT,
            trial.suggest_categorical("h1",[64,128,256,512]),
            trial.suggest_categorical("h2",[32,64,128,256]),
            trial.suggest_categorical("h3",[16,32,64]),
            trial.suggest_float("dr",0.0,0.4)), \
            trial.suggest_float("lr",1e-4,1e-2,log=True), \
            trial.suggest_float("wd",1e-5,1e-2,log=True)
    elif name=="AttentionMLP":
        ed=trial.suggest_categorical("ed",[16,24,32,48])
        nh=trial.suggest_categorical("nh",[2,4])
        if ed%nh!=0: ed=nh*max(ed//nh,1)
        return AttentionMLP(N_FEAT,ed,nh), \
            trial.suggest_float("lr",1e-4,1e-2,log=True), \
            trial.suggest_float("wd",1e-5,1e-2,log=True)
    elif name=="TabularTransformer":
        dm=trial.suggest_categorical("dm",[16,32,64])
        nh=trial.suggest_categorical("nh",[2,4])
        if dm%nh!=0: dm=nh*max(dm//nh,1)
        return TabularTransformer(N_FEAT,dm,nh,trial.suggest_int("nl",1,3)), \
            trial.suggest_float("lr",1e-4,1e-2,log=True), \
            trial.suggest_float("wd",1e-5,1e-2,log=True)

# ── Run ──────────────────────────────────────────────────────────────────────
ml_conv = {m:{tk:[] for tk in ALL_TARGETS} for m in ML_MODELS}
dl_conv = {m:{tk:[] for tk in ALL_TARGETS} for m in DL_MODELS}

sc_global = StandardScaler()
X_tr_s = sc_global.fit_transform(X_full[~val_mask])
X_va_s = sc_global.transform(X_full[val_mask])

for i, (tkey, tcol) in enumerate(ALL_TARGETS.items()):
    y = data[tcol].values.astype(float)
    y_tr, y_va = y[~val_mask], y[val_mask]
    print(f"\n=== [{i+1}/12] {tkey} ===", flush=True)

    for mname in ML_MODELS:
        bv = [-np.inf]
        hist = []
        def obj_ml(trial, _bv=bv, _hist=hist, _mname=mname):
            m = make_ml(_mname, trial)
            m.fit(X_tr_s, y_tr)
            r2 = float(r2_score(y_va, m.predict(X_va_s)))
            if r2 > _bv[0]: _bv[0] = r2
            _hist.append(_bv[0]); return r2
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(obj_ml, n_trials=N_ML, timeout=300)
        ml_conv[mname][tkey] = hist
        print(f"  ML {mname}: {bv[0]:.4f} ({len(hist)} trials)", flush=True)

    Xt = torch.tensor(X_tr_s, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va_s, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)

    for mname in DL_MODELS:
        bv = [-np.inf]
        hist = []
        def obj_dl(trial, _bv=bv, _hist=hist, _mname=mname):
            torch.manual_seed(SEED)
            model, lr, wd = make_dl(_mname, trial)
            r2 = train_dl(model, Xt, yt, Xv, yv, lr, wd)
            if r2 > _bv[0]: _bv[0] = r2
            _hist.append(_bv[0]); return r2
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(obj_dl, n_trials=N_DL, timeout=600)
        dl_conv[mname][tkey] = hist
        print(f"  DL {mname}: {bv[0]:.4f} ({len(hist)} trials)", flush=True)

# Save raw data
with open(OUTDIR / "optuna_conv_data.json","w") as f:
    json.dump({"ml": ml_conv, "dl": dl_conv}, f)
print(f"\nData saved to {OUTDIR / 'optuna_conv_data.json'}")

# ── Plot ──────────────────────────────────────────────────────────────────────
tlist = list(ALL_TARGETS.keys())
nrows, ncols = 3, 4

def make_panel(conv_dict, models, colors, n_trials, title_prefix, fname):
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
    axes = axes.flatten()
    for idx, tkey in enumerate(tlist):
        ax = axes[idx]
        for mname, color in zip(models, colors):
            v = conv_dict[mname][tkey]
            ax.plot(range(1, len(v)+1), v, color=color, lw=1.8, label=mname)
        ax.set_title(tkey, fontsize=10, fontweight="bold")
        ax.set_xlabel("Trial", fontsize=9)
        ax.set_ylabel("Best val R²", fontsize=9)
        ax.set_xlim(1, n_trials)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Supplementary: Optuna TPE Convergence — {title_prefix} ({n_trials} trials, seed=0)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

make_panel(ml_conv, ML_MODELS, ML_COLORS, N_ML, "ML models (7)",
           OUTDIR / "optuna_conv_ML.png")
make_panel(dl_conv, DL_MODELS, DL_COLORS, N_DL, "DL models (3)",
           OUTDIR / "optuna_conv_DL.png")
