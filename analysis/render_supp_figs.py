"""
Re-render S1a, S1b, S2, S3 without titles.
svg.fonttype='none' for editable text in PPT/Illustrator.
Output SVGs and source-data CSVs to results/generated/
"""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "results" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "supp"; OUT.mkdir(exist_ok=True)
BASE_FEAT = BASE / "data" / "features"

# ── S1a: full 93-feature correlation heatmap ─────────────────────────────────
print("S1a...", flush=True)
df_full = pd.read_csv(BASE_FEAT / "features_full.csv")
corr_full = df_full.corr(method="pearson")

n = corr_full.shape[0]
figsize = max(14, n * 0.22)
fig, ax = plt.subplots(figsize=(figsize, figsize))
sns.heatmap(
    corr_full, annot=False, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
    square=True, linewidths=0, xticklabels=True, yticklabels=True,
    cbar_kws={"shrink": 0.5, "label": "Pearson r"}, ax=ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5.5)
plt.tight_layout()
fig.savefig(OUT / "S1a_full_corr.svg", bbox_inches="tight")
fig.savefig(OUT / "S1a_full_corr.png", dpi=200, bbox_inches="tight")
plt.close()

# Source data: correlation matrix (93×93)
corr_full.to_csv(OUT / "S1a_source_corr93.csv")
print(f"  S1a done. shape={corr_full.shape}", flush=True)

# ── S1b: VIF>50 selected 10-feature correlation matrix ───────────────────────
print("S1b...", flush=True)
df_vif50 = pd.read_csv(BASE_FEAT / "features_vif50_selected.csv")
corr_vif = df_vif50.corr(method="pearson")

n2 = corr_vif.shape[0]
fs2 = max(8, n2 * 0.9)
fig, ax = plt.subplots(figsize=(fs2, fs2))
sns.heatmap(
    corr_vif, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
    square=True, linewidths=0.5, cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    annot_kws={"size": 8}, ax=ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "S1b_vif50_corr.svg", bbox_inches="tight")
fig.savefig(OUT / "S1b_vif50_corr.png", dpi=200, bbox_inches="tight")
plt.close()

corr_vif.to_csv(OUT / "S1b_source_corr_vif50.csv")
print(f"  S1b done. shape={corr_vif.shape}", flush=True)

# ── S2 & S3: Optuna TPE convergence ──────────────────────────────────────────
print("S2/S3...", flush=True)
with open(OUTDIR / "optuna_conv_data.json") as f:
    conv = json.load(f)

ml_conv = conv["ml"]
dl_conv = conv["dl"]
tlist = list(list(ml_conv.values())[0].keys())

ML_MODELS = list(ml_conv.keys())
DL_MODELS = list(dl_conv.keys())
ML_COLORS = ["#2196F3","#FF9800","#9C27B0","#4CAF50","#F44336","#00BCD4","#795548"]
DL_COLORS = ["#E91E63","#FF5722","#607D8B"]

def make_panel(conv_dict, models, colors, n_trials, fname_stem):
    nrows, ncols = 3, 4
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
    plt.tight_layout()
    fig.savefig(OUT / f"{fname_stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{fname_stem}.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  {fname_stem} done.", flush=True)

make_panel(ml_conv, ML_MODELS, ML_COLORS, 50, "S3_optuna_ML")
make_panel(dl_conv, DL_MODELS, DL_COLORS, 30, "S2_optuna_DL")

# Source data CSVs for S2/S3
rows_ml, rows_dl = [], []
for mname in ML_MODELS:
    for tkey in tlist:
        for trial, val in enumerate(ml_conv[mname][tkey], 1):
            rows_ml.append({"model": mname, "target": tkey, "trial": trial, "best_val_r2": val})
for mname in DL_MODELS:
    for tkey in tlist:
        for trial, val in enumerate(dl_conv[mname][tkey], 1):
            rows_dl.append({"model": mname, "target": tkey, "trial": trial, "best_val_r2": val})

pd.DataFrame(rows_ml).to_csv(OUT / "S3_source_optuna_ML.csv", index=False)
pd.DataFrame(rows_dl).to_csv(OUT / "S2_source_optuna_DL.csv", index=False)

print("ALL DONE", flush=True)
print(f"Files in {OUT}:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}")
