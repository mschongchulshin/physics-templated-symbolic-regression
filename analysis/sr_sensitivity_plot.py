"""
SR sensitivity final plots.
Run after all workers complete.
Outputs SVG (editable text), PNG and source CSVs to results/generated/
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "results" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)
from pathlib import Path

OUT = OUTDIR / "sr_sens_out"
OUT.mkdir(exist_ok=True)

# ── Merge all worker CSVs ─────────────────────────────────────────────────────
dfs = []
for i in range(5):
    p = OUTDIR / f"sr_sens_w{i}.csv"
    if p.exists():
        dfs.append(pd.read_csv(p))
df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates(subset=["experiment","model","target","parsimony","niterations"])
print(f"Total rows: {len(df)}")
df.to_csv(OUT / "sr_sensitivity_all.csv", index=False)

TARGETS = ["Youngs_modulus","UTS","Disloc_0pct","Disloc_20pct",
           "FCC_0pct","HCP_0pct","BCC_0pct","Other_0pct",
           "FCC_20pct","HCP_20pct","BCC_20pct","Other_20pct"]
TARGET_LABELS = ["Young's modulus","UTS","Disloc 0%","Disloc 20%",
                 "FCC 0%","HCP 0%","BCC 0%","Other 0%",
                 "FCC 20%","HCP 20%","BCC 20%","Other 20%"]

TWO_STAGE = ["additive","multiplicative","thermal_softening","arrhenius",
             "power_law","entropy_weighted","calphad","free_2stage"]
SINGLE_STAGE = ["free_1stage","freeform","6feat_control"]
ALL_TEMPLATES = TWO_STAGE + SINGLE_STAGE

# Color palette
import matplotlib.cm as cm
cmap = cm.get_cmap("tab10")
COLORS = {t: cmap(i % 10) for i, t in enumerate(ALL_TEMPLATES)}

# ── Fig 1: Parsimony sensitivity ──────────────────────────────────────────────
pars_df = df[df["experiment"] == "parsimony"].copy()

fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey=False)
axes = axes.flatten()
for idx, (tkey, tlabel) in enumerate(zip(TARGETS, TARGET_LABELS)):
    ax = axes[idx]
    sub = pars_df[pars_df["target"] == tkey]
    for tmpl in ALL_TEMPLATES:
        row = sub[sub["model"] == tmpl].sort_values("parsimony")
        if len(row) == 0:
            continue
        ax.plot(row["parsimony"], row["train_r2_best"],
                color=COLORS[tmpl], lw=1.6, marker="o", ms=4, label=tmpl)
    ax.set_xscale("log")
    ax.set_xlabel("Parsimony λ", fontsize=9)
    ax.set_ylabel("Train R²", fontsize=9)
    ax.set_title(tlabel, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=6, loc="lower left", ncol=2)

plt.tight_layout()
fig.savefig(OUT / "S_parsimony.svg", bbox_inches="tight")
fig.savefig(OUT / "S_parsimony.png", dpi=180, bbox_inches="tight")
plt.close()
print("S_parsimony done.")

# Source CSV
pars_df.to_csv(OUT / "S_parsimony_source.csv", index=False)

# ── Fig 2: niterations convergence ────────────────────────────────────────────
niter_df = df[df["experiment"] == "niterations"].copy()

fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey=False)
axes = axes.flatten()
for idx, (tkey, tlabel) in enumerate(zip(TARGETS, TARGET_LABELS)):
    ax = axes[idx]
    sub = niter_df[niter_df["target"] == tkey]
    for tmpl in ALL_TEMPLATES:
        row = sub[sub["model"] == tmpl].sort_values("niterations")
        if len(row) == 0:
            continue
        ax.plot(row["niterations"], row["train_r2_best"],
                color=COLORS[tmpl], lw=1.6, marker="o", ms=4, label=tmpl)
    ax.set_xlabel("niterations", fontsize=9)
    ax.set_ylabel("Train R²", fontsize=9)
    ax.set_title(tlabel, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=6, loc="lower right", ncol=2)

plt.tight_layout()
fig.savefig(OUT / "S_niterations.svg", bbox_inches="tight")
fig.savefig(OUT / "S_niterations.png", dpi=180, bbox_inches="tight")
plt.close()
print("S_niterations done.")

niter_df.to_csv(OUT / "S_niterations_source.csv", index=False)

print(f"\nAll files saved to {OUT}:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}")
