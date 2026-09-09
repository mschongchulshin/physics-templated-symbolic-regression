"""Unified Nature-style plotting helpers for the PT-SR HEA manuscript.
Import this in every figure script so Results 2/3/4 + all Supplementary
figures share one consistent look.

Conventions (established in the SOTA comparison figure):
  - sans-serif (Helvetica/Arial), small fonts, svg.fonttype='none' (editable text)
  - thin #333 spines, despined top/right, light grid
  - panel letters a,b,c... bold top-left
  - model-type color families: Symbolic=blue, Neural=green, Tree/ensemble=gold
  - PT-SR hero color = dark navy #08306B
"""
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt

# ---- color system ----
PT_HERO = "#08306B"
PAL = {  # 10 SOTA models (figure display names)
    "PT-SR (this work)":                 "#08306B",
    "Ouyang et al. (2018) SISSO":        "#2171B5",
    "Cranmer (2023) PySR":               "#6BAED6",
    "Jain et al. (2026) DNN":            "#006D2C",
    "Liu et al. (2024) MLP":             "#238B45",
    "Korkmaz et al. (2025) Transformer": "#41AB5D",
    "Zhang et al. (2024) LESets GNN":    "#74C476",
    "Liu et al. (2024) RF":              "#B8860B",
    "Wu et al. (2024) gplearn+RFR":      "#DAA520",
    "JMI Stacking (2024)":               "#E8C547",
}
CLASS = {  # model -> family
    "PT-SR (this work)":"Symbolic","Ouyang et al. (2018) SISSO":"Symbolic","Cranmer (2023) PySR":"Symbolic",
    "Jain et al. (2026) DNN":"Neural","Liu et al. (2024) MLP":"Neural",
    "Korkmaz et al. (2025) Transformer":"Neural","Zhang et al. (2024) LESets GNN":"Neural",
    "Liu et al. (2024) RF":"Tree/ensemble","Wu et al. (2024) gplearn+RFR":"Tree/ensemble","JMI Stacking (2024)":"Tree/ensemble",
}
FAMILY = {"Symbolic":"#2171B5","Neural":"#238B45","Tree/ensemble":"#DAA520"}
# sequential single-hue for heatmaps / continuous (navy family)
SEQ = "Blues"
DIVERGE = "RdBu_r"

# full target names (12) used across all figures
TARGET_ORDER = ["Youngs_modulus","UTS","Disloc_0pct","Disloc_20pct","FCC_0pct","FCC_20pct",
                "Other_0pct","Other_20pct","HCP_0pct","HCP_20pct","BCC_0pct","BCC_20pct"]
TARGET_FULL = {
 "Youngs_modulus":"Young's modulus","UTS":"Ultimate tensile strength",
 "Disloc_0pct":"Dislocation density (ε=0%)","Disloc_20pct":"Dislocation density (ε=20%)",
 "FCC_0pct":"FCC phase fraction (ε=0%)","FCC_20pct":"FCC phase fraction (ε=20%)",
 "Other_0pct":"Other phase fraction (ε=0%)","Other_20pct":"Other phase fraction (ε=20%)",
 "HCP_0pct":"HCP phase fraction (ε=0%)","HCP_20pct":"HCP phase fraction (ε=20%)",
 "BCC_0pct":"BCC phase fraction (ε=0%)","BCC_20pct":"BCC phase fraction (ε=20%)"}
TARGET_SHORT = {k:v.replace(" phase fraction"," ").replace("Ultimate tensile strength","UTS")
                .replace("Young's modulus","Young's mod.").replace("Dislocation density","Disloc.")
                for k,v in TARGET_FULL.items()}

def apply():
    """Set global rcParams. Call once at top of each figure script."""
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
        "font.size":7,"axes.titlesize":8.5,"axes.labelsize":7.5,"legend.fontsize":6.6,
        "xtick.labelsize":6.5,"ytick.labelsize":6.5,
        "axes.linewidth":0.7,"axes.edgecolor":"#333333",
        "xtick.color":"#333333","ytick.color":"#333333","text.color":"#222222","axes.labelcolor":"#222222",
        "xtick.major.size":2.5,"ytick.major.size":2.5,"xtick.major.width":0.6,"ytick.major.width":0.6,
        "savefig.dpi":300,"savefig.bbox":"tight","svg.fonttype":"none","figure.dpi":120,
        "axes.grid":False,"grid.color":"#e9e9e9","grid.linewidth":0.5,
    })

def despine(ax, top=True, right=True, left=False, bottom=False):
    for side,off in [("top",top),("right",right),("left",left),("bottom",bottom)]:
        if off: ax.spines[side].set_visible(False)

def panel(ax, letter, dx=-0.02, dy=1.04, fs=12):
    ax.text(dx,dy,letter,transform=ax.transAxes,fontsize=fs,fontweight="bold",va="bottom",ha="right")

def grid(ax, axis="both"):
    ax.grid(axis=axis,ls="-",lw=0.4,color="#ececec",zorder=0); ax.set_axisbelow(True)

def save(fig, path_noext, outdir=None):
    if outdir is None:
        outdir = Path(__file__).resolve().parent.parent / "results" / "generated"
    from pathlib import Path
    p=Path(outdir); p.mkdir(parents=True,exist_ok=True)
    fig.savefig(p/f"{path_noext}.svg"); fig.savefig(p/f"{path_noext}.png",dpi=300)
    return str(p/f"{path_noext}.svg")
