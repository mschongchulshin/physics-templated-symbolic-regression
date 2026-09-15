import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import natstyle as ns
ns.apply()
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

M = pd.read_csv(BASE / "results" / "generated" / "shap_uts_300K_matrix.csv")
shap_cols = [c for c in M.columns if c.startswith("SHAP_")]
feats = [c[len("SHAP_"):] for c in shap_cols]
order = sorted(feats, key=lambda f: np.abs(M[f"SHAP_{f}"]).mean(), reverse=True)

PRETTY = {"Ni":"Ni","Co":"Co","Fe":"Fe","Cu":"Cu","Cr":"Cr",
 "molar_heat_capacity_delta":"molar heat capacity Δ","vdw_radius_batsanov_avg":"vdW radius (avg)",
 "electron_affinity_delta":"electron affinity Δ","metallic_radius_avg":"metallic radius (avg)",
 "fusion_heat_delta":"fusion heat Δ (ΔH_fus)","atomic_volume_delta":"atomic volume Δ",
 "vdw_radius_batsanov_delta":"vdW radius Δ"}

fig = plt.figure(figsize=(7.6, 5.4))
ax = fig.add_subplot(111)
rng = np.random.default_rng(3)
n = len(order)
for i, f in enumerate(order):
    yc = n - 1 - i
    sv = M[f"SHAP_{f}"].values
    xv = M[f"x_{f}"].values.astype(float)
    rngv = xv.max() - xv.min()
    cn = (xv - xv.min())/rngv if rngv > 0 else np.full_like(xv, 0.5)
    jit = rng.uniform(-0.18, 0.18, size=len(sv))
    sc = ax.scatter(sv, yc + jit, c=cn, cmap="rainbow", s=10, alpha=0.7,
                    edgecolor="none", zorder=3, vmin=0, vmax=1)
ax.axvline(0, color="#888", lw=0.8, zorder=2)
ax.set_yticks(range(n))
ax.set_yticklabels([PRETTY.get(f, f) for f in order[::-1]], fontsize=7.0)
ax.set_xlabel("SHAP value  (impact on UTS prediction; + raises, − lowers)", fontsize=8)
ax.set_title("Per-sample SHAP for UTS — effect sign flips with composition",
             fontsize=9, pad=8)
ax.set_ylim(-0.6, n-0.4)
ns.despine(ax); ns.grid(ax, axis="x")
ns.panel(ax, "b")
cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("feature value (low → high)", fontsize=7); cb.set_ticks([0,1]); cb.set_ticklabels(["low","high"])
cb.ax.tick_params(labelsize=6)
fig.text(0.5, 0.012,
  "Signed mean SHAP ≈ 0 for every feature (effects cancel across compositions); only the spread is informative — "
  "a scalar/mean ranking cannot show this, whereas PT-SR's g(x)=log(Fe+2.382·Ni+ΔH_fus³) states the functional role explicitly.",
  ha="center", fontsize=5.8, color="#666")
fig.subplots_adjust(left=0.26, right=0.93, top=0.91, bottom=0.13)
ns.save(fig, "fig2b_shap_beeswarm", outdir=OUT)
print("SAVED fig2b_shap_beeswarm | features:", n, "| samples:", len(M))
print("signed-mean range:", round(M[[f'SHAP_{f}' for f in order]].mean().min(),4),
      "to", round(M[[f'SHAP_{f}' for f in order]].mean().max(),4))
