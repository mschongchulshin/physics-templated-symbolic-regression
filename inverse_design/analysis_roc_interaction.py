#!/usr/bin/env python3

from pathlib import Path
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

ROOT      = Path(__file__).resolve().parent.parent
RAW_CSV   = str(ROOT / "data" / "CoCrCuFeNi_684.csv")
FEAT_CSV  = str(ROOT / "data" / "features_13.csv")
EQ_XLSX   = str(ROOT / "equations" / "free_2stage_best_equations.xlsx")
FIG_DIR   = str(ROOT / "results" / "generated" / "figures")
DATA_DIR  = str(ROOT / "results" / "generated" / "inverse")
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

ROC_PNG   = os.path.join(FIG_DIR,  "roc_inverse_design.png")
INTER_PNG = os.path.join(FIG_DIR,  "feature_interaction_contour.png")
ROC_JSON  = os.path.join(DATA_DIR, "roc_inverse_design.json")
GRID_JSON = os.path.join(DATA_DIR, "feature_interaction_grid.json")

ELEMENTS = ["Co", "Cr", "Cu", "Fe", "Ni"]
PROPS = {
    "atomic_volume":        np.array([6.6516, 7.2722, 7.0922, 7.0959, 6.5948]),
    "electron_affinity":    np.array([0.6623, 0.6660, 1.2350, 0.1510, 1.1560]),
    "fusion_heat":          np.array([15.48,  21.00,  13.01,  13.80,  17.61]),
    "metallic_radius":      np.array([116.0, 119.0, 118.0, 117.0, 115.0]),
    "molar_heat_capacity":  np.array([24.81, 23.35, 24.44, 25.10, 26.07]),
    "vdw_radius_batsanov":  np.array([200.0, 205.0, 200.0, 205.0, 200.0]),
}
VEC = np.array([9.0, 6.0, 11.0, 8.0, 10.0])

DH_BINARY = np.array([
    [  0.0, -4.0,  6.0, -1.0,   0.0],
    [ -4.0,  0.0, 12.0, -1.0,  -7.0],
    [  6.0, 12.0,  0.0, 13.0,   4.0],
    [ -1.0, -1.0, 13.0,  0.0,  -2.0],
    [  0.0, -7.0,  4.0, -2.0,   0.0],
])
R_GAS = 8.314


def _mole_fractions(comp_pct):
    return np.asarray(comp_pct, dtype=float) / 100.0


def compute_thermo_descriptors(comp_pct):
    x = _mole_fractions(comp_pct)
    vec = float(np.sum(x * VEC))

    r = PROPS["metallic_radius"]
    r_mean = float(np.sum(x * r))
    delta_r = float(100.0 * np.sqrt(np.sum(x * (1 - r / r_mean) ** 2)))

    chi = np.array([1.88, 1.66, 1.90, 1.83, 1.91])
    chi_mean = float(np.sum(x * chi))
    delta_chi = float(np.sqrt(np.sum(x * (chi - chi_mean) ** 2)))

    xs = np.clip(x, 1e-12, None)
    S_mix = float(-R_GAS * np.sum(xs * np.log(xs)))

    DH = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            DH += 4.0 * DH_BINARY[i, j] * x[i] * x[j]
    return {
        "VEC": vec,
        "S_mix": S_mix,
        "delta_r": delta_r,
        "delta_chi": delta_chi,
        "DH_mix": float(DH),
    }


def compute_features_13(comp_pct):
    x = _mole_fractions(comp_pct)
    out = {}
    for pname, vals in PROPS.items():
        avg = float(x @ vals)
        delta = float(np.sqrt(max(x @ (vals ** 2) - avg ** 2, 0.0)))
        out[f"{pname}_avg"] = avg
        out[f"{pname}_delta"] = delta
    return out


def build_evaluators():
    eq_df = pd.read_excel(EQ_XLSX)
    var_names = [
        "Co", "Cr", "Cu", "Fe", "Ni", "T", "g",
        "atomic_volume_delta", "electron_affinity_delta", "fusion_heat_delta",
        "metallic_radius_avg", "molar_heat_capacity_delta",
        "vdw_radius_batsanov_avg", "vdw_radius_batsanov_delta",
    ]
    sym_vars = sp.symbols(var_names)
    parse_ns = dict(zip(var_names, sym_vars))
    parse_ns.update({
        "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp,
        "cos": sp.cos, "sin": sp.sin, "Abs": sp.Abs,
        "square": lambda x: x ** 2, "cube": lambda x: x ** 3,
    })

    evaluators = {}
    for _, row in eq_df.iterrows():
        target = row["Target"]
        g_expr = sp.sympify(row["Stage 1 g(x)"], locals=parse_ns)
        f_expr = sp.sympify(row["Stage 2 f(g,T)"], locals=parse_ns)
        g_func = sp.lambdify(sym_vars, g_expr, "numpy")
        f_func = sp.lambdify(sym_vars, f_expr, "numpy")
        evaluators[target] = {
            "g_func": g_func,
            "f_func": f_func,
            "var_names": var_names,
            "g_str": row["Stage 1 g(x)"],
            "f_str": row["Stage 2 f(g,T)"],
            "train_r2": float(row["Train R²"]),
        }
    return evaluators


def predict_property(evaluators, target, comps_pct, T_kelvin):
    comps = np.atleast_2d(comps_pct).astype(float)
    n = comps.shape[0]

    arrs = {nm: np.zeros(n) for nm in evaluators[target]["var_names"]}
    for i, el in enumerate(ELEMENTS):
        arrs[el] = comps[:, i]
    arrs["T"] = np.full(n, float(T_kelvin))

    for k in range(n):
        f13 = compute_features_13(comps[k])
        for key, val in f13.items():
            if key in arrs:
                arrs[key][k] = val

    var_names = evaluators[target]["var_names"]
    args = [arrs[v] if v != "g" else np.zeros(n) for v in var_names]
    g_vals = evaluators[target]["g_func"](*args)
    arrs["g"] = np.asarray(g_vals)
    args = [arrs[v] for v in var_names]
    pred = evaluators[target]["f_func"](*args)
    return np.asarray(pred, dtype=float)


def task1_roc(evaluators):
    df_raw = pd.read_csv(RAW_CSV)

    df = df_raw[df_raw["T_K"] == 300].reset_index(drop=True)
    comps = df[["Co(%)", "Cr(%)", "Cu(%)", "Fe(%)", "Ni(%)"]].values

    targets_cfg = {
        "UTS":            ("UTS(Gpa)",        "max"),
        "Youngs_modulus": ("Young's modulus(Gpa)", "max"),
        "FCC_0pct":       ("FCC 0%(%)",       "max"),
    }

    rng = np.random.default_rng(42)
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, (tgt, (col, direction)) in zip(axes, targets_cfg.items()):
        true_vals = df[col].values.astype(float)
        thr = np.percentile(true_vals, 90)
        y_true = (true_vals >= thr).astype(int)

        pred = predict_property(evaluators, tgt, comps, 300.0)
        score_ptsr = pred if direction == "max" else -pred

        fpr_p, tpr_p, _ = roc_curve(y_true, score_ptsr)
        auc_p = auc(fpr_p, tpr_p)

        n_rand = 1000
        aucs_rand = np.zeros(n_rand)
        for k in range(n_rand):
            score_r = rng.uniform(size=len(y_true))
            fpr_r, tpr_r, _ = roc_curve(y_true, score_r)
            aucs_rand[k] = auc(fpr_r, tpr_r)
        score_r = rng.uniform(size=len(y_true))
        fpr_r, tpr_r, _ = roc_curve(y_true, score_r)
        auc_r_single = auc(fpr_r, tpr_r)

        order = np.argsort(-score_ptsr)
        n10 = max(1, int(np.ceil(0.10 * len(y_true))))
        precision_top10 = float(y_true[order[:n10]].sum() / n10)
        baseline_rate = float(y_true.sum() / len(y_true))
        ef10 = precision_top10 / baseline_rate

        ax.plot(fpr_p, tpr_p, color="#c0392b", lw=2,
                label=f"PT-SR  AUC={auc_p:.3f}")
        ax.plot(fpr_r, tpr_r, color="#7f8c8d", lw=1.4, ls="--",
                label=f"Random AUC={aucs_rand.mean():.3f}±{aucs_rand.std():.3f}")
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, alpha=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        if ax is axes[0]:
            ax.set_ylabel("True positive rate")
        ax.set_title(
            f"{tgt}  (top 10% = {y_true.sum()}/{len(y_true)})\n"
            f"Top-10% precision: PT-SR {precision_top10:.2f} vs base {baseline_rate:.2f}\n"
            f"Enrichment x{ef10:.1f}"
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)

        summary[tgt] = {
            "n_total": int(len(y_true)),
            "n_positive": int(y_true.sum()),
            "threshold_top10": float(thr),
            "auc_ptsr": float(auc_p),
            "auc_random_mean": float(aucs_rand.mean()),
            "auc_random_std":  float(aucs_rand.std()),
            "precision_top10_ptsr": precision_top10,
            "baseline_positive_rate": baseline_rate,
            "enrichment_factor_top10": float(ef10),
            "ratio_ptsr_over_random": float(auc_p / max(aucs_rand.mean(), 1e-6)),
        }

    fig.suptitle("PT-SR-guided inverse design vs random screening "
                 "(232 candidates, 300 K, top 10% UTS / YM / FCC as positives)",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    fig.savefig(ROC_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(ROC_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def task2_interaction(evaluators):
    df_raw = pd.read_csv(RAW_CSV)
    df = df_raw[df_raw["T_K"] == 300].reset_index(drop=True)
    comps = df[["Co(%)", "Cr(%)", "Cu(%)", "Fe(%)", "Ni(%)"]].values

    descs = np.array([list(compute_thermo_descriptors(c).values()) for c in comps])
    desc_keys = list(compute_thermo_descriptors(comps[0]).keys())
    DH_idx = desc_keys.index("DH_mix")
    VEC_idx = desc_keys.index("VEC")
    DH_train, VEC_train = descs[:, DH_idx], descs[:, VEC_idx]

    rng = np.random.default_rng(0)
    N_TARGET = 12000
    samples = []
    while len(samples) < N_TARGET:
        candidate = rng.dirichlet(alpha=2.0 * np.ones(5),
                                  size=N_TARGET) * 100.0
        ok = np.all((candidate >= 5.0) & (candidate <= 50.0), axis=1)
        samples.append(candidate[ok])
        if sum(s.shape[0] for s in samples) >= N_TARGET:
            break
    samples = np.vstack(samples)[:N_TARGET]
    N_SAMPLE = samples.shape[0]

    uts_pred = predict_property(evaluators, "UTS",      samples, 300.0)
    fcc_pred = predict_property(evaluators, "FCC_0pct", samples, 300.0)
    fcc_pred = np.clip(fcc_pred, 0.0, 100.0)

    DH_s = np.zeros(N_SAMPLE)
    VEC_s = np.zeros(N_SAMPLE)
    for k in range(N_SAMPLE):
        d = compute_thermo_descriptors(samples[k])
        DH_s[k] = d["DH_mix"]
        VEC_s[k] = d["VEC"]

    nbins = 40
    DH_edges  = np.linspace(DH_s.min(),  DH_s.max(),  nbins + 1)
    VEC_edges = np.linspace(VEC_s.min(), VEC_s.max(), nbins + 1)
    DH_cent   = 0.5 * (DH_edges[1:]  + DH_edges[:-1])
    VEC_cent  = 0.5 * (VEC_edges[1:] + VEC_edges[:-1])

    def _grid_mean(values):
        H, _, _ = np.histogram2d(VEC_s, DH_s, bins=[VEC_edges, DH_edges],
                                 weights=values)
        C, _, _ = np.histogram2d(VEC_s, DH_s, bins=[VEC_edges, DH_edges])
        Z = np.where(C > 0, H / np.maximum(C, 1), np.nan)
        Zs = np.full_like(Z, np.nan)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ii = slice(max(i - 1, 0), min(i + 2, Z.shape[0]))
                jj = slice(max(j - 1, 0), min(j + 2, Z.shape[1]))
                window = Z[ii, jj]
                vals = window[~np.isnan(window)]
                if vals.size:
                    Zs[i, j] = vals.mean()
        return Zs

    Z_uts = _grid_mean(uts_pred)
    Z_fcc = _grid_mean(fcc_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, Z, name, cmap in [
        (axes[0], Z_uts, "PT-SR predicted UTS [GPa]", "viridis"),
        (axes[1], Z_fcc, "PT-SR predicted FCC fraction (0%) [%]", "magma"),
    ]:
        cf = ax.contourf(DH_cent, VEC_cent, Z, levels=20, cmap=cmap)
        cs = ax.contour(DH_cent, VEC_cent, Z, levels=10,
                        colors="white", linewidths=0.4, alpha=0.6)
        cb = fig.colorbar(cf, ax=ax, shrink=0.9)
        cb.set_label(name)

        ax.scatter(DH_train, VEC_train, c="white", edgecolor="black",
                   s=22, lw=0.6, alpha=0.9, label="232 training comps")

        cantor = compute_thermo_descriptors([20, 20, 20, 20, 20])
        ax.scatter([cantor["DH_mix"]], [cantor["VEC"]],
                   marker="*", s=320, c="red", edgecolor="white",
                   linewidth=1.2, label="Cantor (20-20-20-20-20)", zorder=5)

        true_vals = df["UTS(Gpa)"].values
        thr_uts = np.percentile(true_vals, 90)
        mask = true_vals >= thr_uts
        ax.scatter(DH_train[mask], VEC_train[mask],
                   marker="^", s=70, c="lime", edgecolor="black",
                   lw=0.6, alpha=0.95, label="Train top-10% UTS")

        ax.set_xlabel(r"$\Delta H_{\rm mix}$ [kJ/mol]  (Miedema, 5-element)")
        ax.set_ylabel("VEC")
        ax.set_title(name.split(" [")[0] + " landscape")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Feature-interaction landscape: PT-SR predictions projected onto "
        f"(ΔH_mix, VEC) plane (T = 300 K, {N_SAMPLE} simplex samples, "
        "each element 5-50%)",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(INTER_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_samples": int(N_SAMPLE),
        "DH_range": [float(DH_s.min()), float(DH_s.max())],
        "VEC_range": [float(VEC_s.min()), float(VEC_s.max())],
        "training_max_UTS": {
            "DH_mix":  float(DH_train[np.argmax(df['UTS(Gpa)'].values)]),
            "VEC":     float(VEC_train[np.argmax(df['UTS(Gpa)'].values)]),
            "UTS_GPa": float(df['UTS(Gpa)'].values.max()),
        },
        "training_max_FCC": {
            "DH_mix":  float(DH_train[np.argmax(df['FCC 0%(%)'].values)]),
            "VEC":     float(VEC_train[np.argmax(df['FCC 0%(%)'].values)]),
            "FCC_0pct_pct": float(df['FCC 0%(%)'].values.max()),
        },
    }
    with open(GRID_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main():
    evaluators = build_evaluators()
    print("Built evaluators for:", list(evaluators.keys()))
    roc_summary = task1_roc(evaluators)
    print("\n=== ROC summary ===")
    print(json.dumps(roc_summary, indent=2))
    grid_summary = task2_interaction(evaluators)
    print("\n=== Interaction summary ===")
    print(json.dumps(grid_summary, indent=2))
    print("\nFigures written:")
    print(" -", ROC_PNG)
    print(" -", INTER_PNG)


if __name__ == "__main__":
    main()
