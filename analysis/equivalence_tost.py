import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "results" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)

MARGIN = 0.02
ALPHA = 0.05
N_SEEDS = 5

TARGET_NAME = {
    "Youngs_modulus": "Young's modulus", "UTS": "Ultimate tensile strength",
    "Disloc_0pct": "Dislocation density (0%)",
    "Disloc_20pct": "Dislocation density (20%)",
    "FCC_0pct": "FCC fraction (0%)", "FCC_20pct": "FCC fraction (20%)",
    "HCP_0pct": "HCP fraction (0%)", "HCP_20pct": "HCP fraction (20%)",
    "BCC_0pct": "BCC fraction (0%)", "BCC_20pct": "BCC fraction (20%)",
    "Other_0pct": "Unclassified fraction (0%)",
    "Other_20pct": "Unclassified fraction (20%)",
}


def tost_from_summary(m1, s1, n1, m2, s2, n2, margin):
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df
    se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    d = m1 - m2
    p_lower = stats.t.sf((d + margin) / se, df)
    p_upper = stats.t.cdf((d - margin) / se, df)
    return max(p_lower, p_upper), df, d


def benjamini_hochberg(p, alpha):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty(m, int)
    ranks[order] = np.arange(1, m + 1)

    adj = np.empty(m)
    running = 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        running = min(running, p[i] * m / r)
        adj[i] = running
    return adj, adj <= alpha, ranks


def main():
    sd = BASE / "data" / "Source_Data.xlsx"
    mean = pd.read_excel(sd, sheet_name="Fig_2a").set_index("method")
    std = pd.read_excel(sd, sheet_name="Fig_2a_sd").set_index("method")
    pt5 = pd.read_csv(BASE / "results" / "cv_results" /
                      "ptsr_best5_seeds.csv").set_index("target")
    PT = "PT-SR (this work)"
    targets = [c for c in mean.columns if c in TARGET_NAME]

    rows = []
    for target in targets:
        rivals = mean[target].drop(PT)
        best = rivals.idxmax()
        p, df, d = tost_from_summary(
            pt5.loc[target, "best5_mean"], pt5.loc[target, "best5_sd"], N_SEEDS,
            mean.loc[best, target], std.loc[best, target], N_SEEDS, MARGIN)
        rows.append({
            "Target": TARGET_NAME[target], "best_baseline": best,
            "R2_PTSR": pt5.loc[target, "best5_mean"],
            "R2_best_baseline": mean.loc[best, target],
            "delta_R2_signed": d, "margin": MARGIN,
            "alpha_per_test": ALPHA, "n": N_SEEDS, "df": df,
            "TOST_p_raw": p,
        })

    out = pd.DataFrame(rows)
    adj, rej, ranks = benjamini_hochberg(out.TOST_p_raw.values, ALPHA)
    out["equivalent_raw"] = out.TOST_p_raw <= ALPHA
    out["TOST_p_BH_FDR"] = adj
    out["equivalent_BH"] = rej
    out["BH_rank"] = ranks
    out["BH_critical_value"] = ranks / len(out) * ALPHA

    path = OUTDIR / "equivalence_tost.csv"
    out.to_csv(path, index=False)
    print(out[["Target", "best_baseline", "delta_R2_signed",
               "TOST_p_BH_FDR", "equivalent_BH"]].to_string(index=False))
    print(f"\nequivalent on {int(out.equivalent_BH.sum())} of {len(out)} targets"
          f"  (margin +/-{MARGIN}, Benjamini-Hochberg at alpha = {ALPHA})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
