"""Two one-sided tests for equivalence between PT-SR and the best baseline.

The paper reports that PT-SR is statistically equivalent to the best-performing
published model on seven of the twelve targets, at a margin of +/-0.02 R2. That
claim rests on a two-one-sided-tests procedure with a Benjamini-Hochberg
correction across the twelve targets, and the result was tabulated in the
source data without the code that produced it.

The procedure, per target:

  The best baseline is the one with the highest mean test R2 over the five
  seeds. Each method contributes one mean R2 per seed, so both samples have
  n = 5 and the pooled two-sample t has 2n - 2 = 8 degrees of freedom.

  Equivalence asks whether the difference lies inside (-margin, +margin). TOST
  runs two one-sided t tests against those bounds and takes the larger p, which
  is the p for the composite null that the difference lies outside the interval.

  Twelve targets means twelve tests, so the p values are corrected by
  Benjamini-Hochberg at alpha = 0.05.

This picks the same comparator as the paper on all twelve targets and the same
verdict on eleven of them, from the deposited cross-validation runs. It does not reproduce the count of
seven reported in the paper: the PT-SR R2 in the source data's Supp_Note_3_TOST
sheet is higher than the same quantity in Fig_2a and in
results/cv_results/sr_9templates_raw.csv, by 0.001 to 0.010 depending on the
target, and no aggregation of the deposited runs reproduces it. Recomputed from
the deposited numbers the count is six.

Run from the repository root. Writes results/generated/equivalence_tost.csv.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "results" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)

MARGIN = 0.02
ALPHA = 0.05
N_SEEDS = 5   # five seeds enter each arm of the test

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
    """The larger of the two one-sided p values, from summary statistics.

    Equivalence is rejected unless both one-sided tests reject, so the
    composite p is the larger of the two.
    """
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df
    se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    d = m1 - m2
    # lower bound: is d > -margin?   upper bound: is d < +margin?
    p_lower = stats.t.sf((d + margin) / se, df)
    p_upper = stats.t.cdf((d - margin) / se, df)
    return max(p_lower, p_upper), df, d


def benjamini_hochberg(p, alpha):
    """Step-up FDR control. Returns adjusted p, rejection flags and ranks."""
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
