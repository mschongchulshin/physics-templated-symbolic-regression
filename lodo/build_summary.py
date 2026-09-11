"""
Step 5: Aggregate PT-SR + 9 baselines into a unified comparison.

Inputs:
  lodo/pt_sr_results.csv
  lodo/pt_sr_per_template.csv
  lodo/pt_sr_best.csv
  lodo/baselines_results.csv
  lodo/baselines_per_fold.csv

Outputs:
  comparison_summary.md         — headline table + win counts + per-fold winner
  all_methods_per_fold.csv      — full long table (one row per (method, fold))
  all_methods_per_fold_wide.csv — pivot: rows=fold, cols=method
  comparison_heatmap.png        — method x fold heatmap
  lodo_box_violin.png           — R^2 distribution per method
  ptsr_fair_vs_oracle.csv       — per-fold fair vs oracle gap
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = str(Path(__file__).resolve().parent.parent)
OUT = Path(f"{BASE}/results/generated/lodo_v2")


def main():
    pt_best = pd.read_csv(OUT / "pt_sr_best.csv")
    base_per_fold = pd.read_csv(OUT / "baselines_per_fold.csv")

    # ----- Build long table: one row per (method, fold) -----
    long_rows = []
    for _, r in pt_best.iterrows():
        long_rows.append({
            "fold_id": r["fold_id"],
            "composition": r["composition"],
            "processing": r["processing"],
            "property": r["property"],
            "test_source": r["test_source"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "has_overlap": r["has_overlap"],
            "method": "PT-SR_fair",
            "test_r2": r["best_test_r2_fair"],
            "extra": r["best_template_fair"],
        })
        long_rows.append({
            "fold_id": r["fold_id"],
            "composition": r["composition"],
            "processing": r["processing"],
            "property": r["property"],
            "test_source": r["test_source"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "has_overlap": r["has_overlap"],
            "method": "PT-SR_oracle",
            "test_r2": r["best_test_r2_oracle"],
            "extra": r["best_template_oracle"],
        })
    for _, r in base_per_fold.iterrows():
        long_rows.append({
            "fold_id": r["fold_id"],
            "composition": r["composition"],
            "processing": r["processing"],
            "property": r["property"],
            "test_source": r["test_source"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "has_overlap": r["has_overlap"],
            "method": r["baseline"],
            "test_r2": r["test_r2_median"],
            "extra": "",
        })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT / "all_methods_per_fold.csv", index=False)

    # ----- Wide pivot -----
    wide = long_df.pivot_table(
        index=["fold_id", "composition", "processing", "property",
               "test_source", "n_train", "n_test", "has_overlap"],
        columns="method", values="test_r2",
    ).reset_index()
    wide.to_csv(OUT / "all_methods_per_fold_wide.csv", index=False)

    methods_order = ["PT-SR_fair", "PT-SR_oracle"] + sorted(
        [m for m in long_df["method"].unique() if not m.startswith("PT-SR")]
    )
    method_cols = [m for m in methods_order if m in wide.columns]

    # ----- Headline summary table -----
    n_folds = len(wide)
    summary_rows = []
    for m in method_cols:
        s = wide[m].dropna()
        summary_rows.append({
            "method": m,
            "n_folds_eval": int(len(s)),
            "median_R2": float(np.median(s)) if len(s) else np.nan,
            "mean_R2": float(np.mean(s)) if len(s) else np.nan,
            "std_R2": float(np.std(s)) if len(s) else np.nan,
            "frac_R2_pos": float((s > 0).mean()) if len(s) else np.nan,
            "frac_R2_above_0p5": float((s > 0.5).mean()) if len(s) else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("median_R2", ascending=False)
    summary_df.to_csv(OUT / "headline_summary.csv", index=False)

    # ----- Win counts: PT-SR_fair vs each baseline -----
    win_rows = []
    fair = wide["PT-SR_fair"]
    for m in method_cols:
        if m == "PT-SR_fair":
            continue
        diff = fair - wide[m]
        valid = diff.dropna()
        n_v = len(valid)
        n_win = int((valid > 0).sum())
        n_tie = int((valid == 0).sum())
        n_lose = int((valid < 0).sum())
        win_rows.append({
            "vs": m,
            "n_compared": n_v,
            "PT-SR_fair_wins": n_win,
            "ties": n_tie,
            "PT-SR_fair_loses": n_lose,
            "PT-SR_fair_win_rate": n_win / max(1, n_v),
            "median_diff_R2": float(valid.median()) if n_v else np.nan,
        })
    wins_df = pd.DataFrame(win_rows).sort_values("PT-SR_fair_win_rate", ascending=False)
    wins_df.to_csv(OUT / "ptsr_fair_win_counts.csv", index=False)

    # ----- Per-fold winner -----
    base_only_methods = [m for m in method_cols if m not in ("PT-SR_fair", "PT-SR_oracle")]
    winners = []
    for _, r in wide.iterrows():
        candidates = {m: r[m] for m in method_cols if pd.notna(r.get(m))}
        if not candidates:
            winners.append({**{c: r[c] for c in ["fold_id", "composition", "processing",
                                                  "property", "test_source"]},
                             "winner": "—", "winner_R2": np.nan})
            continue
        winner = max(candidates, key=candidates.get)
        winners.append({
            "fold_id": r["fold_id"],
            "composition": r["composition"],
            "processing": r["processing"],
            "property": r["property"],
            "test_source": r["test_source"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "has_overlap": r["has_overlap"],
            "winner": winner,
            "winner_R2": candidates[winner],
            "PT-SR_fair_R2": r.get("PT-SR_fair", np.nan),
            "best_baseline_R2": max(
                [r[m] for m in base_only_methods if pd.notna(r.get(m))], default=np.nan
            ),
        })
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(OUT / "per_fold_winners.csv", index=False)

    # ----- PT-SR fair vs oracle gap -----
    if "PT-SR_oracle" in wide.columns:
        gap = wide[["fold_id", "PT-SR_fair", "PT-SR_oracle"]].copy()
        gap["fair_oracle_gap"] = gap["PT-SR_oracle"] - gap["PT-SR_fair"]
        gap.to_csv(OUT / "ptsr_fair_vs_oracle.csv", index=False)

    # ----- Comparison heatmap -----
    fig, ax = plt.subplots(figsize=(max(6, len(method_cols)*0.7), max(8, n_folds*0.18)))
    mat = wide[method_cols].values.astype(float)
    # Sort folds by PT-SR_fair test R^2 descending for readability
    fair_col_idx = method_cols.index("PT-SR_fair")
    sort_idx = np.argsort(-np.nan_to_num(mat[:, fair_col_idx], nan=-99))
    mat_sorted = mat[sort_idx]
    fold_labels = (wide["composition"].astype(str) + "|" + wide["processing"].astype(str)
                    + "|" + wide["property"].astype(str) + "|" + wide["test_source"].astype(str)).values[sort_idx]
    im = ax.imshow(mat_sorted, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(method_cols)))
    ax.set_xticklabels(method_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(fold_labels)))
    ax.set_yticklabels(fold_labels, fontsize=6)
    ax.set_title("LODO v2 test R² (rows: folds sorted by PT-SR_fair)")
    plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    plt.savefig(OUT / "comparison_heatmap.png", dpi=150)
    plt.close()

    # ----- Box+violin -----
    fig, ax = plt.subplots(figsize=(max(8, len(method_cols)*0.9), 6))
    data = [wide[m].dropna().clip(-2, 1).values for m in method_cols]
    parts = ax.violinplot(data, showmedians=True, showmeans=False)
    ax.boxplot(data, widths=0.18, showfliers=False, patch_artist=False,
                medianprops=dict(color="black"))
    ax.set_xticks(range(1, len(method_cols)+1))
    ax.set_xticklabels(method_cols, rotation=30, ha="right")
    ax.set_ylabel("Test R² (clipped to [-2, 1])")
    ax.set_title(f"LODO v2 test R² distribution ({n_folds} folds)")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    plt.tight_layout()
    plt.savefig(OUT / "lodo_box_violin.png", dpi=150)
    plt.close()

    # ----- Markdown summary -----
    lines = [
        "# LODO v2: Honest Cross-Dataset Comparison",
        "",
        f"- **Source data**: `{BASE}/MPEA_figshare_dataset.csv` (1545 rows)",
        f"- **Fold definitions**: `{OUT}/folds.csv` (verified n_train/n_test from real MPEA rows)",
        f"- **Valid LODO folds**: {n_folds} (multi-source, n_train≥3, n_test≥2)",
        "- **Note**: PT-SR_fair selects template by **train R²** (no test peek). "
        "PT-SR_oracle peeks at test (reference only — do NOT use as the headline result).",
        "- **Seeds**: 5 per stochastic method; deterministic methods use 1 seed.",
        "",
        "## Headline summary (sorted by median R²)",
        "",
        "| Method | n_folds | median R² | mean R² | std R² | frac R²>0 | frac R²>0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['method']} | {r['n_folds_eval']} | "
            f"{r['median_R2']:.4f} | {r['mean_R2']:.4f} | {r['std_R2']:.4f} | "
            f"{r['frac_R2_pos']:.2f} | {r['frac_R2_above_0p5']:.2f} |"
        )

    lines += [
        "", "## PT-SR_fair vs each baseline (per-fold head-to-head)",
        "",
        "| vs Baseline | n | PT-SR_fair wins | ties | PT-SR_fair loses | win rate | median ΔR² |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in wins_df.iterrows():
        lines.append(
            f"| {r['vs']} | {r['n_compared']} | {r['PT-SR_fair_wins']} | {r['ties']} | "
            f"{r['PT-SR_fair_loses']} | {r['PT-SR_fair_win_rate']:.2f} | "
            f"{r['median_diff_R2']:+.4f} |"
        )

    # Fair vs Oracle
    if "PT-SR_oracle" in wide.columns:
        fair = wide["PT-SR_fair"].dropna()
        orc = wide["PT-SR_oracle"].dropna()
        gap = (wide["PT-SR_oracle"] - wide["PT-SR_fair"]).dropna()
        lines += [
            "", "## PT-SR_fair vs PT-SR_oracle gap",
            "",
            f"- Median R² (fair):   {fair.median():.4f}",
            f"- Median R² (oracle): {orc.median():.4f}",
            f"- Median gap (oracle - fair): {gap.median():+.4f}",
            f"- Mean gap: {gap.mean():+.4f}",
            f"- N folds where fair == oracle template: "
            f"{(pt_best['best_template_fair'] == pt_best['best_template_oracle']).sum()} / {len(pt_best)}",
        ]

    # Per-fold winner counts
    lines += [
        "", "## Per-fold winners (best test R²)", "",
        "| Method | Wins (out of " + str(len(winners_df)) + ") |",
        "|---|---:|",
    ]
    win_counts = winners_df["winner"].value_counts()
    for m in win_counts.index:
        lines.append(f"| {m} | {int(win_counts[m])} |")

    # Subset (no overlap) — clean LODO
    if wide["has_overlap"].any():
        clean = wide[~wide["has_overlap"]]
        lines += [
            "", f"## Subset: clean folds (no train/test overlap on FORMULA × T) — n={len(clean)}",
            "",
            "| Method | n | median R² | mean R² |",
            "|---|---:|---:|---:|",
        ]
        for m in method_cols:
            s = clean[m].dropna()
            if len(s):
                lines.append(f"| {m} | {len(s)} | {s.median():.4f} | {s.mean():.4f} |")

    # Per-fold table
    lines += [
        "", "## Per-fold test R² (all methods)", "",
        "| Fold | n_tr | n_te | overlap | " + " | ".join(method_cols) + " |",
        "|" + "|".join(["---"] * (4 + len(method_cols))) + "|",
    ]
    for _, r in wide.iterrows():
        cells = [
            r["fold_id"], str(int(r["n_train"])), str(int(r["n_test"])),
            "yes" if r["has_overlap"] else "no",
        ]
        for m in method_cols:
            v = r.get(m, np.nan)
            cells.append(f"{v:.3f}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")

    (OUT / "comparison_summary.md").write_text("\n".join(lines))

    print("=== Done ===")
    print(f"Folds: {n_folds}")
    print(f"Summary -> {OUT/'comparison_summary.md'}")
    print(f"Long table -> {OUT/'all_methods_per_fold.csv'}")
    print(f"Wide table -> {OUT/'all_methods_per_fold_wide.csv'}")
    print(f"Heatmap -> {OUT/'comparison_heatmap.png'}")
    print(f"Box/violin -> {OUT/'lodo_box_violin.png'}")
    print()
    print("Headline:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
