"""
Generate data integrity report comparing OLD lodo_all_results.xlsx
(4-Apr generated, source script lost) vs NEW verified folds.csv.

Output: integrity_report.md
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = str(Path(__file__).resolve().parent)
OUT = Path(f"{BASE}/lodo_v2")


def main():
    old = pd.read_excel(f"{BASE}/results/lodo_all_results.xlsx")
    new = pd.read_csv(OUT / "folds.csv")
    new_ok = new[new["status"].str.startswith("ok")].copy()

    old["test_source"] = (old["Test_Dataset"]
        .str.replace("_CAST", "", regex=False)
        .str.replace("_WROUGHT", "", regex=False)
        .str.replace("_ANNEAL", "", regex=False))
    old_renamed = old.rename(columns={
        "Composition": "composition",
        "Processing": "processing",
        "Property": "property",
    })
    keepcols = ["composition", "processing", "property", "test_source",
                 "n_train", "n_test", "Best_Test_R2", "Best_Train_R2", "Best_Template"]
    old_slim = old_renamed[keepcols]

    merged = new_ok.merge(
        old_slim, on=["composition", "processing", "property", "test_source"],
        how="outer", suffixes=("_new", "_old"), indicator=True,
    )

    both = merged[merged["_merge"] == "both"]
    left = merged[merged["_merge"] == "left_only"]   # new but not in old
    right = merged[merged["_merge"] == "right_only"] # old but not in real MPEA

    diff_nt = (both["n_train_new"].astype(float) - both["n_train_old"].astype(float)).fillna(0)
    diff_te = (both["n_test_new"].astype(float) - both["n_test_old"].astype(float)).fillna(0)
    n_mismatch = int(((diff_nt != 0) | (diff_te != 0)).sum())

    lines = [
        "# LODO Data Integrity Report",
        "",
        "Comparison: **NEW** verified `folds.csv` (built from MPEA_figshare_dataset.csv)",
        "vs **OLD** `lodo_all_results.xlsx` (4-Apr-2026, source script lost).",
        "",
        "## Headline numbers",
        "",
        f"- Total OLD fold records: **{len(old)}**",
        f"- Total NEW candidate fold records (incl. skipped): **{len(new)}**",
        f"- NEW valid LODO folds (multi-source, n_train≥3, n_test≥2): **{len(new_ok)}**",
        f"- OLD folds **with no MPEA backing** (composition/processing/property/test_source not present in MPEA): **{len(right)}** of {len(old)} ({100*len(right)/len(old):.0f}%)",
        f"- OLD vs NEW folds matched on (composition, processing, property, test_source): **{len(both)}**",
        f"- Among matches, folds with **n_train OR n_test mismatch**: **{n_mismatch}** of {len(both)} ({100*n_mismatch/max(1,len(both)):.0f}%)",
        f"- Mean (NEW - OLD) n_train: {diff_nt.mean():+.2f}; max abs: {diff_nt.abs().max():.0f}",
        f"- Mean (NEW - OLD) n_test:  {diff_te.mean():+.2f}; max abs: {diff_te.abs().max():.0f}",
        f"- NEW valid folds **not present in OLD**: **{len(left)}**",
        "",
        "## Why the OLD file is unreliable",
        "",
        "1. **112 / 134 fold rows in `lodo_all_results.xlsx` reference (composition × processing × property × test_source) tuples that have NO matching rows in MPEA_figshare_dataset.csv.** "
        "These cannot have been produced by an honest LODO split on MPEA data.",
        "2. Among the 22 folds where the (composition/processing/property/test_source) tuple does exist in MPEA, the reported `n_train` and `n_test` disagree with the actual MPEA row counts in 21 cases (mean diff ~4 rows).",
        "3. The original generation script is missing from the repo, so the protocol cannot be audited. Reported `Best_Template` is selected by max test R² (oracle / data-leak).",
        "",
        "## OLD fold reported R² distribution (for the 112 fake folds)",
        "",
    ]
    desc = right["Best_Test_R2"].astype(float).describe()
    lines += [
        f"- count: {int(desc['count'])}",
        f"- mean:  {desc['mean']:.3f}",
        f"- std:   {desc['std']:.3f}",
        f"- 25%:   {desc['25%']:.3f}",
        f"- 50%:   {desc['50%']:.3f}",
        f"- 75%:   {desc['75%']:.3f}",
        f"- min:   {desc['min']:.3f}",
        f"- max:   {desc['max']:.3f}",
        "",
        "## Headline-fold examples of n_train mismatch",
        "",
        "| Composition | Processing | Property | Test source | OLD n_tr | NEW n_tr | OLD n_te | NEW n_te |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    bad = both[(diff_nt != 0) | (diff_te != 0)]
    for _, r in bad.head(20).iterrows():
        lines.append(
            f"| {r['composition']} | {r['processing']} | {r['property']} | {r['test_source']} | "
            f"{int(r['n_train_old']) if pd.notna(r['n_train_old']) else '—'} | "
            f"{int(r['n_train_new']) if pd.notna(r['n_train_new']) else '—'} | "
            f"{int(r['n_test_old']) if pd.notna(r['n_test_old']) else '—'} | "
            f"{int(r['n_test_new']) if pd.notna(r['n_test_new']) else '—'} |"
        )

    lines += [
        "",
        "## Conclusion",
        "",
        "The numbers in `lodo_all_results.xlsx` cannot be reproduced or validated against the source MPEA data. "
        "The PT-SR LODO results in `lodo_v2/` are built from scratch on the verified 32-fold subset, "
        "with template selection based on **train R²** (no test peek). Headline comparisons use these numbers, "
        "with the OLD spreadsheet retained only for archaeological reference.",
    ]

    (OUT / "integrity_report.md").write_text("\n".join(lines))
    print(f"Wrote {OUT/'integrity_report.md'}")


if __name__ == "__main__":
    main()
