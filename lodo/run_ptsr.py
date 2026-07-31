"""
Step 3 main: run PT-SR (9 templates x 5 seeds) on every valid LODO fold.

Outputs (incremental, JSON + CSV):
  lodo/pt_sr_results.json
  lodo/pt_sr_results.csv

Each row in the CSV has:
  fold_id, composition, processing, property, test_source,
  template, seed, train_r2, test_r2, test_mae, S1_eq, S2_eq, error

Best_Template selection (FAIR, no test peek):
  - For each fold, average train_r2 across seeds for each template.
  - Pick template with highest mean train_r2.
  - Best_Test_R2_fair = mean test_r2 of that template across seeds.

ORACLE comparison (peeks at test):
  - For each fold, pick template with highest mean test_r2.
  - Best_Test_R2_oracle = that mean test_r2.

NEVER use sleep. Runs serially through folds. Checkpoints incrementally.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import json
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "lodo"))
from features import load_mpea, build_fold
from ptsr_templates import TEMPLATES, run_template_on_fold

BASE = str(Path(__file__).resolve().parent)
OUT = Path(f"{BASE}/lodo_v2")
OUT.mkdir(exist_ok=True)

CSV_OUT = OUT / "pt_sr_results.csv"
JSON_OUT = OUT / "pt_sr_results.json"
LOG_OUT = OUT / "pt_sr_run.log"

SEEDS = [0, 1, 2, 3, 4]
NITER = 30
TEMPLATE_ORDER = list(TEMPLATES.keys())  # arrhenius, free2, therm, additive, multiplicative, power, entropy, calphad, free1


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG_OUT, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    folds_df = pd.read_csv(OUT / "folds.csv")
    valid = folds_df[folds_df["status"].str.startswith("ok")].copy()
    log(f"PT-SR LODO v2 starting: {len(valid)} valid folds, {len(TEMPLATE_ORDER)} templates, "
        f"{len(SEEDS)} seeds = {len(valid)*len(TEMPLATE_ORDER)*len(SEEDS)} runs")

    mpea = load_mpea()

    # ---- Resume from existing CSV if present ----
    if CSV_OUT.exists():
        existing = pd.read_csv(CSV_OUT)
        done_keys = set(zip(existing["fold_id"], existing["template"], existing["seed"]))
        log(f"Resuming: {len(done_keys)} runs already cached")
        rows = existing.to_dict("records")
    else:
        done_keys = set()
        rows = []

    n_total = len(valid) * len(TEMPLATE_ORDER) * len(SEEDS)
    n_done = len(done_keys)
    t_start = time.time()

    for fi, (_, frow) in enumerate(valid.iterrows()):
        fold = build_fold(frow["composition"], frow["processing"],
                           frow["property"], frow["test_source"], mpea)
        if fold is None:
            log(f"  [{frow['fold_id']}] build_fold returned None, skipping")
            continue
        # Verify counts match
        assert fold["X_train"].shape[0] == frow["n_train"], "n_train mismatch"
        assert fold["X_test"].shape[0] == frow["n_test"], "n_test mismatch"

        for tmpl in TEMPLATE_ORDER:
            for seed in SEEDS:
                key = (frow["fold_id"], tmpl, seed)
                if key in done_keys:
                    continue
                t0 = time.time()
                res = run_template_on_fold(tmpl, fold, seed, niter=NITER)
                dt = time.time() - t0
                row = {
                    "fold_id": frow["fold_id"],
                    "composition": frow["composition"],
                    "processing": frow["processing"],
                    "property": frow["property"],
                    "test_source": frow["test_source"],
                    "n_train": frow["n_train"],
                    "n_test": frow["n_test"],
                    "has_overlap": frow["has_overlap"],
                    "y_test_constant": frow["y_test_constant"],
                    "template": tmpl,
                    "seed": seed,
                    "train_r2": res["train_r2"],
                    "test_r2": res["test_r2"],
                    "test_mae": res["test_mae"],
                    "S1_eq": res["S1"],
                    "S2_eq": res["S2"],
                    "error": res["error"],
                    "time_s": dt,
                }
                rows.append(row)
                n_done += 1
                done_keys.add(key)

                # Checkpoint every 20 runs
                if n_done % 20 == 0:
                    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
                    elapsed = time.time() - t_start
                    rate = (n_done - len(done_keys) + 20) / max(1, elapsed)
                    log(f"  progress {n_done}/{n_total} ({100*n_done/n_total:.1f}%), "
                        f"last={tmpl}@s{seed} train={res['train_r2']:.3f} test={res['test_r2']:.3f} ({dt:.1f}s)")
        # End-of-fold checkpoint
        pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
        log(f"  fold {fi+1}/{len(valid)} done: {frow['fold_id']}")

    # ---- Final write ----
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    log(f"Wrote {CSV_OUT}: {len(df)} rows")

    # ---- Aggregate per (fold, template): mean across seeds ----
    agg = (
        df.groupby(["fold_id", "composition", "processing", "property", "test_source",
                    "n_train", "n_test", "has_overlap", "y_test_constant", "template"])
          .agg(train_r2_mean=("train_r2", "mean"),
                test_r2_mean=("test_r2", "mean"),
                train_r2_std=("train_r2", "std"),
                test_r2_std=("test_r2", "std"),
                test_mae_mean=("test_mae", "mean"))
          .reset_index()
    )
    agg.to_csv(OUT / "pt_sr_per_template.csv", index=False)

    # ---- Best per fold: FAIR (train R2) and ORACLE (test R2) ----
    best_records = []
    for fid, grp in agg.groupby("fold_id"):
        # FAIR: pick template with highest mean train_r2 (tie-break: alphabetical)
        valid_train = grp[grp["train_r2_mean"].notna()]
        if len(valid_train) == 0:
            best_fair = {"template": "", "train_r2": np.nan, "test_r2": np.nan}
        else:
            top_fair = valid_train.sort_values(
                ["train_r2_mean", "template"], ascending=[False, True]
            ).iloc[0]
            best_fair = {
                "template": top_fair["template"],
                "train_r2": top_fair["train_r2_mean"],
                "test_r2": top_fair["test_r2_mean"],
            }

        # ORACLE: pick template with highest mean test_r2
        valid_test = grp[grp["test_r2_mean"].notna()]
        if len(valid_test) == 0:
            best_oracle = {"template": "", "train_r2": np.nan, "test_r2": np.nan}
        else:
            top_oracle = valid_test.sort_values(
                ["test_r2_mean", "template"], ascending=[False, True]
            ).iloc[0]
            best_oracle = {
                "template": top_oracle["template"],
                "train_r2": top_oracle["train_r2_mean"],
                "test_r2": top_oracle["test_r2_mean"],
            }

        first_row = grp.iloc[0]
        best_records.append({
            "fold_id": fid,
            "composition": first_row["composition"],
            "processing": first_row["processing"],
            "property": first_row["property"],
            "test_source": first_row["test_source"],
            "n_train": int(first_row["n_train"]),
            "n_test": int(first_row["n_test"]),
            "has_overlap": bool(first_row["has_overlap"]),
            "best_template_fair": best_fair["template"],
            "best_train_r2_fair": best_fair["train_r2"],
            "best_test_r2_fair": best_fair["test_r2"],
            "best_template_oracle": best_oracle["template"],
            "best_test_r2_oracle": best_oracle["test_r2"],
        })

    best_df = pd.DataFrame(best_records)
    best_df.to_csv(OUT / "pt_sr_best.csv", index=False)
    log(f"Wrote {OUT/'pt_sr_best.csv'}: {len(best_df)} folds")

    # ---- Final JSON summary ----
    summary = {
        "n_folds": len(best_df),
        "median_test_r2_fair": float(best_df["best_test_r2_fair"].median()),
        "mean_test_r2_fair": float(best_df["best_test_r2_fair"].mean()),
        "median_test_r2_oracle": float(best_df["best_test_r2_oracle"].median()),
        "mean_test_r2_oracle": float(best_df["best_test_r2_oracle"].mean()),
        "n_total_runs": len(df),
        "templates": TEMPLATE_ORDER,
        "seeds": SEEDS,
        "niter": NITER,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary -> {JSON_OUT}: {summary}")


if __name__ == "__main__":
    main()
