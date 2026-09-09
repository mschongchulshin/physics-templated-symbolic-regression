"""
Step 4: Run all 9 baselines on the verified LODO v2 folds.

Baselines (default published HPs, no tuning):
  1. Liu_RF                — RandomForestRegressor (deterministic, n_seeds=1)
  2. Liu_MLP               — 3-layer MLP, 5 seeds
  3. Wu_gplearn_RFR        — gplearn SymbolicTransformer + RFR, 5 seeds
  4. JMI_Stacking          — ExtraTrees + HistGBR + Lasso meta, 5 seeds
  5. Jain_DNN              — 3 hidden x 128, 5 seeds
  6. SciRep25_Transformer  — per-feature token Transformer, 5 seeds
  7. LESets_GNN            — composition-graph GNN (CGConv), 5 seeds
  8. SISSO                 — combinatorial features + L0 (deterministic, n_seeds=1)
  9. PySR_no_template      — free SR, 5 seeds

All use the SAME features built by features.build_fold (element fractions
+ Hume-Rothery + T). Output: baselines_results.csv with one row per
(fold, baseline, seed).
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
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "lodo"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lodo"))

warnings.filterwarnings("ignore")

# Re-use baseline implementations from existing baselines/lodo_comparison/run.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "lodo_baselines",
    str(Path(__file__).resolve().parent.parent
        / "baselines" / "lodo_comparison" / "run.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

fit_predict_liu_mlp = _mod.fit_predict_liu_mlp
fit_predict_liu_rf = _mod.fit_predict_liu_rf
fit_predict_jmi_stack = _mod.fit_predict_jmi_stack
fit_predict_wu_gp_rf = _mod.fit_predict_wu_gp_rf
fit_predict_jain_dnn = _mod.fit_predict_jain_dnn
fit_predict_scirep_transformer = _mod.fit_predict_scirep_transformer
fit_predict_lesets_gnn = _mod.fit_predict_lesets_gnn
fit_predict_sisso = _mod.fit_predict_sisso
fit_predict_pysr_free = _mod.fit_predict_pysr_free
safe_metrics = _mod.safe_metrics

from features import load_mpea, build_fold

BASE = str(Path(__file__).resolve().parent)
OUT = Path(f"{BASE}/lodo_v2")
CSV_OUT = OUT / "baselines_results.csv"
JSON_OUT = OUT / "baselines_results.json"
LOG_OUT = OUT / "baselines_run.log"

SEEDS = [0, 1, 2, 3, 4]


BASELINES = [
    # (name, fn, n_seeds, needs_elem_order)
    ("Liu_RF",                 fit_predict_liu_rf,             1,  False),
    ("Liu_MLP",                fit_predict_liu_mlp,            5,  False),
    ("Wu_gplearn_RFR",         fit_predict_wu_gp_rf,           5,  False),
    ("JMI_Stacking",           fit_predict_jmi_stack,          5,  False),
    ("Jain_DNN",               fit_predict_jain_dnn,           5,  False),
    ("SciRep25_Transformer",   fit_predict_scirep_transformer, 5,  False),
    ("LESets_GNN",             fit_predict_lesets_gnn,         5,  True),
    ("SISSO",                  fit_predict_sisso,              1,  False),
    ("PySR_no_template",       fit_predict_pysr_free,          5,  False),
]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG_OUT, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    folds_df = pd.read_csv(OUT / "folds.csv")
    valid = folds_df[folds_df["status"].str.startswith("ok")].copy()
    log(f"Baselines LODO v2: {len(valid)} folds x {len(BASELINES)} baselines")

    mpea = load_mpea()

    # Resume
    if CSV_OUT.exists():
        existing = pd.read_csv(CSV_OUT)
        done_keys = set(zip(existing["fold_id"], existing["baseline"], existing["seed"]))
        log(f"Resuming: {len(done_keys)} runs cached")
        rows = existing.to_dict("records")
    else:
        done_keys = set()
        rows = []

    n_total = sum(n_seeds for _, _, n_seeds, _ in BASELINES) * len(valid)
    n_done_init = len(done_keys)
    t_start = time.time()

    # For each fold, run all baselines (better cache locality of feature build)
    for fi, (_, frow) in enumerate(valid.iterrows()):
        fold = build_fold(frow["composition"], frow["processing"],
                           frow["property"], frow["test_source"], mpea)
        if fold is None:
            continue
        assert fold["X_train"].shape[0] == frow["n_train"]
        assert fold["X_test"].shape[0] == frow["n_test"]

        for name, fn, n_seeds, needs_elem in BASELINES:
            for seed in range(n_seeds):
                key = (frow["fold_id"], name, seed)
                if key in done_keys:
                    continue
                t0 = time.time()
                try:
                    if needs_elem:
                        pred = fn(fold["X_train"], fold["y_train"],
                                   fold["X_test"], seed, fold["elem_order"])
                    else:
                        pred = fn(fold["X_train"], fold["y_train"],
                                   fold["X_test"], seed)
                    m_test = safe_metrics(fold["y_test"], pred)
                    # train metrics too
                    try:
                        if needs_elem:
                            pred_tr = fn(fold["X_train"], fold["y_train"],
                                         fold["X_train"], seed, fold["elem_order"])
                        else:
                            pred_tr = fn(fold["X_train"], fold["y_train"],
                                         fold["X_train"], seed)
                        m_train = safe_metrics(fold["y_train"], pred_tr)
                    except Exception:
                        m_train = {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
                    err = ""
                except Exception as e:
                    m_test = {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
                    m_train = {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
                    err = f"{type(e).__name__}: {e}"
                dt = time.time() - t0
                row = {
                    "fold_id": frow["fold_id"],
                    "composition": frow["composition"],
                    "processing": frow["processing"],
                    "property": frow["property"],
                    "test_source": frow["test_source"],
                    "n_train": int(frow["n_train"]),
                    "n_test": int(frow["n_test"]),
                    "has_overlap": bool(frow["has_overlap"]),
                    "y_test_constant": bool(frow["y_test_constant"]),
                    "baseline": name,
                    "seed": seed,
                    "train_r2": m_train["R2"],
                    "test_r2": m_test["R2"],
                    "test_mae": m_test["MAE"],
                    "test_rmse": m_test["RMSE"],
                    "error": err,
                    "time_s": dt,
                }
                rows.append(row)
                done_keys.add(key)

                if len(rows) % 30 == 0:
                    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
                    log(f"  progress {len(done_keys)} done, last={name}@s{seed} on {frow['fold_id']} "
                        f"test_r2={m_test['R2']:.3f} ({dt:.1f}s)")
        pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
        log(f"  fold {fi+1}/{len(valid)} done: {frow['fold_id']}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    log(f"Wrote {CSV_OUT}: {len(df)} rows")

    # Aggregate per (fold, baseline): median across seeds
    agg = (
        df.groupby(["fold_id", "composition", "processing", "property",
                    "test_source", "n_train", "n_test", "has_overlap",
                    "y_test_constant", "baseline"])
          .agg(test_r2_median=("test_r2", "median"),
                test_r2_mean=("test_r2", "mean"),
                test_r2_std=("test_r2", "std"),
                test_mae_median=("test_mae", "median"),
                train_r2_median=("train_r2", "median"))
          .reset_index()
    )
    agg.to_csv(OUT / "baselines_per_fold.csv", index=False)

    # Top-level summary per baseline
    summary = {}
    for name in agg["baseline"].unique():
        sub = agg[agg["baseline"] == name]["test_r2_median"]
        summary[name] = {
            "n_folds_evaluated": int(sub.notna().sum()),
            "median_test_r2": float(sub.median(skipna=True)),
            "mean_test_r2": float(sub.mean(skipna=True)),
        }
    with open(JSON_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary -> {JSON_OUT}: {summary}")


if __name__ == "__main__":
    main()
