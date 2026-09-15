import json, os
from pathlib import Path
import numpy as np
import pandas as pd

REPO   = Path(__file__).resolve().parent.parent
RES    = REPO / "results"
OUT    = RES / "final_results.xlsx"

TARGETS = [
    "Youngs_modulus", "UTS", "Disloc_0pct", "Disloc_20pct",
    "FCC_0pct", "HCP_0pct", "BCC_0pct", "FCC_20pct",
    "HCP_20pct", "BCC_20pct", "Other_0pct", "Other_20pct"
]
SEEDS = [0, 1, 2, 3, 4]
FOLDS = [0, 1, 2, 3, 4]
TEMPLATES = [
    "additive", "multiplicative", "thermal_softening", "arrhenius",
    "power_law", "free_2stage", "free_1stage", "entropy_weighted", "calphad"
]
ML_MODELS = [
    "LinearRegression", "Ridge", "Lasso", "RandomForest",
    "GradientBoosting", "XGBoost", "SVR", "MLP_small", "MLP_large"
]
DL_MODELS = ["DeepMLP", "TabularTransformer", "AttentionMLP"]
ML_OPTUNA_MODELS = ["RandomForest", "GradientBoosting", "XGBoost", "SVR", "Ridge", "Lasso", "MLP"]
DL_OPTUNA_MODELS = ["DeepMLP", "TabularTransformer", "AttentionMLP"]
ML_CV_OPTUNA_MODELS = ["RandomForest", "GradientBoosting", "XGBoost", "SVR", "Ridge", "Lasso", "MLP"]
DL_CV_OPTUNA_MODELS = ["DeepMLP", "TabularTransformer", "AttentionMLP"]

SF_COLS = [f"s{s}_f{f}" for s in SEEDS for f in FOLDS]
S_COLS = [f"s{s}" for s in SEEDS]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def mean_std(vals):
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return None, None
    return float(np.mean(arr)), float(np.std(arr))

def flat_vals(seed_fold_dict):
    return [seed_fold_dict.get(s, {}).get(f, None) for s in SEEDS for f in FOLDS]


print("Loading ML baselines...")
ml_data = {}
for s in SEEDS:
    path = str(RES / f"ml_baseline_results_s{s}.json")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        continue
    d = load_json(path)
    for tgt in TARGETS:
        ml_data.setdefault(tgt, {m: {} for m in ML_MODELS})
        for m in ML_MODELS:
            ml_data[tgt][m][s] = d[tgt][m]["fold_r2s"]


print("Loading GPR...")
gpr_data = {}
for s in SEEDS:
    path = str(RES / f"gpr_results_s{s}.json")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        continue
    d = load_json(path)
    for tgt in TARGETS:
        gpr_data.setdefault(tgt, {})
        gpr_data[tgt][s] = d[tgt]["fold_r2s"]


print("Loading DL models...")
dl_data = {}
for s in SEEDS:
    path = str(RES / f"deep_model_results_s{s}.json")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        continue
    d = load_json(path)
    for tgt in TARGETS:
        dl_data.setdefault(tgt, {m: {} for m in DL_MODELS})
        for m in DL_MODELS:
            dl_data[tgt][m][s] = d[tgt][m]["fold_r2s"]


print("Loading 9-template SR...")

CV = RES / "cv_results"


def load_runs(csv_name, stem, has_template):
    csv_path = CV / csv_name
    rows = []
    if csv_path.exists():
        df = pd.read_csv(csv_path).rename(columns={"test_r2": "r2"})
        if "time_s" not in df.columns:
            df["time_s"] = np.nan
        rows = df.to_dict("records")
    else:
        for seed in SEEDS:
            path = str(RES / f"{stem}_s{seed}to{seed}.json")
            if not os.path.exists(path):
                print(f"  MISSING: {path}")
                continue
            for key, v in load_json(path).items():
                parts = key.split("__")
                rows.append({
                    "seed": seed, "target": parts[0],
                    "template": parts[1] if has_template else None,
                    "fold": int(parts[2 if has_template else 1]
                                .replace("f", "")),
                    "r2": v.get("test_r2"), "time_s": v.get("time")})

    nested = {}
    for r in rows:
        if has_template:
            (nested.setdefault(r["template"], {})
                   .setdefault(r["target"], {})
                   .setdefault(r["seed"], {}))[r["fold"]] = r["r2"]
        else:
            (nested.setdefault(r["target"], {})
                   .setdefault(r["seed"], {}))[r["fold"]] = r["r2"]
    return rows, nested


sr9_raw, sr9_data = load_runs("sr_9templates_raw.csv", "checkpoint", True)


print("Loading Freeform SR...")
ff_raw, ff_data = load_runs("sr_freeform.csv", "checkpoint_freeform", False)


print("Loading 6-feat control SR...")
sf6_raw, sf6_data = load_runs("sr_6feat_control.csv", "checkpoint_6feat", False)


print("Loading Shuffled-y...")
shuf_raw, shuf_data = load_runs("shuffled_y_raw.csv", "checkpoint_shuffled", True)


print("Loading ML Optuna fullfit...")
ml_optuna_data = {}
ml_opt_path = str(RES / "ml_optuna_results.json")
if os.path.exists(ml_opt_path):
    d = load_json(ml_opt_path)
    for tgt in TARGETS:
        ml_optuna_data[tgt] = {}
        for m in ML_OPTUNA_MODELS:
            if tgt in d and m in d[tgt]:
                ml_optuna_data[tgt][m] = d[tgt][m]
else:
    print(f"  MISSING: {ml_opt_path}")


print("Loading DL Optuna fullfit...")
dl_optuna_data = {}
dl_opt_path = str(RES / "dl_optuna_results.json")
if os.path.exists(dl_opt_path):
    d = load_json(dl_opt_path)
    for tgt in TARGETS:
        dl_optuna_data[tgt] = {}
        for m in DL_OPTUNA_MODELS:
            if tgt in d and m in d[tgt]:
                dl_optuna_data[tgt][m] = d[tgt][m]
else:
    print(f"  MISSING: {dl_opt_path}")


print("Loading ML Optuna CV...")
ml_cv_optuna_data = {}
ml_cv_path = str(RES / "ml_optuna_cv_results.json")
if os.path.exists(ml_cv_path):
    d = load_json(ml_cv_path)
    for tgt in TARGETS:
        ml_cv_optuna_data[tgt] = {}
        for m in ML_CV_OPTUNA_MODELS:
            if tgt in d and m in d[tgt]:
                ml_cv_optuna_data[tgt][m] = d[tgt][m]
else:
    print(f"  MISSING: {ml_cv_path}")


print("Loading DL Optuna CV...")
dl_cv_optuna_data = {}
dl_cv_path = str(RES / "dl_optuna_cv_results.json")
if os.path.exists(dl_cv_path):
    d = load_json(dl_cv_path)
    for tgt in TARGETS:
        dl_cv_optuna_data[tgt] = {}
        for m in DL_CV_OPTUNA_MODELS:
            if tgt in d and m in d[tgt]:
                dl_cv_optuna_data[tgt][m] = d[tgt][m]
else:
    print(f"  MISSING: {dl_cv_path}")


print(f"\nWriting Excel to {OUT} ...")
writer = pd.ExcelWriter(OUT, engine="xlsxwriter")
wb = writer.book


ws = wb.add_worksheet("README")
info = [
    ("Dataset", "HEA CoCrCuFeNi, 228 compositions × 3 temperatures = 684 rows, 12 targets"),
    ("Feature set", "13 features: 5 compositions (Co,Cr,Cu,Fe,Ni %) + 8 elemental features"),
    ("Validation", "5-fold GroupKFold (grouped by composition) × 5 seeds = 25 R² per model"),
    ("Column naming", "s0_f0 = seed 0 fold 0, s0_f1 = seed 0 fold 1, ..., s4_f4 = seed 4 fold 4"),
    ("", ""),
    ("Sheet", "Contents"),
    ("Model_Comparison", "Summary: mean ± std R² for all models × 12 targets"),
    ("ML_Baselines", "9 scikit-learn models × 12 targets: s0_f0 … s4_f4 (25 values)"),
    ("GPR", "Gaussian Process Regression: s0_f0 … s4_f4 per target (n_restarts=50 for s1,s2 fix)"),
    ("DL_Models", "DeepMLP, TabularTransformer, AttentionMLP: s0_f0 … s4_f4 per model × target"),
    ("SR_9templates_Summary", "9 physics templates × 12 targets: mean ± std"),
    ("SR_9templates_Raw", "All 2700 SR runs: seed, template, target, fold, R², time_s"),
    ("SR_Freeform", "Free-form SR: s0_f0 … s4_f4 per target"),
    ("SR_6feat_Control", "6-feature control SR: s0_f0 … s4_f4 per target"),
    ("Shuffled_y_Summary", "Permutation test: mean/median R² per template × target (shuffled labels)"),
    ("Shuffled_y_Raw", "All shuffled-y runs: seed, template, target, fold, R²"),
    ("", ""),
    ("Mac Studio (M3 Ultra)", "9-template SR (checkpoint_s{0-4}to{0-4}.json)"),
    ("Mac Mini (M4 Pro)", "ML/DL/GPR/Freeform/6feat/Shuffled results"),
    ("GPR fix", "run_gpr_fix_proper.py: n_restarts_optimizer=50, random_state=seed for s1 Youngs+HCP_20, s2 Youngs"),
]
fmt_bold = wb.add_format({"bold": True})
for r, (k, v) in enumerate(info):
    ws.write(r, 0, k, fmt_bold if k else None)
    ws.write(r, 1, v)
ws.set_column(0, 0, 25)
ws.set_column(1, 1, 90)


rows = []
for m in ML_MODELS:
    row = {"Model": m, "Type": "ML"}
    all_vals = []
    for tgt in TARGETS:
        vals = [ml_data.get(tgt, {}).get(m, {}).get(s, [None]*5)[f]
                for s in SEEDS for f in FOLDS]
        mu, sd = mean_std(vals)
        row[tgt] = f"{mu:.4f} ± {sd:.4f}" if mu is not None else "N/A"
        if mu is not None: all_vals.append(mu)
    row["Overall_mean"] = float(np.mean(all_vals)) if all_vals else None
    rows.append(row)

row = {"Model": "GPR", "Type": "GPR"}
all_vals = []
for tgt in TARGETS:
    vals = [gpr_data.get(tgt, {}).get(s, [None]*5)[f]
            for s in SEEDS for f in FOLDS]
    mu, sd = mean_std(vals)
    row[tgt] = f"{mu:.4f} ± {sd:.4f}" if mu is not None else "N/A"
    if mu is not None: all_vals.append(mu)
row["Overall_mean"] = float(np.mean(all_vals)) if all_vals else None
rows.append(row)

for m in DL_MODELS:
    row = {"Model": m, "Type": "DL"}
    all_vals = []
    for tgt in TARGETS:
        vals = [dl_data.get(tgt, {}).get(m, {}).get(s, [None]*5)[f]
                for s in SEEDS for f in FOLDS]
        mu, sd = mean_std(vals)
        row[tgt] = f"{mu:.4f} ± {sd:.4f}" if mu is not None else "N/A"
        if mu is not None: all_vals.append(mu)
    row["Overall_mean"] = float(np.mean(all_vals)) if all_vals else None
    rows.append(row)

for tmpl in TEMPLATES:
    row = {"Model": f"SR_{tmpl}", "Type": "SR_template"}
    all_vals = []
    for tgt in TARGETS:
        vals = [sr9_data.get(tmpl, {}).get(tgt, {}).get(s, {}).get(f, None)
                for s in SEEDS for f in FOLDS]
        mu, sd = mean_std(vals)
        row[tgt] = f"{mu:.4f} ± {sd:.4f}" if mu is not None else "N/A"
        if mu is not None: all_vals.append(mu)
    row["Overall_mean"] = float(np.mean(all_vals)) if all_vals else None
    rows.append(row)

row = {"Model": "SR_freeform", "Type": "SR_freeform"}
all_vals = []
for tgt in TARGETS:
    vals = [ff_data.get(tgt, {}).get(s, {}).get(f, None)
            for s in SEEDS for f in FOLDS]
    mu, sd = mean_std(vals)
    row[tgt] = f"{mu:.4f} ± {sd:.4f}" if mu is not None else "N/A"
    if mu is not None: all_vals.append(mu)
row["Overall_mean"] = float(np.mean(all_vals)) if all_vals else None
rows.append(row)

row = {"Model": "SR_6feat_control", "Type": "SR_6feat"}
all_vals = []
for tgt in TARGETS:
    vals = [sf6_data.get(tgt, {}).get(s, {}).get(f, None)
            for s in SEEDS for f in FOLDS]
    mu, sd = mean_std(vals)
    row[tgt] = f"{mu:.4f} ± {sd:.4f}" if mu is not None else "N/A"
    if mu is not None: all_vals.append(mu)
row["Overall_mean"] = float(np.mean(all_vals)) if all_vals else None
rows.append(row)

df_cmp = pd.DataFrame(rows)[["Model", "Type", "Overall_mean"] + TARGETS]
df_cmp.to_excel(writer, sheet_name="Model_Comparison", index=False)
print("  ✓ Model_Comparison")


ml_rows = []
for m in ML_MODELS:
    for tgt in TARGETS:
        vals = [ml_data.get(tgt, {}).get(m, {}).get(s, [None]*5)[f]
                for s in SEEDS for f in FOLDS]
        mu, sd = mean_std(vals)
        row = {"Model": m, "Target": tgt, "mean_R2": mu, "std_R2": sd}
        for col, v in zip(SF_COLS, vals):
            row[col] = v
        ml_rows.append(row)

pd.DataFrame(ml_rows)[["Model", "Target", "mean_R2", "std_R2"] + SF_COLS].to_excel(
    writer, sheet_name="ML_Baselines", index=False)
print("  ✓ ML_Baselines")


gpr_rows = []
for tgt in TARGETS:
    vals = [gpr_data.get(tgt, {}).get(s, [None]*5)[f]
            for s in SEEDS for f in FOLDS]
    mu, sd = mean_std(vals)
    row = {"Target": tgt, "mean_R2": mu, "std_R2": sd}
    for col, v in zip(SF_COLS, vals):
        row[col] = v
    gpr_rows.append(row)

pd.DataFrame(gpr_rows)[["Target", "mean_R2", "std_R2"] + SF_COLS].to_excel(
    writer, sheet_name="GPR", index=False)
print("  ✓ GPR")


dl_rows = []
for m in DL_MODELS:
    for tgt in TARGETS:
        vals = [dl_data.get(tgt, {}).get(m, {}).get(s, [None]*5)[f]
                for s in SEEDS for f in FOLDS]
        mu, sd = mean_std(vals)
        row = {"Model": m, "Target": tgt, "mean_R2": mu, "std_R2": sd}
        for col, v in zip(SF_COLS, vals):
            row[col] = v
        dl_rows.append(row)

pd.DataFrame(dl_rows)[["Model", "Target", "mean_R2", "std_R2"] + SF_COLS].to_excel(
    writer, sheet_name="DL_Models", index=False)
print("  ✓ DL_Models")


sr9_sum_rows = []
for tmpl in TEMPLATES:
    for tgt in TARGETS:
        vals = [sr9_data.get(tmpl, {}).get(tgt, {}).get(s, {}).get(f, None)
                for s in SEEDS for f in FOLDS]
        mu, sd = mean_std(vals)
        row = {"Template": tmpl, "Target": tgt, "mean_R2": mu, "std_R2": sd}
        for col, v in zip(SF_COLS, vals):
            row[col] = v
        sr9_sum_rows.append(row)

pd.DataFrame(sr9_sum_rows)[["Template", "Target", "mean_R2", "std_R2"] + SF_COLS].to_excel(
    writer, sheet_name="SR_9templates_Summary", index=False)
print("  ✓ SR_9templates_Summary")


df_sr9_raw = pd.DataFrame(sr9_raw)[["seed", "template", "target", "fold", "r2", "time_s"]]
df_sr9_raw.sort_values(["seed", "template", "target", "fold"]).reset_index(drop=True).to_excel(
    writer, sheet_name="SR_9templates_Raw", index=False)
print("  ✓ SR_9templates_Raw")


ff_rows = []
for tgt in TARGETS:
    vals = [ff_data.get(tgt, {}).get(s, {}).get(f, None)
            for s in SEEDS for f in FOLDS]
    mu, sd = mean_std(vals)
    row = {"Target": tgt, "mean_R2": mu, "std_R2": sd}
    for col, v in zip(SF_COLS, vals):
        row[col] = v
    ff_rows.append(row)

pd.DataFrame(ff_rows)[["Target", "mean_R2", "std_R2"] + SF_COLS].to_excel(
    writer, sheet_name="SR_Freeform", index=False)
print("  ✓ SR_Freeform")


sf6_rows = []
for tgt in TARGETS:
    vals = [sf6_data.get(tgt, {}).get(s, {}).get(f, None)
            for s in SEEDS for f in FOLDS]
    mu, sd = mean_std(vals)
    row = {"Target": tgt, "mean_R2": mu, "std_R2": sd}
    for col, v in zip(SF_COLS, vals):
        row[col] = v
    sf6_rows.append(row)

pd.DataFrame(sf6_rows)[["Target", "mean_R2", "std_R2"] + SF_COLS].to_excel(
    writer, sheet_name="SR_6feat_Control", index=False)
print("  ✓ SR_6feat_Control")


shuf_sum_rows = []
for tmpl in TEMPLATES:
    for tgt in TARGETS:
        vals = [shuf_data.get(tmpl, {}).get(tgt, {}).get(s, {}).get(f, None)
                for s in SEEDS for f in FOLDS]
        vals_clean = [v for v in vals if v is not None]
        mu, sd = mean_std(vals_clean)
        row = {"Template": tmpl, "Target": tgt,
               "N_values": len(vals_clean),
               "mean_R2": mu, "std_R2": sd,
               "median_R2": float(np.median(vals_clean)) if vals_clean else None,
               "pct_below_0": float(np.mean(np.array(vals_clean) < 0) * 100) if vals_clean else None}
        shuf_sum_rows.append(row)

pd.DataFrame(shuf_sum_rows).to_excel(writer, sheet_name="Shuffled_y_Summary", index=False)
print("  ✓ Shuffled_y_Summary")


df_shuf_raw = pd.DataFrame(shuf_raw)[["seed", "template", "target", "fold", "r2"]]
df_shuf_raw.sort_values(["seed", "template", "target", "fold"]).reset_index(drop=True).to_excel(
    writer, sheet_name="Shuffled_y_Raw", index=False)
print("  ✓ Shuffled_y_Raw")


if ml_optuna_data:
    ml_opt_rows = []
    for m in ML_OPTUNA_MODELS:
        for tgt in TARGETS:
            entry = ml_optuna_data.get(tgt, {}).get(m, {})
            seed_r2s = entry.get("seed_train_r2s", [None]*5)
            mu = entry.get("mean_train_r2", None)
            sd = entry.get("std_train_r2", None)
            best_val = entry.get("optuna_best_val_r2", None)
            row = {"Model": m, "Target": tgt,
                   "mean_train_R2": mu, "std_train_R2": sd,
                   "optuna_best_val_R2": best_val}
            for col, v in zip(S_COLS, seed_r2s):
                row[col] = v
            ml_opt_rows.append(row)
    pd.DataFrame(ml_opt_rows)[["Model", "Target", "mean_train_R2", "std_train_R2",
                                "optuna_best_val_R2"] + S_COLS].to_excel(
        writer, sheet_name="ML_Optuna_Fullfit", index=False)
    print("  ✓ ML_Optuna_Fullfit")


if dl_optuna_data:
    dl_opt_rows = []
    for m in DL_OPTUNA_MODELS:
        for tgt in TARGETS:
            entry = dl_optuna_data.get(tgt, {}).get(m, {})
            seed_r2s = entry.get("seed_train_r2s", [None]*5)
            mu = entry.get("mean_train_r2", None)
            sd = entry.get("std_train_r2", None)
            best_val = entry.get("optuna_best_val_r2", None)
            row = {"Model": m, "Target": tgt,
                   "mean_train_R2": mu, "std_train_R2": sd,
                   "optuna_best_val_R2": best_val}
            for col, v in zip(S_COLS, seed_r2s):
                row[col] = v
            dl_opt_rows.append(row)
    pd.DataFrame(dl_opt_rows)[["Model", "Target", "mean_train_R2", "std_train_R2",
                                "optuna_best_val_R2"] + S_COLS].to_excel(
        writer, sheet_name="DL_Optuna_Fullfit", index=False)
    print("  ✓ DL_Optuna_Fullfit")


if ml_cv_optuna_data:
    ml_cv_rows = []
    for m in ML_CV_OPTUNA_MODELS:
        for tgt in TARGETS:
            entry = ml_cv_optuna_data.get(tgt, {}).get(m, {})
            sfr = entry.get("seed_fold_r2s", [[None]*5]*5)
            vals = [sfr[s][f] for s in range(5) for f in range(5)]
            mu = entry.get("mean_r2", None)
            sd = entry.get("std_r2", None)
            row = {"Model": m, "Target": tgt, "mean_R2": mu, "std_R2": sd}
            for col, v in zip(SF_COLS, vals):
                row[col] = v
            ml_cv_rows.append(row)
    pd.DataFrame(ml_cv_rows)[["Model", "Target", "mean_R2", "std_R2"] + SF_COLS].to_excel(
        writer, sheet_name="ML_Baselines_Optuna", index=False)
    print("  ✓ ML_Baselines_Optuna")


if dl_cv_optuna_data:
    dl_cv_rows = []
    for m in DL_CV_OPTUNA_MODELS:
        for tgt in TARGETS:
            entry = dl_cv_optuna_data.get(tgt, {}).get(m, {})
            sfr = entry.get("seed_fold_r2s", [[None]*5]*5)
            vals = [sfr[s][f] for s in range(5) for f in range(5)]
            mu = entry.get("mean_r2", None)
            sd = entry.get("std_r2", None)
            row = {"Model": m, "Target": tgt, "mean_R2": mu, "std_R2": sd}
            for col, v in zip(SF_COLS, vals):
                row[col] = v
            dl_cv_rows.append(row)
    pd.DataFrame(dl_cv_rows)[["Model", "Target", "mean_R2", "std_R2"] + SF_COLS].to_excel(
        writer, sheet_name="DL_Models_Optuna", index=False)
    print("  ✓ DL_Models_Optuna")

writer.close()
print(f"\n✅ Done! Saved to {OUT}")
print(f"   File size: {os.path.getsize(OUT)/1024:.0f} KB")


print("\n" + "="*60)
print("QUICK SUMMARY (mean R² across all 12 targets)")
print("="*60)
summary = {}
for m in ML_MODELS:
    vals = [ml_data.get(t, {}).get(m, {}).get(s, [None]*5)[f]
            for t in TARGETS for s in SEEDS for f in FOLDS]
    mu, _ = mean_std(vals); summary[m] = mu
vals = [gpr_data.get(t, {}).get(s, [None]*5)[f]
        for t in TARGETS for s in SEEDS for f in FOLDS]
mu, _ = mean_std(vals); summary["GPR"] = mu
for m in DL_MODELS:
    vals = [dl_data.get(t, {}).get(m, {}).get(s, [None]*5)[f]
            for t in TARGETS for s in SEEDS for f in FOLDS]
    mu, _ = mean_std(vals); summary[m] = mu
for tmpl in TEMPLATES:
    vals = [sr9_data.get(tmpl, {}).get(t, {}).get(s, {}).get(f, None)
            for t in TARGETS for s in SEEDS for f in FOLDS]
    mu, _ = mean_std(vals); summary[f"SR_{tmpl}"] = mu
vals = [ff_data.get(t, {}).get(s, {}).get(f, None)
        for t in TARGETS for s in SEEDS for f in FOLDS]
mu, _ = mean_std(vals); summary["SR_freeform"] = mu

for name, mu in sorted(summary.items(), key=lambda x: -(x[1] or -999)):
    if mu is not None:
        print(f"  {name:30s}: {mu:.4f}")
