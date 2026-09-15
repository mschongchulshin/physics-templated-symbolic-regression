
import numpy as np
import pandas as pd
from mendeleev import element
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from pathlib import Path
import warnings

def _load_sheet(path, sheet_name):
    import pandas as _pd
    _t = int(str(sheet_name).rstrip("Kk"))
    _df = _pd.read_csv(path)
    return _df[_df["T_K"] == _t].drop(columns=["T_K"]).reset_index(drop=True)


warnings.filterwarnings("ignore")

R_GAS = 8.314
ELEMENTS = ["Co", "Cr", "Cu", "Fe", "Ni"]
COMP_COLS = ["Co(%)", "Cr(%)", "Cu(%)", "Fe(%)", "Ni(%)"]
SHEET_TEMP = {"80K": 80, "300K": 300, "1100K": 1100}

REPO     = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data/CoCrCuFeNi_684.csv"
OUT_DIR   = REPO / "data/features"

EXCLUDE_PROPS = {
    "price_per_kg", "political_stability_of_top_producer",
    "political_stability_of_top_reserve_holder",
    "abundance_crust", "abundance_sea",
    "production_concentration", "reserve_distribution",
    "relative_supply_risk", "glawe_number", "mendeleev_number",
    "pettifor_number", "atomic_number", "electrons", "protons",
    "neutrons", "mass_number",
}

print("=" * 60)
print("Step 1: Loading HEA data")
print("=" * 60)

frames = []
for sheet, T in SHEET_TEMP.items():
    df = _load_sheet(DATA_PATH, sheet)
    df = df[COMP_COLS].copy()
    df["T"] = T
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
print(f"Total data points: {len(data)}")

fracs = data[COMP_COLS].values / 100.0

print("\n" + "=" * 60)
print("Step 2: Extracting elemental properties from mendeleev")
print("=" * 60)

elem_objs = {sym: element(sym) for sym in ELEMENTS}

candidate_attrs = set()
for sym in ELEMENTS:
    e = elem_objs[sym]
    for attr in dir(e):
        if attr.startswith("_"):
            continue
        candidate_attrs.add(attr)

valid_props = {}
for attr in sorted(candidate_attrs):
    if attr in EXCLUDE_PROPS:
        continue
    vals = []
    all_valid = True
    for sym in ELEMENTS:
        try:
            v = getattr(elem_objs[sym], attr)
            if v is None:
                all_valid = False
                break
            v_float = float(v)
            if np.isnan(v_float) or np.isinf(v_float):
                all_valid = False
                break
            vals.append(v_float)
        except (TypeError, ValueError, AttributeError):
            all_valid = False
            break
    if all_valid and len(vals) == 5:
        if len(set(vals)) > 1:
            valid_props[attr] = np.array(vals)

print(f"Valid numeric properties (vary across elements): {len(valid_props)}")
for p in sorted(valid_props.keys()):
    vals = valid_props[p]
    print(f"  {p}: {dict(zip(ELEMENTS, vals))}")

print("\n" + "=" * 60)
print("Step 3: Computing composition-weighted features")
print("=" * 60)

feature_dict = {}

for prop_name, prop_vals in sorted(valid_props.items()):
    avg = fracs @ prop_vals
    feature_dict[f"{prop_name}_avg"] = avg

    diff_sq = (prop_vals[np.newaxis, :] - avg[:, np.newaxis]) ** 2
    delta = np.sqrt(np.sum(fracs * diff_sq, axis=1))
    feature_dict[f"{prop_name}_delta"] = delta

s_mix = np.zeros(len(data))
for j in range(5):
    mask = fracs[:, j] > 0
    s_mix[mask] += fracs[mask, j] * np.log(fracs[mask, j])
s_mix = -R_GAS * s_mix
feature_dict["S_mix"] = s_mix

feature_dict["T"] = data["T"].values

group_ids = np.array([elem_objs[sym].group_id for sym in ELEMENTS], dtype=float)
vec = fracs @ group_ids
feature_dict["VEC"] = vec

features_full = pd.DataFrame(feature_dict)

zero_var = features_full.columns[features_full.std() == 0]
if len(zero_var) > 0:
    print(f"Dropping zero-variance columns: {list(zero_var)}")
    features_full = features_full.drop(columns=zero_var)

print(f"Total features: {features_full.shape[1]}")
print(f"Feature matrix shape: {features_full.shape}")

full_path = os.path.join(OUT_DIR, "features_93_library.csv")
features_full.to_csv(full_path, index=False)
print(f"Saved full features to {full_path}")

print("\n" + "=" * 60)
print("Step 4: VIF-based feature selection (threshold = 10)")
print("=" * 60)

features_sel = features_full.copy()
removal_log = []

iteration = 0
while True:
    iteration += 1
    X = features_sel.values
    X_const = np.column_stack([np.ones(X.shape[0]), X])

    vifs = {}
    for i, col in enumerate(features_sel.columns):
        vif_val = variance_inflation_factor(X_const, i + 1)
        vifs[col] = vif_val

    max_feat = max(vifs, key=vifs.get)
    max_vif = vifs[max_feat]

    if max_vif <= 10:
        print(f"\nIteration {iteration}: All VIFs ≤ 10. Done.")
        break

    print(f"Iteration {iteration}: Removing '{max_feat}' (VIF = {max_vif:.2f})")
    removal_log.append((max_feat, max_vif))
    features_sel = features_sel.drop(columns=[max_feat])

print(f"\nFeatures remaining: {features_sel.shape[1]} (removed {len(removal_log)})")

X_final = features_sel.values
X_const = np.column_stack([np.ones(X_final.shape[0]), X_final])
final_vifs = {}
for i, col in enumerate(features_sel.columns):
    final_vifs[col] = variance_inflation_factor(X_const, i + 1)

print("\nFinal VIF values:")
for col in sorted(final_vifs, key=final_vifs.get, reverse=True):
    print(f"  {col}: {final_vifs[col]:.2f}")

sel_path = os.path.join(OUT_DIR, "features_vif_selected.csv")
features_sel.to_csv(sel_path, index=False)
print(f"\nSaved VIF-selected features to {sel_path}")

summary_path = os.path.join(OUT_DIR, "feature_summary.md")
kept_set = set(features_sel.columns)

with open(summary_path, "w") as f:
    f.write("# Feature Summary\n\n")
    f.write(f"Total features before VIF: {features_full.shape[1]}\n")
    f.write(f"Features after VIF selection: {features_sel.shape[1]}\n")
    f.write(f"Features removed: {len(removal_log)}\n\n")

    f.write("## Kept Features (after VIF selection)\n\n")
    f.write("| Feature | Final VIF |\n")
    f.write("|---------|----------|\n")
    for col in sorted(final_vifs, key=final_vifs.get):
        f.write(f"| {col} | {final_vifs[col]:.2f} |\n")

    f.write("\n## Removed Features (VIF > 10)\n\n")
    f.write("| Feature | VIF at Removal | Removal Order |\n")
    f.write("|---------|---------------|---------------|\n")
    for idx, (feat, vif) in enumerate(removal_log, 1):
        f.write(f"| {feat} | {vif:.2f} | {idx} |\n")

    f.write("\n## Elemental Properties Used\n\n")
    for p in sorted(valid_props.keys()):
        vals = valid_props[p]
        f.write(f"- **{p}**: Co={vals[0]:.4g}, Cr={vals[1]:.4g}, Cu={vals[2]:.4g}, Fe={vals[3]:.4g}, Ni={vals[4]:.4g}\n")

print(f"Saved feature summary to {summary_path}")
print("\nDone!")
