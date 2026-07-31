"""
Inverse Design using SR equations.
Given target properties, find optimal compositions using scipy.optimize.

Key advantage of SR over black-box ML:
  - Equations are differentiable → gradient-based optimization
  - Equations are transparent → constraints naturally enforced
  - Multi-objective optimization with Pareto front

Scenarios:
  1. Maximize Young's modulus at 300K
  2. Maximize UTS at 300K
  3. Maximize UTS while keeping FCC > 80%
  4. Multi-objective: maximize UTS + minimize dislocation density
  5. Scan: optimal composition at each temperature
"""

import pickle
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'Arial',
    'figure.dpi': 300,
})

RESULTS_DIR = "./sr_results_full"
INVERSE_DIR = "./inverse_design_results"
os.makedirs(INVERSE_DIR, exist_ok=True)

INPUT_NAMES = ["Co", "Cr", "Cu", "Fe", "Ni", "T"]

TARGETS = {
    "Youngs_modulus": "Young's modulus(Gpa)",
    "UTS": "UTS(Gpa)",
    "Disloc_0pct": "Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc_20pct": "Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC_0pct": "FCC 0%(%)",
    "HCP_0pct": "HCP 0%(%)",
    "BCC_0pct": "BCC 0%(%)",
    "FCC_20pct": "FCC 20%(%)",
    "HCP_20pct": "HCP 20%(%)",
    "BCC_20pct": "BCC 20%(%)",
}

# Composition constraints (from data)
COMP_BOUNDS = {
    "Co": (20, 40),
    "Cr": (15, 25),
    "Cu": (0, 20),
    "Fe": (15, 40),
    "Ni": (5, 20),
}


def load_models():
    """Load all trained SR models."""
    models = {}
    for target_key in TARGETS:
        path = os.path.join(RESULTS_DIR, f"comp_{target_key}_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[target_key] = pickle.load(f)
        else:
            print(f"  Warning: model not found for {target_key}")
    return models


def predict_property(models, target_key, composition, T):
    """
    Predict a property given composition and temperature.
    composition: dict with Co, Cr, Cu, Fe, Ni values
    """
    if target_key not in models:
        return None

    X = pd.DataFrame([{
        "Co": composition["Co"],
        "Cr": composition["Cr"],
        "Cu": composition["Cu"],
        "Fe": composition["Fe"],
        "Ni": composition["Ni"],
        "T": T,
    }])
    return models[target_key].predict(X)[0]


def composition_from_vector(x):
    """Convert optimization vector [Co, Cr, Cu, Fe] → full composition.
    Ni = 100 - sum(others) to enforce sum=100 constraint.
    """
    Co, Cr, Cu, Fe = x
    Ni = 100 - Co - Cr - Cu - Fe
    return {"Co": Co, "Cr": Cr, "Cu": Cu, "Fe": Fe, "Ni": Ni}


# ============================================================
# Scenario 1: Maximize single property at given temperature
# ============================================================
def optimize_single_property(models, target_key, T, maximize=True):
    """Find composition that maximizes/minimizes a single property."""

    def objective(x):
        comp = composition_from_vector(x)
        if comp["Ni"] < COMP_BOUNDS["Ni"][0] or comp["Ni"] > COMP_BOUNDS["Ni"][1]:
            return 1e10 if maximize else -1e10
        val = predict_property(models, target_key, comp, T)
        if val is None:
            return 1e10 if maximize else -1e10
        return -val if maximize else val

    bounds = [
        COMP_BOUNDS["Co"],
        COMP_BOUNDS["Cr"],
        COMP_BOUNDS["Cu"],
        COMP_BOUNDS["Fe"],
    ]

    # Constraint: Ni must be in valid range
    def ni_constraint_lower(x):
        return (100 - sum(x)) - COMP_BOUNDS["Ni"][0]

    def ni_constraint_upper(x):
        return COMP_BOUNDS["Ni"][1] - (100 - sum(x))

    constraints = [
        {"type": "ineq", "fun": ni_constraint_lower},
        {"type": "ineq", "fun": ni_constraint_upper},
    ]

    # Use differential evolution (global optimizer)
    # Ni = 100 - sum(x) must be in [COMP_BOUNDS["Ni"][0], COMP_BOUNDS["Ni"][1]]
    ni_lo, ni_hi = COMP_BOUNDS["Ni"]
    ni_nlc = NonlinearConstraint(lambda x: 100 - sum(x), ni_lo, ni_hi)

    result = differential_evolution(
        objective, bounds, seed=42, maxiter=1000, tol=1e-8,
        constraints=(ni_nlc,),
    )

    # Fallback: multiple random starts with L-BFGS-B
    best_result = result
    best_val = result.fun

    for seed in range(20):
        rng = np.random.RandomState(seed)
        x0 = [
            rng.uniform(*COMP_BOUNDS["Co"]),
            rng.uniform(*COMP_BOUNDS["Cr"]),
            rng.uniform(*COMP_BOUNDS["Cu"]),
            rng.uniform(*COMP_BOUNDS["Fe"]),
        ]
        # Ensure Ni in bounds
        Ni = 100 - sum(x0)
        if Ni < COMP_BOUNDS["Ni"][0] or Ni > COMP_BOUNDS["Ni"][1]:
            continue

        try:
            res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=constraints, options={"maxiter": 500})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res
        except Exception:
            pass

    comp = composition_from_vector(best_result.x)
    prop_val = predict_property(models, target_key, comp, T)

    return {
        "composition": comp,
        "temperature": T,
        "property": target_key,
        "predicted_value": float(prop_val) if prop_val else None,
        "maximize": maximize,
    }


# ============================================================
# Scenario 2: Multi-objective optimization
# ============================================================
def optimize_multi_objective(models, objectives, T, n_points=200):
    """
    Multi-objective Pareto front via weighted scalarization.
    objectives: list of (target_key, weight, maximize)
    Returns Pareto front of compositions.
    """
    pareto_points = []

    # Scan weights
    weights = np.linspace(0, 1, n_points)

    for w in weights:
        def objective(x):
            comp = composition_from_vector(x)
            Ni = comp["Ni"]
            if Ni < COMP_BOUNDS["Ni"][0] or Ni > COMP_BOUNDS["Ni"][1]:
                return 1e10

            total = 0
            for i, (target_key, _, maximize) in enumerate(objectives):
                val = predict_property(models, target_key, comp, T)
                if val is None:
                    return 1e10
                # Normalize
                sign = -1 if maximize else 1
                if i == 0:
                    total += w * sign * val
                else:
                    total += (1 - w) * sign * val
            return total

        bounds = [
            COMP_BOUNDS["Co"],
            COMP_BOUNDS["Cr"],
            COMP_BOUNDS["Cu"],
            COMP_BOUNDS["Fe"],
        ]

        constraints = [
            {"type": "ineq", "fun": lambda x: (100 - sum(x)) - COMP_BOUNDS["Ni"][0]},
            {"type": "ineq", "fun": lambda x: COMP_BOUNDS["Ni"][1] - (100 - sum(x))},
        ]

        best_val = 1e10
        best_x = None

        for seed in range(5):
            rng = np.random.RandomState(seed + int(w * 1000))
            x0 = [
                rng.uniform(*COMP_BOUNDS["Co"]),
                rng.uniform(*COMP_BOUNDS["Cr"]),
                rng.uniform(*COMP_BOUNDS["Cu"]),
                rng.uniform(*COMP_BOUNDS["Fe"]),
            ]
            Ni = 100 - sum(x0)
            if Ni < COMP_BOUNDS["Ni"][0] or Ni > COMP_BOUNDS["Ni"][1]:
                continue

            try:
                res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                               constraints=constraints)
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

        if best_x is not None:
            comp = composition_from_vector(best_x)
            props = {}
            for target_key, _, _ in objectives:
                props[target_key] = float(predict_property(models, target_key, comp, T))

            pareto_points.append({
                "weight": float(w),
                "composition": comp,
                **props,
            })

    return pareto_points


# ============================================================
# Scenario 3: Constrained optimization
# ============================================================
def optimize_constrained(models, target_key, T, maximize=True,
                         property_constraints=None):
    """
    Optimize target_key subject to constraints on other properties.
    property_constraints: list of (target_key, min_val, max_val)
    """
    if property_constraints is None:
        property_constraints = []

    def objective(x):
        comp = composition_from_vector(x)
        Ni = comp["Ni"]
        if Ni < COMP_BOUNDS["Ni"][0] or Ni > COMP_BOUNDS["Ni"][1]:
            return 1e10 if maximize else -1e10

        val = predict_property(models, target_key, comp, T)
        if val is None:
            return 1e10 if maximize else -1e10

        # Penalty for constraint violations
        penalty = 0
        for ckey, cmin, cmax in property_constraints:
            cval = predict_property(models, ckey, comp, T)
            if cval is not None:
                if cmin is not None and cval < cmin:
                    penalty += 1000 * (cmin - cval) ** 2
                if cmax is not None and cval > cmax:
                    penalty += 1000 * (cval - cmax) ** 2

        sign = -1 if maximize else 1
        return sign * val + penalty

    bounds = [
        COMP_BOUNDS["Co"],
        COMP_BOUNDS["Cr"],
        COMP_BOUNDS["Cu"],
        COMP_BOUNDS["Fe"],
    ]

    constraints = [
        {"type": "ineq", "fun": lambda x: (100 - sum(x)) - COMP_BOUNDS["Ni"][0]},
        {"type": "ineq", "fun": lambda x: COMP_BOUNDS["Ni"][1] - (100 - sum(x))},
    ]

    best_val = 1e10
    best_x = None

    for seed in range(30):
        rng = np.random.RandomState(seed)
        x0 = [
            rng.uniform(*COMP_BOUNDS["Co"]),
            rng.uniform(*COMP_BOUNDS["Cr"]),
            rng.uniform(*COMP_BOUNDS["Cu"]),
            rng.uniform(*COMP_BOUNDS["Fe"]),
        ]
        Ni = 100 - sum(x0)
        if Ni < COMP_BOUNDS["Ni"][0] or Ni > COMP_BOUNDS["Ni"][1]:
            continue

        try:
            res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=constraints)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        return None

    comp = composition_from_vector(best_x)

    # Get all predicted properties
    all_props = {}
    for tk in TARGETS:
        val = predict_property(models, tk, comp, T)
        if val is not None:
            all_props[tk] = float(val)

    return {
        "composition": comp,
        "temperature": T,
        "optimized_property": target_key,
        "all_properties": all_props,
        "constraints": [(c[0], c[1], c[2]) for c in property_constraints],
    }


# ============================================================
# Scenario 4: Temperature scan
# ============================================================
def temperature_scan(models, target_key, T_range, maximize=True):
    """Find optimal composition at each temperature."""
    results = []
    for T in T_range:
        res = optimize_single_property(models, target_key, T, maximize)
        results.append(res)
        print(f"  T={T:>6.0f}K: {target_key}={res['predicted_value']:.4f}, "
              f"comp={res['composition']}")
    return results


# ============================================================
# MD Validation candidates
# ============================================================
def generate_md_candidates(models, n_candidates=10):
    """
    Generate novel compositions for MD validation.
    Find compositions that are NOT in the training data
    but predicted to have exceptional properties.
    """
    # Load training data compositions
    dfs = []
    for sheet, temp in TEMPS.items():
        df = pd.read_excel(DATA_FILE, sheet_name=sheet)
        dfs.append(df)
    train_data = pd.concat(dfs, ignore_index=True)
    existing_comps = set(
        train_data.apply(
            lambda r: f"{r['Co(%)']:.0f}_{r['Cr(%)']:.0f}_{r['Cu(%)']:.0f}_{r['Fe(%)']:.0f}_{r['Ni(%)']:.0f}",
            axis=1
        )
    )

    candidates = []

    # Scenario A: Max Young's modulus at 300K
    for T in [80, 300, 1100]:
        res = optimize_single_property(models, "Youngs_modulus", T, maximize=True)
        comp = res["composition"]
        comp_str = f"{comp['Co']:.0f}_{comp['Cr']:.0f}_{comp['Cu']:.0f}_{comp['Fe']:.0f}_{comp['Ni']:.0f}"

        # Round to nearest 5% for practical alloy making
        comp_rounded = {k: round(v / 5) * 5 for k, v in comp.items()}
        if sum(comp_rounded.values()) != 100:
            comp_rounded["Fe"] += 100 - sum(comp_rounded.values())

        all_props = {}
        for tk in TARGETS:
            val = predict_property(models, tk, comp_rounded, T)
            if val is not None:
                all_props[tk] = float(val)

        candidates.append({
            "scenario": f"Max_Youngs_{T}K",
            "composition_optimal": comp,
            "composition_rounded": comp_rounded,
            "temperature": T,
            "predicted_properties": all_props,
            "is_novel": comp_str not in existing_comps,
        })

    # Scenario B: Max UTS at each temperature
    for T in [80, 300, 1100]:
        res = optimize_single_property(models, "UTS", T, maximize=True)
        comp = res["composition"]

        comp_rounded = {k: round(v / 5) * 5 for k, v in comp.items()}
        if sum(comp_rounded.values()) != 100:
            comp_rounded["Fe"] += 100 - sum(comp_rounded.values())

        all_props = {}
        for tk in TARGETS:
            val = predict_property(models, tk, comp_rounded, T)
            if val is not None:
                all_props[tk] = float(val)

        candidates.append({
            "scenario": f"Max_UTS_{T}K",
            "composition_rounded": comp_rounded,
            "temperature": T,
            "predicted_properties": all_props,
        })

    # Scenario C: Max UTS with FCC > 60% at 300K
    res = optimize_constrained(
        models, "UTS", 300, maximize=True,
        property_constraints=[("FCC_20pct", 60, None)],
    )
    if res:
        candidates.append({
            "scenario": "Max_UTS_FCC>60%_300K",
            "composition": res["composition"],
            "predicted_properties": res["all_properties"],
        })

    return candidates


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Loading SR models...")
    models = load_models()
    print(f"Loaded {len(models)} models")

    all_inverse = {}

    # --- Single property optimization ---
    print(f"\n{'='*60}")
    print("SCENARIO 1: Single Property Optimization")
    print(f"{'='*60}")

    for target_key in ["Youngs_modulus", "UTS"]:
        for T in [80, 300, 1100]:
            print(f"\n  Maximize {target_key} at {T}K:")
            res = optimize_single_property(models, target_key, T, maximize=True)
            print(f"    Composition: {res['composition']}")
            print(f"    Predicted: {res['predicted_value']:.4f}")
            all_inverse[f"max_{target_key}_{T}K"] = res

    # --- Multi-objective ---
    print(f"\n{'='*60}")
    print("SCENARIO 2: Multi-Objective (UTS vs Disloc density)")
    print(f"{'='*60}")

    pareto = optimize_multi_objective(
        models,
        objectives=[
            ("UTS", 1.0, True),           # maximize UTS
            ("Disloc_0pct", 1.0, False),   # minimize dislocation density
        ],
        T=300,
        n_points=50,
    )
    all_inverse["pareto_UTS_vs_Disloc_300K"] = pareto
    print(f"  Generated {len(pareto)} Pareto points")

    # --- Constrained ---
    print(f"\n{'='*60}")
    print("SCENARIO 3: Constrained Optimization")
    print(f"{'='*60}")

    print("\n  Max UTS with FCC_20% > 60% at 300K:")
    res = optimize_constrained(
        models, "UTS", 300, maximize=True,
        property_constraints=[("FCC_20pct", 60, None)],
    )
    if res:
        print(f"    Composition: {res['composition']}")
        print(f"    Properties: {res['all_properties']}")
        all_inverse["constrained_UTS_FCC60_300K"] = res

    # --- Temperature scan ---
    print(f"\n{'='*60}")
    print("SCENARIO 4: Temperature Scan")
    print(f"{'='*60}")

    T_range = np.linspace(80, 1100, 20)
    scan = temperature_scan(models, "UTS", T_range, maximize=True)
    all_inverse["temp_scan_UTS"] = scan

    # --- MD validation candidates ---
    print(f"\n{'='*60}")
    print("SCENARIO 5: MD Validation Candidates")
    print(f"{'='*60}")

    candidates = generate_md_candidates(models)
    for c in candidates:
        print(f"\n  {c['scenario']}:")
        comp = c.get("composition_rounded", c.get("composition", {}))
        print(f"    Composition: {comp}")
        if "predicted_properties" in c:
            for k, v in c["predicted_properties"].items():
                print(f"      {k}: {v:.4f}")

    all_inverse["md_candidates"] = candidates

    # --- Save ---
    with open(os.path.join(INVERSE_DIR, "inverse_design_results.json"), "w") as f:
        json.dump(all_inverse, f, indent=2, default=str)

    # --- Plot multi-objective Pareto ---
    if pareto:
        fig, ax = plt.subplots(figsize=(8, 6))
        uts_vals = [p["UTS"] for p in pareto]
        disloc_vals = [p["Disloc_0pct"] for p in pareto]

        ax.scatter(uts_vals, disloc_vals, c="steelblue", s=30, edgecolors="white")
        ax.set_xlabel("UTS (GPa)")
        ax.set_ylabel("Dislocation Density (×10¹⁷ m⁻²)")
        ax.set_title("Multi-Objective Pareto Front\n(UTS ↑ vs Dislocation Density ↓) at 300K")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(INVERSE_DIR, "pareto_UTS_vs_Disloc.png"), dpi=300)
        plt.savefig(os.path.join(INVERSE_DIR, "pareto_UTS_vs_Disloc.pdf"))
        print("\nPareto front plot saved")
        plt.close()

    print(f"\nAll results saved to {INVERSE_DIR}/")
