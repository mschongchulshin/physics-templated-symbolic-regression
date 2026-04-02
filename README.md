# Physics-Template Symbolic Regression for High-Entropy Alloys

Code and data for: **"[논문 제목]"** published in *Nature Communications* (2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements **physics-template competition for symbolic regression (SR)**, applied to discover composition–temperature governing laws in CoCrCuFeNi high-entropy alloys (HEAs). Nine physics-motivated templates (Arrhenius, thermal softening, CALPHAD-inspired, etc.) compete via SR to find the best-fitting analytical expressions for 12 mechanical and microstructural targets across 696 molecular dynamics simulations.

## Repository Structure

```
physics-template-SR-HEA/
├── data/
│   ├── raw/CoCrCuFeNi_684.csv      # Full MD dataset (696 rows × 42 columns)
│   ├── features/features_13.csv    # VIF-selected 13 elemental features
│   └── compositions/               # 232 unique compositions
├── equations/
│   └── all_equations_660.json      # All 660 discovered equations (sympy-parseable)
├── src/
│   ├── templates/                  # 9 physics templates (__init__.py)
│   ├── run_sr_template.py          # Full-fit SR per template (9 templates × 12 targets × 5 seeds)
│   ├── run_sr_freeform.py          # Free-form SR baseline
│   ├── run_6feat_control.py        # 6-feature descriptor-free control
│   ├── run_ml_baselines.py         # ML baselines with Optuna (50 trials)
│   ├── run_deep_models.py          # DL models with Optuna (50 trials)
│   ├── run_cv_ml.py                # 5-fold × 5-seed CV for ML (Optuna best HP)
│   ├── run_cv_dl.py                # 5-fold × 5-seed CV for DL (Optuna best HP)
│   ├── run_gpr.py                  # Gaussian Process Regression baseline
│   ├── run_shuffled_y.py           # Permutation test (shuffled-y control)
│   ├── feature_engineering.py      # VIF-based feature selection (93 → 13)
│   └── compile_results.py          # Compile all results into Excel
├── analysis/
│   ├── inverse_design.py           # Inverse design via DE + L-BFGS
│   ├── lc_worker.py                # Learning curve workers (4 fractions × 11 templates × 12 targets)
│   ├── sr_sensitivity_worker.py    # SR hyperparameter sensitivity (parsimony λ + niterations sweep)
│   ├── sr_sensitivity_plot.py      # Plot S_parsimony + S_niterations figures
│   ├── render_supp_figs.py         # S1a (93-feat corr), S1b (VIF-50 corr), S2/S3 Optuna convergence
│   └── optuna_convergence.py       # Optuna TPE convergence for ML (50 trials) + DL (30 trials)
├── results/
│   ├── cv_results/                 # CV R² (25 values per model × target)
│   └── optuna/                     # Optuna best hyperparameters (JSON)
├── md_simulations/                 # LAMMPS input scripts
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/[username]/physics-template-SR-HEA.git
cd physics-template-SR-HEA
pip install -r requirements.txt
```

**Note:** PySR requires Julia. Install Julia first:
```bash
# Install Julia (https://julialang.org/downloads/)
# Then install PySR which auto-installs Julia packages:
pip install pysr
python -c "import pysr; pysr.install()"
```

## Requirements

- Python ≥ 3.12
- PySR 1.5.9 (requires Julia ≥ 1.9)
- scikit-learn 1.8.0
- PyTorch 2.6.0
- XGBoost 3.2.0
- Optuna 3.6.1
- mendeleev 1.1.0

## Quick Start

### Run physics-template SR (full-data fit, 5 seeds)
```bash
# Single template, all targets, seeds 0-4
python src/run_sr_template.py 0 5 arrhenius

# All 9 templates in parallel (run separately for each template)
for tmpl in arrhenius thermal_softening calphad additive multiplicative \
            power_law entropy_weighted free_2stage free_1stage; do
    python src/run_sr_template.py 0 5 $tmpl &
done
```

### Run ML baselines with Optuna HP optimization
```bash
python src/run_ml_baselines.py   # → results/optuna/best_params_ml.json
python src/run_cv_ml.py          # → results/cv_results/ml_baselines_optuna.csv
```

### Run DL models with Optuna
```bash
python src/run_deep_models.py    # → results/optuna/best_params_dl.json
python src/run_cv_dl.py          # → results/cv_results/dl_models_optuna.csv
```

### Compile all results
```bash
python src/compile_results.py    # → final_results.xlsx
```

### Supplementary analysis (sensitivity + learning curve)
```bash
# SR hyperparameter sensitivity (parsimony λ sweep + niterations sweep)
# Run 5 workers in parallel (each takes ~6-8 hours)
for i in 0 1 2 3 4; do
    python analysis/sr_sensitivity_worker.py $i 5 > /tmp/sr_sens_w${i}.log 2>&1 &
done
# After all workers finish:
python analysis/sr_sensitivity_plot.py    # → S_parsimony.svg/png, S_niterations.svg/png

# Learning curve (4 fractions × 11 templates × 12 targets, fixed val set)
for i in 0 1 2 3 4; do
    python analysis/lc_worker.py $i 5 > /tmp/lc_w${i}.log 2>&1 &
done

# Optuna TPE convergence
python analysis/optuna_convergence.py

# Feature correlation heatmaps (S1a, S1b)
python analysis/render_supp_figs.py
```

## Data

| File | Description | Shape |
|------|-------------|-------|
| `data/raw/CoCrCuFeNi_684.csv` | Full MD dataset | 696 × 42 |
| `data/features/features_13.csv` | VIF-selected elemental features | 696 × 8 |
| `data/compositions/compositions_228.csv` | Unique compositions | 232 × 5 |

**Targets (12):** Young's modulus, UTS, dislocation density (0%/20% strain), FCC/HCP/BCC/Other fraction (0%/20% strain)

## Equations

`equations/all_equations_660.json` contains all 660 symbolic expressions (9 templates × 12 targets × 5 seeds + freeform + 6-feat control), each with:
- `full_equation`: sympy-parseable string with feature names (Co, Cr, T, etc.)
- `train_r2`: training R²
- `complexity`: equation complexity score
- `feature_mapping`: variable name → feature name mapping
- `pareto_front`: full Pareto front (complexity vs. loss)

## MD Simulations

LAMMPS input scripts and Atomsk polycrystal generation scripts are provided in `md_simulations/`. The EAM interatomic potential (Farkas–Caro) can be downloaded from:
> https://www.ctcms.nist.gov/potentials/entry/2018--Farkas-D-Caro-A--Fe-Ni-Cr-Co-Cu/

## Citation

```bibtex
@article{[author]2026physics,
  title={Physics-template symbolic regression as a computational
         characterization tool for high-entropy alloys},
  author={[Authors]},
  journal={Nature Communications},
  volume={},
  pages={},
  year={2026},
  doi={}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
