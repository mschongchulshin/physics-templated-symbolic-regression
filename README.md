<p align="center">
  <img src="docs_banner.png" alt="Left, the PT-SR workflow: a genetic-programming search, the two-stage
  template competition, and the shared forward and inverse use of the elected
  equation. Right, the selected equation for each of the twelve targets,
  plotted over composition at 80, 300 and 1100 K." width="100%">
</p>

# Physics-Templated Symbolic Regression for High-Entropy Alloys

Code and data for **"Physics-Templated Symbolic Regression discovers closed-form, differentiable equations for high-entropy alloy properties"**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22784255.svg)](https://doi.org/10.5281/zenodo.22784255)

## Abstract

Black-box machine-learning models predict materials properties accurately, but
they return models that domain researchers cannot interpret. SHAP attribution
ranks inputs without revealing how they act. Here we introduce
physics-templated symbolic regression (PT-SR), where candidate physical laws
compete as templates and the Bayesian information criterion elects one
closed-form equation per property. On high-entropy alloys, PT-SR matches
state-of-the-art black-box models on key mechanical properties and the dominant
FCC phase, with four parameters against up to 4.4 × 10⁵. The equations expose
what SHAP cannot, such as element coefficient ratios, the channel of each
element and an element acting only through temperature. Under
leave-one-dataset-out cross-validation on published experimental data, PT-SR
remains equivalent to state-of-the-art models. Molecular-dynamics simulations
confirm that the differentiable equations enable inverse alloy design. By
bridging accuracy with interpretability, PT-SR establishes symbolic regression
as a practical route to governing laws wherever candidate physics exists and
the response is approximately separable.

## Layout

```
data/          the corpus, its derived features, the experimental measurements
               behind the leave-one-dataset-out folds, and the two workbooks
               submitted with the paper
equations/     all 660 discovered equations, sympy-parseable
src/           the two-stage search, the nine templates, the ML and DL baselines
baselines/     the published models re-trained on this corpus
lodo/          leave-one-dataset-out over eighteen folds
inverse_design/  optimisation of the closed forms and the screening benchmark
analysis/      SHAP attribution, the equivalence tests, hyperparameter sweeps,
               learning curves, supplementary figures
results/       cross-validation results and the selected hyperparameters
```

## Install

```bash
git clone https://github.com/mschongchulshin/physics-templated-symbolic-regression.git
cd physics-templated-symbolic-regression
pip install -r requirements.txt
```

PySR runs on a Julia backend, which it installs on first use:

```python
import pysr; pysr.install()
```

## Reproducing the figures

Unless noted, each command writes under `results/`.

| Figure | Command |
|---|---|
| Fig. 2a, 2b | `python src/run_sr_template.py 0 5 && python src/compile_results.py` |
| Fig. 3 | the same run, which writes the elected equation per target to `equations/` |
| Fig. 4 | `python analysis/shap_attribution.py && python analysis/shap_beeswarm.py` |
| Fig. 5 | `python lodo/make_lodo_folds.py && python lodo/make_lodo_folds_all.py`, then `python lodo/ptsr_main_lodo.py`, then `python lodo/baselines_lodo.py run 4` |
| Fig. 6 | `python inverse_design/run_inverse_design.py` |
| Supp. Figs. 1, 2 | `python analysis/render_supp_figs.py` |
| Supp. Figs. 3-5 | `python analysis/sr_sensitivity_worker.py <worker_id> <n_workers>` then `python analysis/sr_sensitivity_plot.py` |
| Supp. Fig. 6 | `python src/run_shuffled_y.py 0 5` |
| Supp. Fig. 7 | `python inverse_design/analysis_roc_interaction.py` |
| Supp. Note 3 | `python analysis/equivalence_tost.py` |
| Supp. Table 13 | `python analysis/hume_rothery_ols.py` |
| Optuna traces | `python analysis/optuna_convergence.py` |

The Fig. 5 chain runs per property. The first command cuts the yield-strength
folds and the second cuts the folds for the other five properties.
`baselines_lodo.py` reads the yield-strength folds by default, and
`LODO_PROP=UTS`, `elongation`, `HV` or `modulus` points it at one of the
others. Density yields no usable fold and has no run.

Scripts that produce the deposited inputs rather than a figure:

| Produces | Command |
|---|---|
| `data/features_13.csv`, `data/features_93_library.csv`, `data/features_vif50.csv` | `python src/feature_engineering.py` |
| `results/cv_results/ml_baselines_optuna.csv` | `python src/run_ml_baselines.py` then `python src/run_cv_ml.py` |
| `results/cv_results/dl_models_optuna.csv` | `python src/run_deep_models.py` then `python src/run_cv_dl.py` |
| `results/cv_results/gpr.csv` | `python src/run_gpr.py` |
| `results/cv_results/sr_freeform.csv` | `python src/run_sr_freeform.py 0 5` |
| `results/cv_results/sr_6feat_control.csv` | `python src/run_6feat_control.py 0 5` |
| learning curves over training-set fraction | `python analysis/lc_worker.py <worker_id> <n_workers>` |
| the alloy-system pools the LOCO benchmark reads | `python lodo/make_property_pools.py` |

The full symbolic-regression sweep is 2,700 runs and takes days on a laptop.
It does not have to be rerun to check the paper: the finished sweep is in
`results/cv_results/sr_9templates_raw.csv`, the equations are in `equations/`,
and `src/compile_results.py` reads them to reproduce every published number.

Values quoted in the paper are rounded from these files. The test R2 of 0.994
for UTS, for example, is the mean of the 25 cross-validation runs of the
elected Arrhenius template in `results/cv_results/sr_9templates_raw.csv`,
which comes to 0.9936. The equivalence tests are tabulated in `data/Source_Data.xlsx`, sheet
`Supp_Note_3_TOST`, and `analysis/equivalence_tost.py` recomputes them from
the deposited runs, reproducing all twelve comparators, p values and
verdicts.

Rerunning the search itself is a different matter. PySR is seeded
(`random_state=seed`) but runs with `deterministic=False` and `procs=1`, since
its strict determinism mode requires single-threaded evolution. A rerun
therefore recovers the same operator structure at the rate reported in the
paper rather than byte-identical expressions. Set `procs=0,
multithreading=False, deterministic=True` in `src/run_sr_template.py` for an
exact replay, at a large cost in wall time.

## Data

### Submitted with the paper

| File | What it holds |
|---|---|
| `data/Source_Data.xlsx` | the numbers behind every figure panel and supplementary table |
| `data/Supplementary_Data_1_MD_corpus.xlsx` | the 684-point molecular-dynamics training corpus |
| `data/Supplementary_Data_2_LODO_dataset.xlsx` | the 390 experimental rows the eighteen folds are built from |

### What the code reads

| File | What it holds |
|---|---|
| `data/CoCrCuFeNi_684.csv` | the corpus as the scripts read it. Supplementary Data 1 is the 228 compositions that were trained on; this adds the four held out for verification |
| `data/features_13.csv` | the thirteen model inputs, per sample |
| `data/features_93_library.csv` | the feature library before VIF reduction, per sample. Source Data holds its correlation matrix, not the values |
| `data/features_vif50.csv` | what survived VIF < 50, per sample, before the two removed on physical grounds |
| `data/LODO_experimental_dataset.csv` | the full published alloy table, third party, see below. Supplementary Data 2 is the subset the folds use |
| `lodo/verified_V.pkl` | the 138 yield-strength rows checked against the original papers |
| `lodo/verified_M.pkl` | which of those rows is train and which is test, per fold |
| `lodo/verified_F.pkl` | the candidate yield-strength folds, with the reason each one was kept or dropped |
| `equations/all_equations_660.json` | every discovered equation with its training fit and Pareto front |
| `results/cv_results/ptsr_best5_seeds.csv` | the five best-scoring seeds per target out of thirty, which the equivalence test compares |

The molecular-dynamics trajectories and the LAMMPS input scripts that produced
them are not part of this deposit. The corpus above is what the symbolic
regression consumes, and it is complete. Simulations used the Farkas-Caro EAM
potential, available from
<https://www.ctcms.nist.gov/potentials/entry/2018--Farkas-D-Caro-A--Fe-Ni-Cr-Co-Cu/>.

## Third-party data

`data/LODO_experimental_dataset.csv` is redistributed from Borg et al.,
*Scientific Data* **7**, 430 (2020), https://doi.org/10.1038/s41597-020-00768-9,
unmodified and under its CC BY 4.0 licence, with the credit that licence
requires. See `data/LODO_experimental_dataset_SOURCE.md`. Everything else in
`data/` is ours.

## License

Code is MIT, see [LICENSE](LICENSE).

The data and equations we generated, everything under `data/`, `equations/`
and `results/`, are released under CC BY 4.0. The one exception is
`data/LODO_experimental_dataset.csv`, which is third-party and carries its own
terms, recorded in `data/LODO_experimental_dataset_SOURCE.md`.
