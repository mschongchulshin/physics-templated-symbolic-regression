<p align="center">
  <img src="docs_banner.png" alt="Fitted parameters against test accuracy for
  ten methods over twelve targets" width="100%">
</p>

# Physics-Templated Symbolic Regression for High-Entropy Alloys

Code and data for **"Physics-Templated Symbolic Regression discovers closed-form, differentiable equations for high-entropy alloy properties"**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What this is

Symbolic regression usually searches an unconstrained space of expressions. Here
the search is split in two and each half is given a physical form to fill in.
Stage 1 fits a composition function g(x) at a single temperature. Stage 2 fits a
temperature function f(x, T) to the residual ratios, choosing among nine
physics-motivated templates, and BIC picks the winner.

The result is one closed-form equation per target: four fitted coefficients
against the 28,065 to 436,147 of the published surrogates it is benchmarked
against, at accuracy that is statistically equivalent on the majority of the
twelve targets. Because the output is an equation rather than a fitted object,
it can be differentiated analytically, which is what makes the inverse design in
the paper possible.

The system is CoCrCuFeNi, 232 compositions at 80, 300 and 1100 K, with twelve
mechanical and microstructural targets from molecular dynamics.

## Layout

```
data/          the 684-point corpus, the 13 model inputs, the 232 compositions
equations/     all 660 discovered equations, sympy-parseable
src/           the two-stage search, the nine templates, the ML and DL baselines
baselines/     the published models re-trained on this corpus
lodo/          leave-one-dataset-out over six cross-source folds
inverse_design/  optimisation of the closed forms and the screening benchmark
analysis/      hyperparameter sweeps, learning curves, supplementary figures
results/       cross-validation results and the selected hyperparameters
```

## Install

```bash
git clone https://github.com/<user>/physics-template-SR-HEA.git
cd physics-template-SR-HEA
pip install -r requirements.txt
```

PySR runs on a Julia backend, which it installs on first use:

```python
import pysr; pysr.install()
```

## Reproducing the figures

Each command writes to `results/generated/`.

| Figure | Command |
|---|---|
| Fig. 1c, 1d | `python src/run_sr_template.py && python src/compile_results.py` |
| Fig. 1c baselines | `python baselines/lodo_comparison_run.py` |
| Fig. 3 | `python lodo/build_folds.py && python lodo/run_ptsr.py && python lodo/run_baselines.py` |
| Fig. 4 | `python inverse_design/run_inverse_design.py` |
| Fig. 4d | `python inverse_design/analysis_roc_interaction.py` |
| Supp. Figs. 1, 2 | `python analysis/render_supp_figs.py` |
| Supp. Figs. 3-5 | `python analysis/sr_sensitivity_worker.py && python analysis/sr_sensitivity_plot.py` |
| Supp. Figs. 6, 7 | `python analysis/optuna_convergence.py` |
| Supp. Fig. 8 | `python src/run_shuffled_y.py` |

The full symbolic-regression sweep is 2,700 runs and takes days on a laptop.
`results/cv_results/sr_9templates_raw.csv` holds the finished sweep, and
`src/compile_results.py` reads it, so the published numbers can be checked
without rerunning the search.

## Data

| File | What it holds |
|---|---|
| `data/raw/CoCrCuFeNi_684.csv` | 232 compositions x 3 temperatures, all measured targets |
| `data/features/features_13.csv` | the thirteen model inputs |
| `data/features/features_full.csv` | the 93-feature library before VIF reduction |
| `data/compositions/compositions_228.csv` | the unique compositions |
| `equations/all_equations_660.json` | every discovered equation with its training fit and Pareto front |

Molecular-dynamics trajectories are not deposited here for size. They were
generated in LAMMPS with the Farkas-Caro EAM potential, available from
<https://www.ctcms.nist.gov/potentials/entry/2018--Farkas-D-Caro-A--Fe-Ni-Cr-Co-Cu/>.

## Citation

Add once the DOI is issued.

## License

MIT. See [LICENSE](LICENSE).
