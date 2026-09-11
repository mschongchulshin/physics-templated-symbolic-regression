<p align="center">
  <img src="docs_banner.png" alt="Left, the PT-SR workflow: a genetic-programming search, the two-stage
  template competition, and the shared forward and inverse use of the elected
  equation. Right, the selected equation for each of the twelve targets,
  plotted over composition at 80, 300 and 1100 K." width="100%">
</p>

# Physics-Templated Symbolic Regression for High-Entropy Alloys

Code and data for **"Physics-Templated Symbolic Regression discovers closed-form, differentiable equations for high-entropy alloy properties"**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

High-entropy alloys open a composition space too large to search by experiment,
and black-box machine-learning surrogates now predict their properties with R²
above 0.99. A surrogate returns a number and not an equation, so its prediction
cannot be interpreted, inspected against physical law, or differentiated for
design. Here we constrain a symbolic-regression search with nine competing
physical templates and let a model-selection criterion elect the winner for each
property. Trained on a 684-point molecular-dynamics corpus of CoCrCuFeNi, our
method returns one closed-form equation per target property. The equation for
ultimate tensile strength reaches a test R² of 0.994 with only four parameters,
replacing a surrogate that reaches the same accuracy with 4.4 × 10⁵. Used as a
design objective, the equation extrapolates beyond the training envelope to
compositions whose measured strength exceeds the highest and falls below the
lowest in the corpus. The same procedure holds on independent experimental
alloys sharing no elements with the corpus, reaching R² above 0.97 on folds of
only four measurements, where black-box models fall below zero. The approach can
be extended to any system where candidate physical laws are available but the
form that governs the measured response is unknown.

## Layout

```
data/          the corpus, its derived features, the experimental measurements
               behind the cross-source folds, and the two workbooks submitted
               with the paper
equations/     all 660 discovered equations, sympy-parseable
src/           the two-stage search, the nine templates, the ML and DL baselines
baselines/     the published models re-trained on this corpus
lodo/          leave-one-dataset-out over six cross-source folds
inverse_design/  optimisation of the closed forms and the screening benchmark
analysis/      SHAP attribution, hyperparameter sweeps, learning curves,
               supplementary figures
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

Each command writes to `results/generated/`.

| Figure | Command |
|---|---|
| Fig. 2a, 2b | `python src/run_sr_template.py && python src/compile_results.py` |
| Fig. 2 baselines | `python baselines/lodo_comparison/run.py` |
| Fig. 3 | `python src/run_sr_template.py`, which writes the elected equation per target |
| Fig. 4 | `python analysis/shap_attribution.py && python analysis/shap_beeswarm.py` |
| Fig. 5 | `python lodo/build_folds.py && python lodo/run_ptsr.py && python lodo/run_baselines.py` |
| Fig. 6a-h | `python inverse_design/run_inverse_design.py` |
| Fig. 6i | `python inverse_design/analysis_roc_interaction.py` |
| Supp. Figs. 1, 2 | `python analysis/render_supp_figs.py` |
| Supp. Figs. 3-5 | `python analysis/sr_sensitivity_worker.py <worker_id> <n_workers>` then `python analysis/sr_sensitivity_plot.py` |
| Supp. Figs. 6, 7 | `python analysis/optuna_convergence.py` |
| Supp. Fig. 8 | `python src/run_shuffled_y.py` |

The full symbolic-regression sweep is 2,700 runs and takes days on a laptop.
`results/cv_results/sr_9templates_raw.csv` holds the finished sweep, and
`src/compile_results.py` reads it, so the published numbers can be checked
without rerunning the search.

## Reproducibility

The cross-validation numbers, the equations and every table in the paper come
from the files deposited under `results/` and `equations/`, and rerunning the
compile step reproduces them exactly.

Rerunning the symbolic search itself is a different matter. PySR is seeded
(`random_state=seed`) but runs with `deterministic=False` and `procs=1`, since
its strict determinism mode requires single-threaded evolution. A rerun
therefore recovers the same operator structure at the rate reported in the
paper rather than byte-identical expressions. Set `procs=0,
multithreading=False, deterministic=True` in `src/run_sr_template.py` for an
exact replay, at a large cost in wall time.

## Data

| File | What it holds |
|---|---|
| `data/CoCrCuFeNi_684.csv` | the corpus the code reads, 232 compositions x 3 temperatures. Supplementary Data 1 is the 228 that were trained on; this adds the four held out for verification |
| `data/compositions_228.csv` | the composition grid on its own |
| `data/features_13.csv` | the thirteen model inputs, per sample |
| `data/features_93_library.csv` | the feature library before VIF reduction, per sample. Source Data holds its correlation matrix, not the values |
| `data/features_vif50.csv` | what survived VIF < 50, per sample, before the two removed on physical grounds |
| `data/LODO_experimental_dataset.csv` | the experimental measurements the cross-source folds are cut from. Source Data holds the fold results, not the underlying table |
| `data/Supplementary_Data_1_MD_corpus.xlsx` | the training corpus as submitted with the paper |
| `data/Source_Data.xlsx` | the numbers behind every figure panel and supplementary table |
| `equations/all_equations_660.json` | every discovered equation with its training fit and Pareto front |

The molecular-dynamics trajectories and the LAMMPS input scripts that produced
them are not part of this deposit. The corpus above is what the symbolic
regression consumes, and it is complete. Simulations used the Farkas-Caro EAM
potential, available from
<https://www.ctcms.nist.gov/potentials/entry/2018--Farkas-D-Caro-A--Fe-Ni-Cr-Co-Cu/>.

## Third-party data

`data/LODO_experimental_dataset.csv` is redistributed from Borg et al.,
*Scientific Data* **7**, 430 (2020), https://doi.org/10.1038/s41597-020-00768-9,
unmodified. See `data/LODO_experimental_dataset_SOURCE.md`. Everything else in `data/` is
ours.

## Citation

Cite the paper and, separately, the archived release of this code.

```bibtex
@article{ptsr2026,
  title   = {Physics-Templated Symbolic Regression discovers closed-form,
             differentiable equations for high-entropy alloy properties},
  author  = {Shin, Hongchul and Moon, Chanhyuk and Jo, Hyeonjin and
             Hwang, Kwang Yeon and Cheong, Jun Young and Jang, Hyo-Sun and
             Yoon, Taeyoung},
  year    = {2026}
}

@software{ptsr_code,
  title     = {physics-templated-symbolic-regression},
  author    = {Shin, Hongchul and Moon, Chanhyuk and Jo, Hyeonjin and
               Hwang, Kwang Yeon and Cheong, Jun Young and Jang, Hyo-Sun and
               Yoon, Taeyoung},
  year      = {2026}
}
```

## License

Code is MIT, see [LICENSE](LICENSE).

The data and equations we generated, everything under `data/`, `equations/`
and `results/`, are released under CC BY 4.0. The one exception is
`data/LODO_experimental_dataset.csv`, which is third-party and carries its own
terms, recorded in `data/LODO_experimental_dataset_SOURCE.md`.
