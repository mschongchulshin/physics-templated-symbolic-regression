import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.metrics import r2_score


PYSR_DEFAULTS = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "cube", "sqrt", "log", "exp"],
    populations=40,
    population_size=60,
    parsimony=0.005,
    turbo=True,
    bumper=True,
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    deterministic=False,
    procs=1,
    verbosity=0,
)


def make_pysr(seed, niterations=200, maxsize=25, parsimony=0.005,
              tempdir=None, **kwargs):
    params = dict(PYSR_DEFAULTS)
    params.update(dict(niterations=niterations, maxsize=maxsize,
                       parsimony=parsimony, random_state=seed))
    if tempdir is not None:
        params["temp_equation_file"] = True
        params["tempdir"] = tempdir
    params.update(kwargs)
    return PySRRegressor(**params)


def best_info(model, X_fit, y_fit):
    pred = model.predict(X_fit)
    info = {
        "train_r2": float(r2_score(y_fit, pred)),
        "train_mae": float(np.mean(np.abs(y_fit - pred))),
    }
    try:
        row = model.get_best()
        info["complexity"] = int(row["complexity"])
        info["equation_sympy"] = str(model.sympy())
        info["equation_latex"] = str(model.latex())
        info["pysr_raw_string"] = str(row["equation"])
    except Exception:
        pass
    return info


def extract_pareto(model, X_fit, y_fit):
    pareto = []
    if model.equations_ is None:
        return pareto
    for idx, row in model.equations_.iterrows():
        entry = {
            "complexity": int(row["complexity"]),
            "equation": str(row["equation"]),
            "loss": float(row["loss"]),
        }
        try:
            pred = model.predict(X_fit, index=int(idx))
            entry["train_r2"] = float(r2_score(y_fit, pred))
            entry["train_mae"] = float(np.mean(np.abs(y_fit - pred)))
        except Exception:
            pass
        pareto.append(entry)
    return pareto


def feat_mapping(cols):
    return {f"x{i}": c for i, c in enumerate(cols)}


def fit_stage1(X_comp300, y300, seed, tmpdir):
    gm = make_pysr(seed, niterations=200, maxsize=20, parsimony=0.005,
                   unary_operators=["square", "cube", "sqrt", "log"],
                   tempdir=tmpdir)
    gm.fit(X_comp300, y300)
    return gm, gm.predict(X_comp300)
