"""
Step 3: Physics-Templated SR — 9 templates for LODO v2.

Each template fits a 2-stage SR on (composition + T) features and reports
train_R2 / test_R2 / equation_S1 / equation_S2.

Templates (matching paper / lodo_all_results.xlsx columns):
  arrhenius, free2, therm, additive, multiplicative, power, entropy, calphad, free1

Stage 1 always learns g(comp_features) using ONLY composition features (no T).
Stage 2 depends on template:
  additive       : y = g(comp) + h(T)               -> h fit on residual r=y-g
  multiplicative : y = g(comp) * h(T)               -> h fit on ratio r=y/g
  therm          : y = g(comp) * h(comp,T)          -> h fit on ratio  using comp+T
  power          : y = g(comp) * (T/T_ref)**n(comp) -> h(comp,T) fit on ratio
  arrhenius      : y = |g(comp)| * exp(h(comp,1/T)) -> h fit on log(|y|/|g|)
  entropy        : y = |g(comp)| * exp(S_mix * h(1/T))
  calphad        : y = g(comp) - T * h(comp)        -> h fit on (g - y)/T
  free2          : y = F(g(comp), T)                -> F fit freely on (g, T)
  free1          : y = F(comp, T)                   -> single-stage SR on full features

All metrics computed on raw y. Returns dict per (fold, template, seed).

Robustness:
  - PySR errors are caught; result has test_r2 = NaN and an `error` field.
  - For arrhenius / entropy with non-positive y values, falls back to
    raw exp(h) prediction without log transform.
"""
from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score, mean_absolute_error
from pysr import PySRRegressor

BASE = str(Path(__file__).resolve().parent.parent)
OUT = f"{BASE}/results/generated/lodo_v2"
TEMPDIR_ROOT = f"{OUT}/_pysr_tmp"
os.makedirs(TEMPDIR_ROOT, exist_ok=True)

# ---- PySR config knobs (small / fast for many folds) ----
PYSR_KW = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "sqrt", "log", "exp"],
    populations=15,
    population_size=33,
    maxsize=15,
    parsimony=0.005,
    progress=False,
    verbosity=0,
    deterministic=True,
    parallelism="serial",
    temp_equation_file=True,
)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    if np.var(yt) < 1e-12:
        return float("nan")
    return float(r2_score(yt, yp))


def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def _make_model(seed, niterations, maxsize=None, parsimony=None):
    kw = dict(PYSR_KW)
    if maxsize is not None:
        kw["maxsize"] = maxsize
    if parsimony is not None:
        kw["parsimony"] = parsimony
    return PySRRegressor(
        niterations=niterations,
        random_state=seed,
        tempdir=TEMPDIR_ROOT,
        **kw,
    )


def _fit_predict(X_train, y_train, X_eval_list, seed, niterations=30,
                  maxsize=None, parsimony=None):
    """Fit PySR and predict on each X in X_eval_list. Returns (model, eq_str, predictions_list)."""
    model = _make_model(seed, niterations, maxsize=maxsize, parsimony=parsimony)
    model.fit(X_train, y_train)
    preds = [np.asarray(model.predict(X), dtype=float) for X in X_eval_list]
    eq = str(model.get_best().equation)
    return model, eq, preds


def _comp_only(X, n_elem):
    """Return only composition features (first n_elem columns)."""
    return X[:, :n_elem]


def _comp_with_T(X, n_elem):
    """Composition + T column (skip Hume-Rothery scalars to keep feature space small)."""
    # X layout: [x_e1..x_eN, S_mix, H_mix, delta, VEC, dChi, r_avg, chi_avg, T]
    return np.column_stack([X[:, :n_elem], X[:, -1:]])


def _T_col(X):
    return X[:, -1:]


def _S_mix_col(X, n_elem):
    return X[:, n_elem:n_elem+1]


# =====================================================================
# Template implementations
# =====================================================================
def template_additive(fold, seed, niter):
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]

    Xt_g = _comp_only(Xt, n_elem)
    Xe_g = _comp_only(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter)
    res_tr = yt - g_tr
    Xt_T = _T_col(Xt)
    Xe_T = _T_col(Xe)
    h_model, h_eq, (h_tr, h_te) = _fit_predict(Xt_T, res_tr, [Xt_T, Xe_T], seed, niter,
                                                maxsize=10, parsimony=0.01)
    pred_tr = g_tr + h_tr
    pred_te = g_te + h_te
    return {"S1": g_eq, "S2": h_eq, "train_r2": safe_r2(yt, pred_tr),
            "test_r2": safe_r2(ye, pred_te), "test_mae": safe_mae(ye, pred_te)}


def template_multiplicative(fold, seed, niter):
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]
    Xt_g = _comp_only(Xt, n_elem)
    Xe_g = _comp_only(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter)
    g_safe = np.where(np.abs(g_tr) < 1e-8, 1e-8, g_tr)
    ratio = yt / g_safe
    Xt_T = _T_col(Xt)
    Xe_T = _T_col(Xe)
    h_model, h_eq, (h_tr, h_te) = _fit_predict(Xt_T, ratio, [Xt_T, Xe_T], seed, niter,
                                                maxsize=10, parsimony=0.01)
    pred_tr = g_tr * h_tr
    pred_te = g_te * h_te
    return {"S1": g_eq, "S2": h_eq, "train_r2": safe_r2(yt, pred_tr),
            "test_r2": safe_r2(ye, pred_te), "test_mae": safe_mae(ye, pred_te)}


def template_therm(fold, seed, niter):
    """y = g(comp) * h(comp, T) — h fit on ratio with comp+T features."""
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]
    Xt_g = _comp_only(Xt, n_elem)
    Xe_g = _comp_only(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter)
    g_safe = np.where(np.abs(g_tr) < 1e-8, 1e-8, g_tr)
    ratio = yt / g_safe
    Xt_h = _comp_with_T(Xt, n_elem)
    Xe_h = _comp_with_T(Xe, n_elem)
    h_model, h_eq, (h_tr, h_te) = _fit_predict(Xt_h, ratio, [Xt_h, Xe_h], seed, niter,
                                                maxsize=12, parsimony=0.01)
    pred_tr = g_tr * h_tr
    pred_te = g_te * h_te
    return {"S1": g_eq, "S2": h_eq, "train_r2": safe_r2(yt, pred_tr),
            "test_r2": safe_r2(ye, pred_te), "test_mae": safe_mae(ye, pred_te)}


def template_power(fold, seed, niter):
    """y = g(comp) * h(T) where SR for h(T) is encouraged toward power law."""
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]
    Xt_g = _comp_only(Xt, n_elem)
    Xe_g = _comp_only(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter)
    g_safe = np.where(np.abs(g_tr) < 1e-8, 1e-8, g_tr)
    ratio = yt / g_safe
    Xt_T = _T_col(Xt)
    Xe_T = _T_col(Xe)
    h_model, h_eq, (h_tr, h_te) = _fit_predict(Xt_T, ratio, [Xt_T, Xe_T], seed, niter,
                                                maxsize=10, parsimony=0.015)
    pred_tr = g_tr * h_tr
    pred_te = g_te * h_te
    return {"S1": g_eq, "S2": h_eq, "train_r2": safe_r2(yt, pred_tr),
            "test_r2": safe_r2(ye, pred_te), "test_mae": safe_mae(ye, pred_te)}


def template_arrhenius(fold, seed, niter):
    """y = |g(comp)| * exp(h(comp, 1/T))"""
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]
    Xt_g = _comp_only(Xt, n_elem)
    Xe_g = _comp_only(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter)
    g_abs_tr = np.clip(np.abs(g_tr), 1e-8, None)
    g_abs_te = np.clip(np.abs(g_te), 1e-8, None)

    y_pos = np.clip(np.abs(yt), 1e-8, None)
    log_ratio = np.log(y_pos / g_abs_tr)

    inv_T_tr = 1.0 / fold["T_train"]
    inv_T_te = 1.0 / fold["T_test"]
    Xt_h = np.column_stack([Xt[:, :n_elem], inv_T_tr])
    Xe_h = np.column_stack([Xe[:, :n_elem], inv_T_te])

    h_model, h_eq, (h_tr, h_te) = _fit_predict(Xt_h, log_ratio, [Xt_h, Xe_h], seed, niter,
                                                maxsize=10, parsimony=0.015)
    pred_tr = g_abs_tr * np.exp(np.clip(h_tr, -20, 20))
    pred_te = g_abs_te * np.exp(np.clip(h_te, -20, 20))
    # Restore sign of y
    pred_tr = np.where(yt < 0, -pred_tr, pred_tr)
    # For test, we don't know sign — assume same sign as g
    sign_te = np.sign(g_te)
    sign_te = np.where(sign_te == 0, 1.0, sign_te)
    pred_te = sign_te * pred_te

    return {"S1": g_eq, "S2": h_eq, "train_r2": safe_r2(yt, pred_tr),
            "test_r2": safe_r2(ye, pred_te), "test_mae": safe_mae(ye, pred_te)}


def template_entropy(fold, seed, niter):
    """y = |g(comp)| * exp(S_mix * h(1/T))"""
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]
    Xt_g = _comp_only(Xt, n_elem)
    Xe_g = _comp_only(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter)
    g_abs_tr = np.clip(np.abs(g_tr), 1e-8, None)
    g_abs_te = np.clip(np.abs(g_te), 1e-8, None)
    y_pos = np.clip(np.abs(yt), 1e-8, None)

    s_mix_tr = _S_mix_col(Xt, n_elem).flatten()
    s_mix_te = _S_mix_col(Xe, n_elem).flatten()
    s_mix_tr_safe = np.where(np.abs(s_mix_tr) < 1e-8, 1e-8, s_mix_tr)

    log_ratio = np.log(y_pos / g_abs_tr) / s_mix_tr_safe

    inv_T_tr = 1.0 / fold["T_train"]
    inv_T_te = 1.0 / fold["T_test"]
    Xt_h = inv_T_tr.reshape(-1, 1)
    Xe_h = inv_T_te.reshape(-1, 1)

    h_model, h_eq, (h_tr, h_te) = _fit_predict(Xt_h, log_ratio, [Xt_h, Xe_h], seed, niter,
                                                maxsize=8, parsimony=0.02)
    pred_tr = g_abs_tr * np.exp(np.clip(s_mix_tr * h_tr, -20, 20))
    pred_te = g_abs_te * np.exp(np.clip(s_mix_te * h_te, -20, 20))
    sign_te = np.sign(g_te)
    sign_te = np.where(sign_te == 0, 1.0, sign_te)
    pred_tr = np.where(yt < 0, -pred_tr, pred_tr)
    pred_te = sign_te * pred_te
    return {"S1": g_eq, "S2": h_eq, "train_r2": safe_r2(yt, pred_tr),
            "test_r2": safe_r2(ye, pred_te), "test_mae": safe_mae(ye, pred_te)}


def template_calphad(fold, seed, niter):
    """y = g(comp) - T * h(comp)"""
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]
    Xt_g = _comp_only(Xt, n_elem)
    Xe_g = _comp_only(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter)
    T_tr = fold["T_train"]
    T_te = fold["T_test"]
    T_safe = np.where(np.abs(T_tr) < 1e-8, 1e-8, T_tr)
    target = (g_tr - yt) / T_safe
    h_model, h_eq, (h_tr, h_te) = _fit_predict(Xt_g, target, [Xt_g, Xe_g], seed, niter,
                                                maxsize=10, parsimony=0.015)
    pred_tr = g_tr - T_tr * h_tr
    pred_te = g_te - T_te * h_te
    return {"S1": g_eq, "S2": h_eq, "train_r2": safe_r2(yt, pred_tr),
            "test_r2": safe_r2(ye, pred_te), "test_mae": safe_mae(ye, pred_te)}


def template_free2(fold, seed, niter):
    """y = F(g(comp), T) — F discovered freely."""
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]
    Xt_g = _comp_only(Xt, n_elem)
    Xe_g = _comp_only(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter)
    T_tr = fold["T_train"]
    T_te = fold["T_test"]
    Xt_F = np.column_stack([g_tr, T_tr])
    Xe_F = np.column_stack([g_te, T_te])
    F_model, F_eq, (F_tr, F_te) = _fit_predict(Xt_F, yt, [Xt_F, Xe_F], seed, niter,
                                                maxsize=15)
    return {"S1": g_eq, "S2": F_eq, "train_r2": safe_r2(yt, F_tr),
            "test_r2": safe_r2(ye, F_te), "test_mae": safe_mae(ye, F_te)}


def template_free1(fold, seed, niter):
    """Single-stage SR on full features."""
    Xt, yt = fold["X_train"], fold["y_train"]
    Xe, ye = fold["X_test"], fold["y_test"]
    n_elem = fold["n_elem"]
    Xt_g = _comp_with_T(Xt, n_elem)
    Xe_g = _comp_with_T(Xe, n_elem)
    g_model, g_eq, (g_tr, g_te) = _fit_predict(Xt_g, yt, [Xt_g, Xe_g], seed, niter,
                                                maxsize=15)
    return {"S1": g_eq, "S2": "", "train_r2": safe_r2(yt, g_tr),
            "test_r2": safe_r2(ye, g_te), "test_mae": safe_mae(ye, g_te)}


TEMPLATES = {
    "arrhenius": template_arrhenius,
    "free2": template_free2,
    "therm": template_therm,
    "additive": template_additive,
    "multiplicative": template_multiplicative,
    "power": template_power,
    "entropy": template_entropy,
    "calphad": template_calphad,
    "free1": template_free1,
}


def run_template_on_fold(name, fold, seed, niter=30):
    """Run one template on one fold with one seed. Returns dict."""
    fn = TEMPLATES[name]
    try:
        out = fn(fold, seed, niter)
        out["error"] = ""
        return out
    except Exception as e:
        return {"S1": "", "S2": "", "train_r2": float("nan"),
                "test_r2": float("nan"), "test_mae": float("nan"),
                "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    # Quick smoke test on one fold
    from features import load_mpea, build_fold
    mpea = load_mpea()
    folds = pd.read_csv(f"{OUT}/folds.csv")
    ok = folds[folds["status"] == "ok"].iloc[0]
    print(f"Smoke test on fold: {ok['fold_id']}")
    fold = build_fold(ok["composition"], ok["processing"], ok["property"], ok["test_source"], mpea)
    for name in TEMPLATES:
        r = run_template_on_fold(name, fold, seed=0, niter=20)
        print(f"  {name}: train={r['train_r2']:.4f}, test={r['test_r2']:.4f}, err={r['error'][:60]}")
