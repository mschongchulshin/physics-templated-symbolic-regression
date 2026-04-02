import numpy as np
import pandas as pd
from .base_template import make_pysr, best_info, extract_pareto, feat_mapping


def fit(X_comp_all, y_all, T_all, mask300, seed, tmpdir):
    g_model = make_pysr(seed, niterations=200, maxsize=20, parsimony=0.005,
                        unary_operators=["square", "cube", "sqrt", "log"],
                        tempdir=f"{tmpdir}_g")
    g_model.fit(X_comp_all[mask300], y_all[mask300])
    g_pred = g_model.predict(X_comp_all)

    g_safe = np.where(np.abs(g_pred) < 1e-10, 1e-10, g_pred)
    ratio = np.nan_to_num(y_all / g_safe, nan=1.0, posinf=1.0, neginf=1.0)
    X_T = pd.DataFrame({"T": T_all})

    h_model = make_pysr(seed, niterations=200, maxsize=15, parsimony=0.005,
                        tempdir=f"{tmpdir}_h")
    h_model.fit(X_T, ratio)

    y_pred = g_pred * h_model.predict(X_T)

    return {
        "stage1": {
            "model": g_model,
            "info": best_info(g_model, X_comp_all[mask300], y_all[mask300]),
            "pareto": extract_pareto(g_model, X_comp_all[mask300], y_all[mask300]),
            "feature_mapping": feat_mapping(X_comp_all.columns.tolist()),
        },
        "stage2": {
            "model": h_model,
            "info": best_info(h_model, X_T, ratio),
            "pareto": extract_pareto(h_model, X_T, ratio),
            "feature_mapping": feat_mapping(X_T.columns.tolist()),
        },
        "y_pred": y_pred,
    }
