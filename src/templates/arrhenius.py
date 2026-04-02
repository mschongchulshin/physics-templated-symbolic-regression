import numpy as np
import pandas as pd
from .base_template import make_pysr, best_info, extract_pareto, feat_mapping


def fit(X_comp_all, y_all, T_all, mask300, seed, tmpdir):
    g_model = make_pysr(seed, niterations=200, maxsize=20, parsimony=0.005,
                        unary_operators=["square", "cube", "sqrt", "log"],
                        tempdir=f"{tmpdir}_g")
    g_model.fit(X_comp_all[mask300], y_all[mask300])
    g_pred = g_model.predict(X_comp_all)

    g_safe = np.clip(np.abs(g_pred), 1e-10, None)
    y_safe = np.clip(np.abs(y_all), 1e-10, None)
    log_resid = np.nan_to_num(np.log(y_safe) - np.log(g_safe),
                              nan=0.0, posinf=0.0, neginf=0.0)

    inv_T = 1.0 / T_all
    X_stage2 = pd.concat(
        [X_comp_all.reset_index(drop=True), pd.DataFrame({"inv_T": inv_T})],
        axis=1,
    )
    h_model = make_pysr(seed, niterations=200, maxsize=15, parsimony=0.01,
                        tempdir=f"{tmpdir}_h")
    h_model.fit(X_stage2, log_resid)

    h_pred = np.clip(h_model.predict(X_stage2), -20, 20)
    y_pred = g_safe * np.exp(h_pred)

    return {
        "stage1": {
            "model": g_model,
            "info": best_info(g_model, X_comp_all[mask300], y_all[mask300]),
            "pareto": extract_pareto(g_model, X_comp_all[mask300], y_all[mask300]),
            "feature_mapping": feat_mapping(X_comp_all.columns.tolist()),
        },
        "stage2": {
            "model": h_model,
            "info": best_info(h_model, X_stage2, log_resid),
            "pareto": extract_pareto(h_model, X_stage2, log_resid),
            "feature_mapping": feat_mapping(X_stage2.columns.tolist()),
        },
        "y_pred": y_pred,
    }
