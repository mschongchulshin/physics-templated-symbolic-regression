import numpy as np
import pandas as pd
from .base_template import make_pysr, best_info, extract_pareto, feat_mapping


def fit(X_comp_all, y_all, T_all, mask300, seed, tmpdir):
    g_model = make_pysr(seed, niterations=200, maxsize=20, parsimony=0.005,
                        unary_operators=["square", "cube", "sqrt", "log"],
                        tempdir=f"{tmpdir}_g")
    g_model.fit(X_comp_all[mask300], y_all[mask300])
    g_pred = g_model.predict(X_comp_all)

    X_stage2 = pd.DataFrame({"g": g_pred, "T": T_all})

    f_model = make_pysr(seed, niterations=300, maxsize=25, parsimony=0.003,
                        tempdir=f"{tmpdir}_F")
    f_model.fit(X_stage2, y_all)

    y_pred = f_model.predict(X_stage2)

    return {
        "stage1": {
            "model": g_model,
            "info": best_info(g_model, X_comp_all[mask300], y_all[mask300]),
            "pareto": extract_pareto(g_model, X_comp_all[mask300], y_all[mask300]),
            "feature_mapping": feat_mapping(X_comp_all.columns.tolist()),
        },
        "stage2": {
            "model": f_model,
            "info": best_info(f_model, X_stage2, y_all),
            "pareto": extract_pareto(f_model, X_stage2, y_all),
            "feature_mapping": feat_mapping(X_stage2.columns.tolist()),
        },
        "y_pred": y_pred,
    }
