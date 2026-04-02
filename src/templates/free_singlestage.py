import numpy as np
import pandas as pd
from .base_template import make_pysr, best_info, extract_pareto, feat_mapping


def fit(X_full, y_all, seed, tmpdir):
    model = make_pysr(seed, niterations=200, maxsize=25, parsimony=0.005,
                      tempdir=tmpdir)
    model.fit(X_full, y_all)
    y_pred = model.predict(X_full)

    return {
        "stage1": {
            "model": model,
            "info": best_info(model, X_full, y_all),
            "pareto": extract_pareto(model, X_full, y_all),
            "feature_mapping": feat_mapping(X_full.columns.tolist()),
        },
        "y_pred": y_pred,
    }
