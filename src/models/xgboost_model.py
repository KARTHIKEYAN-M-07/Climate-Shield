"""
src/models/xgboost_model.py
───────────────────────────
XGBoost Regressor wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_xgboost_model(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """Build an XGBoost regression pipeline with median imputation."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    return pipeline
