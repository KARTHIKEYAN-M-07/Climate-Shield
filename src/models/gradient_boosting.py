"""
src/models/gradient_boosting.py
───────────────────────────────
Gradient Boosting Regressor wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_gradient_boosting_model(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    random_state: int = 42,
) -> Pipeline:
    """Build a Gradient Boosting regression pipeline with median imputation."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )),
    ])
    return pipeline
