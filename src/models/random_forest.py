"""
src/models/random_forest.py
───────────────────────────
Random Forest Regressor wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_random_forest_model(
    n_estimators: int = 100,
    max_depth: int | None = 8,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> Pipeline:
    """Build a Random Forest regression pipeline with median imputation."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    return pipeline
