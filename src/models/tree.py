"""
src/models/tree.py
──────────────────
Decision Tree Regressor wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_decision_tree_model(
    max_depth: int | None = 6,
    min_samples_split: int = 5,
    min_samples_leaf: int = 3,
    random_state: int = 42,
) -> Pipeline:
    """Build a Decision Tree regression pipeline with median imputation."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )),
    ])
    return pipeline
