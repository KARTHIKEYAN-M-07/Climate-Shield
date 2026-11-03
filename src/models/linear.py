"""
src/models/linear.py
────────────────────
Ridge Regression wrapper with StandardScaler preprocessing.
Scalers and imputers are fit STRICTLY on training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_ridge_model(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    """Build a Ridge Regression pipeline with median imputation and standard scaling."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha, random_state=random_state)),
    ])
    return pipeline
