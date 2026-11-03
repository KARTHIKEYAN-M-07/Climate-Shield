"""
src/models/persistence.py
─────────────────────────
Persistence baseline model:
  Forecast for future horizon t + h is simply the last observed discharge at t.

  y_hat(t + h) = river_discharge(t)

Mandatory baseline against which all ML models must be evaluated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Persistence baseline: predicts future discharge as the current observed discharge."""

    def __init__(self, discharge_col: str = "river_discharge"):
        self.discharge_col = discharge_col
        self.is_fitted = True  # Baseline requires no training parameter fits

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> "PersistenceBaseline":
        """No-op fit to satisfy estimator interface."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the value of the most recent river_discharge available at origin t.

        X must contain the column specified by self.discharge_col.
        """
        if self.discharge_col not in X.columns:
            raise KeyError(
                f"Persistence model requires column '{self.discharge_col}' in features X."
            )
        return np.asarray(X[self.discharge_col].values, dtype=float)
