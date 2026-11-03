"""
src/models/baselines.py
───────────────────────
Additional empirical & statistical baseline models:
  1. SeasonalClimatology: Day-of-year mean computed STRICTLY on the training set.
  2. DampedPersistence (AR(1)):
       y_hat(t + h) = mu + (phi ^ h) * (y(t) - mu)
     where mu (mean) and phi (lag-1 autocorrelation) are fit on train only.
  3. Persistence: Re-exported for convenience (y_hat(t+h) = y(t)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.persistence import PersistenceBaseline


class SeasonalClimatology:
    """Seasonal Climatology baseline: predicts the training-set historical mean

    for the specific calendar day-of-year (1 to 366).
    """

    def __init__(self, date_col: str = "date", discharge_col: str = "river_discharge"):
        self.date_col = date_col
        self.discharge_col = discharge_col
        self.day_of_year_means: dict[int, float] = {}
        self.overall_mean: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> "SeasonalClimatology":
        """Fit day-of-year mean strictly on the training set."""
        df = X.copy()
        if self.discharge_col not in df.columns:
            if y is not None:
                df[self.discharge_col] = np.asarray(y, dtype=float)
            else:
                raise KeyError(f"SeasonalClimatology requires '{self.discharge_col}' in X or as y.")

        dates = pd.to_datetime(df[self.date_col])
        md = list(zip(dates.dt.month, dates.dt.day))
        df["_md"] = md
        means = df.groupby("_md")[self.discharge_col].mean().to_dict()
        self.day_of_year_means = means
        self.overall_mean = float(df[self.discharge_col].mean())
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict seasonal mean for each sample's calendar month and day."""
        if not self.is_fitted:
            raise RuntimeError("SeasonalClimatology must be fitted before predict().")
        dates = pd.to_datetime(X[self.date_col])
        md = list(zip(dates.dt.month, dates.dt.day))
        preds = [self.day_of_year_means.get(d, self.overall_mean) for d in md]
        return np.asarray(preds, dtype=float)


class DampedPersistence:
    """Damped Persistence / AR(1) baseline:

      y_hat(t + h) = mu + (phi ^ h) * (y(t) - mu)

    Fit strictly on train to estimate mu and lag-1 autocorrelation phi.
    """

    def __init__(
        self,
        horizon_days: int = 1,
        discharge_col: str = "river_discharge",
    ):
        self.horizon_days = horizon_days
        self.discharge_col = discharge_col
        self.mu: float = 0.0
        self.phi: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> "DampedPersistence":
        """Estimate mean (mu) and lag-1 autocorrelation (phi) on train only."""
        series = pd.Series(X[self.discharge_col]).dropna()
        self.mu = float(series.mean())
        if len(series) > 2:
            autocorr = float(series.autocorr(lag=1))
            # Bound phi between 0 and 1 for realistic hydrologic damping
            self.phi = float(np.clip(autocorr, 0.0, 0.99))
        else:
            self.phi = 0.9
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict damped persistence."""
        if not self.is_fitted:
            raise RuntimeError("DampedPersistence must be fitted before predict().")
        yt = X[self.discharge_col].values
        damping = self.phi ** self.horizon_days
        preds = self.mu + damping * (yt - self.mu)
        return np.asarray(preds, dtype=float)


class SeasonalNaive:
    """Seasonal Naive baseline: predicts the discharge value from the exact same calendar day last year.

    If the date from 1 year prior is not available in historical lookback,
    falls back to historical day-of-year climatological mean.
    """

    def __init__(
        self,
        date_col: str = "date",
        discharge_col: str = "river_discharge",
    ):
        self.date_col = date_col
        self.discharge_col = discharge_col
        self.history_map: dict[str, float] = {}
        self.climatology_map: dict[tuple[int, int], float] = {}
        self.overall_mean: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> "SeasonalNaive":
        """Record historical date-indexed flow and day-of-year climatology on train set only."""
        df = X.copy()
        if self.discharge_col not in df.columns:
            if y is not None:
                df[self.discharge_col] = np.asarray(y, dtype=float)
            else:
                raise KeyError(f"SeasonalNaive requires '{self.discharge_col}' in X or as y.")

        dates = pd.to_datetime(df[self.date_col])
        values = df[self.discharge_col].values
        self.history_map = {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(dates, values)}

        md = list(zip(dates.dt.month, dates.dt.day))
        df["_md"] = md
        self.climatology_map = df.groupby("_md")[self.discharge_col].mean().to_dict()
        self.overall_mean = float(df[self.discharge_col].mean())
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict flow from 1 year prior, or fallback to climatology."""
        if not self.is_fitted:
            raise RuntimeError("SeasonalNaive must be fitted before predict().")
        dates = pd.to_datetime(X[self.date_col])
        preds = []
        for d in dates:
            prior_date = (d - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
            if prior_date in self.history_map:
                preds.append(self.history_map[prior_date])
            else:
                md = (d.month, d.day)
                preds.append(self.climatology_map.get(md, self.overall_mean))
        return np.asarray(preds, dtype=float)
