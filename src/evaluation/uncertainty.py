"""
src/evaluation/uncertainty.py
─────────────────────────────
Uncertainty quantification for flood forecasting models:
  1. Split-Conformal Prediction:
     - Calibrated strictly on the validation set.
     - Guarantees finite-sample marginal coverage under exchangeability.
     - Lower bound enforced at 0.0 m³/s (non-negative hydrologic discharge).
  2. Quantile Regression:
     - GradientBoostingRegressor with pinball loss (loss='quantile').
  3. Verification Metrics:
     - Empirical coverage (% of test samples inside interval)
     - Mean interval width
     - Pinball loss for quantiles

Honesty Rule:
  Never display an interval without its measured test coverage displayed next to it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# =============================================================================
# Split-Conformal Prediction Calibrator
# =============================================================================


class ConformalIntervalCalibrator:
    """Split-conformal prediction interval generator calibrated on validation set."""

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.q_hat: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(self, y_val_true: np.ndarray | list[float], y_val_pred: np.ndarray | list[float]) -> "ConformalIntervalCalibrator":
        """Compute non-conformity residuals on the validation set only."""
        yt = np.asarray(y_val_true, dtype=float)
        yp = np.asarray(y_val_pred, dtype=float)
        n = len(yt)
        if n < 5:
            raise ValueError(f"Validation set too small (n={n}) for conformal calibration.")

        # Absolute error non-conformity scores
        residuals = np.abs(yt - yp)

        # Finite-sample quantile formula: ceil((n + 1) * (1 - alpha)) / n
        level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.q_hat = float(np.quantile(residuals, level, method="higher"))
        self.is_calibrated = True
        return self

    def predict_interval(self, y_pred: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Produce lower and upper bounds given point forecasts."""
        if not self.is_calibrated:
            raise RuntimeError("ConformalIntervalCalibrator must be calibrated on validation set first.")

        yp = np.asarray(y_pred, dtype=float)
        # Discharge cannot physically be negative
        lower = np.maximum(0.0, yp - self.q_hat)
        upper = yp + self.q_hat
        return lower, upper


# =============================================================================
# Quantile Regression Models
# =============================================================================


def build_quantile_regressor(quantile: float, n_estimators: int = 100, random_state: int = 42) -> Pipeline:
    """Build a GradientBoostingRegressor pipeline targeting a specific quantile."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
        )),
    ])


class QuantileIntervalForecaster:
    """Trains separate lower and upper quantile regression models."""

    def __init__(self, confidence_level: float = 0.90, random_state: int = 42):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.q_low = self.alpha / 2.0
        self.q_high = 1.0 - self.alpha / 2.0
        self.model_low = build_quantile_regressor(self.q_low, random_state=random_state)
        self.model_high = build_quantile_regressor(self.q_high, random_state=random_state)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray | pd.Series) -> "QuantileIntervalForecaster":
        self.model_low.fit(X_train, y_train)
        self.model_high.fit(X_train, y_train)
        return self

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lower = np.maximum(0.0, self.model_low.predict(X))
        upper = self.model_high.predict(X)
        # Enforce crossing prevention: upper >= lower
        upper = np.maximum(lower, upper)
        return lower, upper


# =============================================================================
# Verification & Calibration Metrics
# =============================================================================


def pinball_loss(y_true: np.ndarray | list[float], y_pred_q: np.ndarray | list[float], quantile: float) -> float:
    """Calculate pinball / quantile loss for quantile q in (0, 1)."""
    yt = np.asarray(y_true, dtype=float)
    yq = np.asarray(y_pred_q, dtype=float)
    err = yt - yq
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def evaluate_prediction_interval(
    y_true: np.ndarray | list[float],
    lower_bound: np.ndarray | list[float],
    upper_bound: np.ndarray | list[float],
    nominal_confidence: float = 0.90,
) -> dict[str, Any]:
    """Evaluate empirical coverage, mean interval width, and coverage status on the test set."""
    yt = np.asarray(y_true, dtype=float)
    lb = np.asarray(lower_bound, dtype=float)
    ub = np.asarray(upper_bound, dtype=float)

    if len(yt) == 0:
        return {"empirical_coverage_pct": 0.0, "mean_width_m3s": 0.0, "is_undercovering": True}

    inside = (yt >= lb) & (yt <= ub)
    coverage = float(np.mean(inside) * 100.0)
    mean_width = float(np.mean(ub - lb))

    # Tolerance of 5% below nominal
    is_undercovering = bool(coverage < (nominal_confidence * 100.0 - 5.0))

    status_message = (
        f"Empirical coverage ({coverage:.1f}%) meets nominal ({nominal_confidence * 100:.0f}%)."
        if not is_undercovering
        else f"Warning: Empirical coverage ({coverage:.1f}%) falls below nominal ({nominal_confidence * 100:.0f}%)."
    )

    return {
        "nominal_confidence_pct": round(nominal_confidence * 100.0, 1),
        "empirical_coverage_pct": round(coverage, 2),
        "mean_width_m3s": round(mean_width, 2),
        "is_undercovering": is_undercovering,
        "status_message": status_message,
        "total_test_samples": len(yt),
        "covered_samples": int(np.sum(inside)),
    }
