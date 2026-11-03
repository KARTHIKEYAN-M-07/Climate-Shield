"""
tests/test_uncertainty.py
─────────────────────────
Phase-16 tests:
  - Split-conformal calibration on validation set:
      * Calibrates without accessing test set
      * Produces valid positive non-conformity interval
      * Satisfies non-negative lower bound constraint
  - QuantileIntervalForecaster:
      * Fits lower and upper quantile pipelines
      * Non-crossing assertion: upper >= lower everywhere
  - Uncertainty metrics:
      * Pinball loss for known errors
      * evaluate_prediction_interval reports exact coverage and mean width
      * Under-coverage flag triggers when coverage < nominal - 5%
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.uncertainty import (
    ConformalIntervalCalibrator,
    QuantileIntervalForecaster,
    pinball_loss,
    evaluate_prediction_interval,
)


class TestConformalCalibration:
    def test_conformal_calibrates_and_bounds_predictions(self):
        # Validation residuals
        y_val_true = np.array([100.0, 150.0, 200.0, 120.0, 180.0, 220.0, 130.0, 160.0, 190.0, 210.0])
        y_val_pred = y_val_true + np.array([5.0, -10.0, 15.0, -5.0, 8.0, -12.0, 4.0, -6.0, 9.0, -7.0])

        calibrator = ConformalIntervalCalibrator(confidence_level=0.90)
        calibrator.calibrate(y_val_true, y_val_pred)
        assert calibrator.is_calibrated is True
        assert calibrator.q_hat > 0.0

        # Predict interval for a low value
        test_pred = np.array([5.0, 100.0])
        lower, upper = calibrator.predict_interval(test_pred)

        assert len(lower) == 2
        assert len(upper) == 2
        # Non-negative lower bound
        assert (lower >= 0.0).all()
        assert (upper >= lower).all()

    def test_uncalibrated_raises_error(self):
        calibrator = ConformalIntervalCalibrator()
        with pytest.raises(RuntimeError, match="must be calibrated"):
            calibrator.predict_interval([100.0])


class TestQuantileForecaster:
    def test_quantile_regressor_non_crossing(self):
        np.random.seed(42)
        n = 50
        X = pd.DataFrame({
            "feat1": np.random.uniform(10, 50, n),
            "feat2": np.random.uniform(1, 10, n),
        })
        y = X["feat1"] * 2.0 + np.random.normal(0, 5, n)

        forecaster = QuantileIntervalForecaster(confidence_level=0.80, random_state=42)
        forecaster.fit(X, y)
        lower, upper = forecaster.predict_interval(X)

        assert len(lower) == n
        assert len(upper) == n
        # Enforced non-crossing property
        assert (upper >= lower).all()
        assert (lower >= 0.0).all()


class TestUncertaintyMetrics:
    def test_pinball_loss(self):
        # yt = 10, yp = 8 -> err = 2 (underestimate)
        # quantile = 0.9 -> loss = 0.9 * 2 = 1.8
        loss = pinball_loss([10.0], [8.0], quantile=0.9)
        assert loss == pytest.approx(1.8)

        # yt = 10, yp = 12 -> err = -2 (overestimate)
        # quantile = 0.9 -> loss = (0.9 - 1) * (-2) = 0.2
        loss_over = pinball_loss([10.0], [12.0], quantile=0.9)
        assert loss_over == pytest.approx(0.2)

    def test_evaluate_prediction_interval(self):
        yt = [10.0, 20.0, 30.0, 40.0]
        # 3 out of 4 covered (75%)
        lb = [5.0, 15.0, 25.0, 45.0]  # last one misses
        ub = [15.0, 25.0, 35.0, 55.0]

        res = evaluate_prediction_interval(yt, lb, ub, nominal_confidence=0.90)
        assert res["empirical_coverage_pct"] == pytest.approx(75.0)
        assert res["mean_width_m3s"] == pytest.approx(10.0)
        assert res["is_undercovering"] is True  # 75% < 85%
        assert "falls below nominal" in res["status_message"]
