"""
tests/test_robust_evaluation.py
───────────────────────────────
Phase-15 tests:
  - Kling-Gupta Efficiency (KGE):
      * Perfect model: KGE = 1.0
      * Hand-calculated case with known r, beta, gamma
  - Percent Bias (PBIAS):
      * Zero bias = 0.0
      * Systematic overestimation / underestimation
  - log-NSE for low flows
  - MSE skill score vs baseline
  - Top 5% relative flow error
  - Paired Diebold-Mariano test:
      * Statistically significant case (large consistent error difference)
      * Non-significant case (identical forecasts)
  - Multi-threshold event metrics (POD, FAR, CSI at P90, P95, P99)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    kling_gupta_efficiency,
    percent_bias,
    log_nash_sutcliffe_efficiency,
    mse_skill_score,
    top_flow_relative_error,
    diebold_mariano_test,
    compute_standard_metrics,
)
from src.evaluation.event_metrics import compute_multi_threshold_event_metrics


class TestAdvancedContinuousMetrics:
    def test_kge_perfect_score(self):
        y = [10.0, 20.0, 30.0, 40.0]
        kge = kling_gupta_efficiency(y, y)
        assert kge == pytest.approx(1.0)

    def test_kge_known_value(self):
        # Known series
        yt = np.array([10.0, 20.0, 30.0, 40.0])
        # Scaled by 1.1 -> r = 1.0, beta = 1.1, gamma = 1.0
        # KGE = 1 - sqrt(0^2 + 0.1^2 + 0^2) = 1 - 0.1 = 0.90
        yp = yt * 1.1
        kge = kling_gupta_efficiency(yt, yp)
        assert kge == pytest.approx(0.90)

    def test_pbias_calculation(self):
        yt = [10.0, 20.0, 30.0]  # sum = 60
        yp = [12.0, 22.0, 32.0]  # sum = 66 -> diff = +6 -> PBIAS = 10.0%
        assert percent_bias(yt, yp) == pytest.approx(10.0)

    def test_log_nse(self):
        yt = [1.0, 2.0, 3.0, 4.0]
        yp = [1.0, 2.0, 3.0, 4.0]
        assert log_nash_sutcliffe_efficiency(yt, yp) == pytest.approx(1.0)

    def test_mse_skill_score(self):
        yt = [10.0, 20.0]
        yp = [10.0, 20.0]        # MSE = 0
        yb = [0.0, 0.0]          # MSE = 250
        # SS = 1 - 0/250 = 1.0
        assert mse_skill_score(yt, yp, yb) == pytest.approx(1.0)

    def test_top_flow_relative_error(self):
        yt = np.linspace(1, 100, 100)
        # 10% overestimate
        yp = yt * 1.10
        err = top_flow_relative_error(yt, yp, top_quantile=0.95)
        assert err == pytest.approx(10.0, abs=0.1)


class TestDieboldMarianoTest:
    def test_dm_identical_forecasts_not_significant(self):
        y = np.random.uniform(50, 100, 30)
        res = diebold_mariano_test(y, y, y)
        assert res["statistically_significant"] is False
        assert res["p_value"] == pytest.approx(1.0)
        assert "No statistically significant difference" in res["interpretation"]

    def test_dm_significant_improvement(self):
        np.random.seed(42)
        yt = np.random.uniform(50, 100, 40)
        # Model is almost exact
        yp = yt + np.random.normal(0, 0.1, 40)
        # Baseline is very inaccurate
        yb = yt + np.random.normal(20, 5, 40)
        res = diebold_mariano_test(yt, yp, yb)
        assert res["statistically_significant"] is True
        assert res["p_value"] < 0.001
        assert "Statistically significant improvement" in res["interpretation"]


class TestMultiThresholdEventMetrics:
    def test_multi_threshold_metrics(self):
        yt = np.array([10.0, 50.0, 100.0, 200.0])
        yp = np.array([15.0, 55.0, 110.0, 190.0])
        threshs = {"p90": 90.0, "p95": 150.0}

        res = compute_multi_threshold_event_metrics(yt, yp, thresholds_dict=threshs)
        assert "threshold_metrics" in res
        assert "p90" in res["threshold_metrics"]
        assert "p95" in res["threshold_metrics"]

        p95 = res["threshold_metrics"]["p95"]
        assert p95["TP"] == 1
        assert p95["POD"] == 1.0
        assert p95["CSI"] == 1.0
