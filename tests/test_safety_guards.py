"""
tests/test_safety_guards.py
───────────────────────────
Phase-17 tests:
  - OutOfDistributionGuard:
      * Normal in-distribution samples pass (is_ood = False)
      * Extreme outlier triggers OOD warning (is_ood = True) with violation details
  - explain_prediction:
      * Generates natural language statements for top rainfall, surge, API, and monsoon
  - simulate_rainfall_what_if:
      * Correctly scales precipitation features
      * Carries explicit HYPOTHETICAL flag and tree-extrapolation caveat
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk.safety_guards import (
    OutOfDistributionGuard,
    explain_prediction,
    simulate_rainfall_what_if,
)
from src.models.linear import build_ridge_model


class TestSafetyGuards:
    def test_ood_guard_normal_and_outlier(self):
        np.random.seed(42)
        X_train = pd.DataFrame({
            "precipitation_sum": np.random.uniform(0, 50, 100),
            "river_discharge": np.random.uniform(50, 200, 100),
        })

        guard = OutOfDistributionGuard(iqr_multiplier=3.0)
        guard.fit(X_train)

        # Normal sample
        normal_sample = {"precipitation_sum": 25.0, "river_discharge": 120.0}
        res_norm = guard.check_sample(normal_sample)
        assert res_norm["is_ood"] is False
        assert res_norm["status"] == "IN_DISTRIBUTION"

        # Outlier sample (5000 mm rainfall!)
        outlier_sample = {"precipitation_sum": 5000.0, "river_discharge": 120.0}
        res_out = guard.check_sample(outlier_sample)
        assert res_out["is_ood"] is True
        assert res_out["status"] == "OOD_DETECTED"
        assert len(res_out["violations"]) == 1
        assert res_out["violations"][0]["feature"] == "precipitation_sum"

    def test_explain_prediction(self):
        sample = {
            "precipitation_sum_cumul_7d": 120.0,
            "river_discharge_roc_1d": 35.0,
            "api_decay_90": 55.0,
            "monsoon_ne": 1.0,
        }
        train_stats = {
            "precipitation_sum_cumul_7d": {"p90": 80.0},
        }

        explanations = explain_prediction(sample, train_stats)
        assert len(explanations) >= 3
        text = " ".join(explanations)
        assert "7-day cumulative rainfall" in text
        assert "Rapid discharge surge" in text
        assert "Northeast Monsoon" in text

    def test_simulate_rainfall_what_if(self):
        X = pd.DataFrame({
            "precipitation_sum": [10.0, 20.0, 30.0],
            "river_discharge": [50.0, 60.0, 70.0],
        })
        y = np.array([55.0, 68.0, 80.0])

        model = build_ridge_model()
        model.fit(X, y)

        sample = X.iloc[[0]]
        # +50% rain scenario (1.5x)
        res = simulate_rainfall_what_if(model, sample, rainfall_multiplier=1.5)

        assert res["status"] == "HYPOTHETICAL_SIMULATION"
        assert res["delta_rainfall_pct"] == 50.0
        assert "HYPOTHETICAL SCENARIO ONLY" in res["caveat"]
        assert "cannot extrapolate beyond the maximum observed rainfall" in res["caveat"]
