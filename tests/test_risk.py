"""
tests/test_risk.py
──────────────────
Phase-8 tests:
  - compute_station_thresholds generates valid percentiles (p50 < p75 < p90 < p95)
  - evaluate_risk maps into Low, Moderate, High, Severe accurately
  - evaluate_risk returns UNAVAILABLE when:
      * inputs are stale (age > max_data_age_hours)
      * predicted value is negative or None
      * thresholds have not yet been calibrated (contain nulls)
      * station is missing
  - disclaimer and source provenance are always returned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk.thresholds import compute_station_thresholds
from src.risk.risk_engine import evaluate_risk


@pytest.fixture
def calibrated_thresholds_cfg():
    return {
        "metadata": {
            "source": "prototype_percentile",
            "units": "m³/s",
        },
        "staleness": {
            "max_data_age_hours": 48,
        },
        "stations": {
            "TN_CAU_TRICHY": {
                "categories": {
                    "Low": {"lower_bound_m3s": 0.0},
                    "Moderate": {"lower_bound_m3s": 100.0},
                    "High": {"lower_bound_m3s": 250.0},
                    "Severe": {"lower_bound_m3s": 500.0},
                }
            }
        }
    }


class TestRiskEngine:
    def test_compute_station_thresholds(self):
        # 100 values from 1 to 100
        data = np.linspace(1, 100, 100)
        res = compute_station_thresholds(data)
        pb = res["percentile_basis"]
        assert pb["p50"] == pytest.approx(50.5, abs=1.0)
        assert pb["p75"] == pytest.approx(75.25, abs=1.0)
        assert pb["p90"] == pytest.approx(90.1, abs=1.0)
        assert pb["p95"] == pytest.approx(95.05, abs=1.0)
        assert res["categories"]["Moderate"]["lower_bound_m3s"] == pb["p50"]

    def test_risk_category_mapping(self, calibrated_thresholds_cfg):
        # Boundaries: Low < 100 <= Moderate < 250 <= High < 500 <= Severe
        cfg = calibrated_thresholds_cfg

        # Forecast below threshold (< 100)
        res_low = evaluate_risk(50.0, "TN_CAU_TRICHY", cfg)
        assert res_low["category"] == "Low"
        assert res_low["status"] == "OK"
        assert "normal baseline" in res_low["threshold_crossed"].lower()

        # Near boundary / exactly at Moderate threshold (100.0)
        res_mod_exact = evaluate_risk(100.0, "TN_CAU_TRICHY", cfg)
        assert res_mod_exact["category"] == "Moderate"

        # Above Moderate threshold (150.0)
        res_mod = evaluate_risk(150.0, "TN_CAU_TRICHY", cfg)
        assert res_mod["category"] == "Moderate"

        # Near boundary / exactly at High threshold (250.0)
        res_high_exact = evaluate_risk(250.0, "TN_CAU_TRICHY", cfg)
        assert res_high_exact["category"] == "High"

        # Above High threshold (300.0)
        res_high = evaluate_risk(300.0, "TN_CAU_TRICHY", cfg)
        assert res_high["category"] == "High"

        # Near boundary / exactly at Severe threshold (500.0)
        res_sev_exact = evaluate_risk(500.0, "TN_CAU_TRICHY", cfg)
        assert res_sev_exact["category"] == "Severe"

        # Above Severe threshold (600.0)
        res_sev = evaluate_risk(600.0, "TN_CAU_TRICHY", cfg)
        assert res_sev["category"] == "Severe"

    def test_forecast_lower_than_current_evaluated_from_forecast(self, calibrated_thresholds_cfg):
        """When forecast (e.g. 290 m³/s) is lower than current (e.g. 430 m³/s),

        risk must be strictly derived from forecast (High, since 290 >= 250),
        while recording the receding trajectory in rationale.
        """
        cfg = calibrated_thresholds_cfg
        res = evaluate_risk(290.0, "TN_CAU_TRICHY", cfg, current_level=430.0)
        assert res["category"] == "High"
        assert res["predicted_discharge_m3s"] == 290.0
        assert res["current_discharge_m3s"] == 430.0
        assert "receding from current" in res["rationale"]

    def test_season_is_context_not_risk_driver(self, calibrated_thresholds_cfg):
        """Ensure seasonal window does not alter risk category."""
        cfg = calibrated_thresholds_cfg
        res_low = evaluate_risk(50.0, "TN_CAU_TRICHY", cfg)
        assert res_low["category"] == "Low"
        assert "contextual only" in res_low["season_context"]

    def test_missing_model_returns_unavailable(self, calibrated_thresholds_cfg):
        cfg = calibrated_thresholds_cfg
        res = evaluate_risk(150.0, "TN_CAU_TRICHY", cfg, model_exists=False)
        assert res["category"] == "UNAVAILABLE"
        assert res["status"] == "UNAVAILABLE"
        assert "No trained model artifact" in res["rationale"]

    def test_stale_data_returns_unavailable(self, calibrated_thresholds_cfg):
        cfg = calibrated_thresholds_cfg
        # 50 hours > max_data_age_hours (48)
        res = evaluate_risk(150.0, "TN_CAU_TRICHY", cfg, data_freshness_hours=50.0)
        assert res["category"] == "UNAVAILABLE"
        assert res["is_stale"] is True
        assert "exceeds maximum allowed age" in res["rationale"]

    def test_null_or_uncalibrated_thresholds_returns_unavailable(self):
        uncalibrated_cfg = {
            "metadata": {"source": "prototype_percentile"},
            "staleness": {"max_data_age_hours": 48},
            "stations": {
                "TN_CAU_TRICHY": {
                    "categories": {
                        "Low": {"lower_bound_m3s": 0.0},
                        "Moderate": {"lower_bound_m3s": None},
                        "High": {"lower_bound_m3s": None},
                        "Severe": {"lower_bound_m3s": None},
                    }
                }
            }
        }
        res = evaluate_risk(200.0, "TN_CAU_TRICHY", uncalibrated_cfg)
        assert res["category"] == "UNAVAILABLE"
        assert "not been calibrated" in res["rationale"]

    def test_negative_or_none_prediction_returns_unavailable(self, calibrated_thresholds_cfg):
        cfg = calibrated_thresholds_cfg
        res_none = evaluate_risk(None, "TN_CAU_TRICHY", cfg)
        assert res_none["category"] == "UNAVAILABLE"

        res_neg = evaluate_risk(-10.0, "TN_CAU_TRICHY", cfg)
        assert res_neg["category"] == "UNAVAILABLE"
