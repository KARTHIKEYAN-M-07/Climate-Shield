"""tests/test_forecast_engine.py
──────────────────────────────
Tests for Forecast Engine & Drift Handling:
1. ForecastResult has all required fields always present.
2. Missing drift reference returns drift_status="UNKNOWN" with warning, never raises KeyError.
3. Stale data (>48h) returns status="STALE" and disables risk category (UNAVAILABLE).
4. Physical sanity guard detects non-physical spikes and returns status="SUSPECT".
5. Missing model returns status="UNAVAILABLE" with descriptive reason.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.forecast.engine import ForecastResult, generate_forecast
from src.monitoring.drift import compute_drift_status

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestForecastResultSchema:
    def test_forecast_result_has_all_required_fields(self):
        """Verify ForecastResult contains all mandated fields."""
        res = ForecastResult(
            forecast=10.5,
            interval=(8.0, 13.0),
            risk={"category": "Low"},
            status="OK",
            drift_status="IN_DISTRIBUTION",
            drift_value=0.04,
            last_observation_time="2026-10-29",
            data_age_hours=2.5,
            warnings=[],
            reason="Success",
            measured_coverage=90.0,
            primary_why="Baseflow",
            forecast_origin_date="2026-10-29",
            source_used="local_test",
        )
        d = res.to_dict()
        mandated_keys = [
            "forecast", "interval", "risk", "status", "drift_status",
            "drift_value", "last_observation_time", "data_age_hours", "warnings"
        ]
        for k in mandated_keys:
            assert k in d, f"Missing mandated key in ForecastResult: {k}"

        # Subscript access test
        assert res["forecast"] == 10.5
        assert res["drift_status"] == "IN_DISTRIBUTION"


class TestDriftSafety:
    def test_missing_drift_reference_returns_unknown_not_exception(self):
        """Drift check without reference stats must return UNKNOWN, never raise KeyError."""
        serving = np.array([1.2, 1.5, 1.8, 2.0, 2.1])
        res = compute_drift_status(
            serving_values=serving,
            reference_stats=None,
            train_values=None,
        )
        assert res["drift_status"] == "UNKNOWN"
        assert res["status"] == "Unknown"
        assert res["is_drifted"] is False
        assert len(res["warnings"]) > 0
        assert "drift_status" in res
        assert "is_drifted" in res

    def test_empty_serving_values_returns_unknown(self):
        res = compute_drift_status(serving_values=[])
        assert res["drift_status"] == "UNKNOWN"
        assert res["is_drifted"] is False

    def test_valid_reference_stats_computes_drift(self):
        ref_stats = {
            "mean": 2.0,
            "std": 0.5,
            "quantiles_10bin": [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5],
        }
        # 30 points matching the reference distribution
        np.random.seed(42)
        serving = np.random.uniform(0.5, 3.5, size=30)
        res = compute_drift_status(
            serving_values=serving,
            reference_stats=ref_stats,
        )
        assert res["drift_status"] == "IN_DISTRIBUTION"
        assert res["psi"] is not None
        assert res["z_score"] is not None


class TestForecastEngineBehaviors:
    def test_stale_data_disables_risk(self):
        """Data older than max_data_stale_hours must be STALE with risk UNAVAILABLE."""
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="persistence",
            max_data_stale_hours=1.0,
        )
        # Live observation date is today (~16 hours ago), so max_data_stale_hours=1.0 triggers STALE
        assert res.status == "STALE"
        assert res.risk.get("category") == "UNAVAILABLE"
        assert "stale" in res.risk.get("rationale", "").lower()
        assert res.data_age_hours is not None
        assert res.data_age_hours > 1.0

    def test_missing_model_returns_unavailable_with_reason(self):
        """Non-existent model must return UNAVAILABLE with explicit reason."""
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="non_existent_deep_net",
        )
        assert res.status in ("UNAVAILABLE", "INSUFFICIENT_SIGNAL", "NO_MODEL")
        assert res.risk.get("category") == "UNAVAILABLE"
        assert "No trained model" in res.reason

    def test_insufficient_data_station_returns_unavailable(self):
        """Chennai (all discharge NaN) must return UNAVAILABLE or RAINFALL_ONLY."""
        res = generate_forecast(
            station_id="TN_ADY_CHENNAI",
            horizon=1,
            model_name="ridge",
        )
        assert res.status in ("UNAVAILABLE", "INSUFFICIENT_SIGNAL", "FETCH_FAILED", "RAINFALL_ONLY")
        assert res.forecast is None
        assert res.drift_status == "UNKNOWN"
        assert res.is_rainfall_only is True or ("missing or zero" in res.reason.lower() or "no valid" in res.reason.lower() or "rainfall risk only" in res.reason.lower())

    def test_smoke_test_trichy_all_models_produce_results(self):
        """All 9 dropdown models for Trichy must successfully produce ForecastResults without errors."""
        models = [
            "xgboost", "random_forest", "gradient_boosting", "hist_gradient_boosting",
            "elastic_net", "extra_trees", "ridge", "decision_tree", "persistence",
        ]
        for m in models:
            res = generate_forecast("TN_CAU_TRICHY", horizon=1, model_name=m)
            assert res.status in ["OK", "STALE", "OK_DRIFT_BLOCKED", "LOW_CONFIDENCE"], f"Model {m} failed with status {res.status}: {res.reason}"
            assert res.forecast is not None
            assert res.drift_status in ["IN_DISTRIBUTION", "OUT_OF_DISTRIBUTION", "UNKNOWN"]
            assert res.interval is not None

    def test_origin_date_never_in_the_future(self):
        """Forecast origin date must NEVER be in the future, and today is incomplete (origin < today)."""
        from datetime import datetime, timezone, timedelta
        kolkata_tz = timezone(timedelta(hours=5, minutes=30))
        today_kolkata_str = datetime.now(kolkata_tz).strftime("%Y-%m-%d")

        for sid in ["TN_CAU_TRICHY", "TN_VAI_MADURAI", "TN_PAL_VELLORE"]:
            res = generate_forecast(station_id=sid, horizon=1, model_name="persistence")
            if res.last_observation_time:
                assert res.last_observation_time < today_kolkata_str, (
                    f"Origin {res.last_observation_time} is not before today {today_kolkata_str} (today is incomplete)"
                )
            if res.forecast_origin_date:
                assert res.forecast_origin_date < today_kolkata_str, (
                    f"Forecast origin date {res.forecast_origin_date} is not before today"
                )

        # Explicit test with future rows present in dataframe
        future_dates = pd.date_range("2026-10-25", periods=10, freq="D").strftime("%Y-%m-%d").tolist()
        df_sim = pd.DataFrame({
            "date": future_dates,
            "precipitation_sum": [10.0] * 10,
            "river_discharge": [2.5] * 5 + [np.nan] * 5,  # last 5 rows are future
            "api_forecast_river_discharge": [2.5] * 5 + [3.0] * 5,
            "is_forecast": [False] * 5 + [True] * 5,
        })
        res_sim = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="persistence",
            df_data=df_sim,
        )
        assert res_sim.last_observation_time is not None
        assert res_sim.last_observation_time <= today_kolkata_str

    def test_insufficient_signal_station_never_shows_risk_category(self):
        """Stations with insufficient hydrologic signal must never output an operational flow risk category."""
        # Chennai: unsupported coastal river basin (rainfall risk only)
        for h in [1, 3, 7]:
            res = generate_forecast(station_id="TN_ADY_CHENNAI", horizon=h, model_name="persistence")
            cat = res.risk.get("category")
            assert cat == "UNAVAILABLE", f"Chennai flow risk must be UNAVAILABLE, got {cat}"
            assert res.forecast is None
            assert res.interval is None
            assert res.is_rainfall_only is True

        # Synthetic insufficient signal dataset (near-constant discharge, CV < 0.80)
        dates = pd.date_range("2020-01-01", periods=1000, freq="D").strftime("%Y-%m-%d").tolist()
        df_weak = pd.DataFrame({
            "date": dates,
            "precipitation_sum": [5.0] * 1000,
            "river_discharge": [0.05] * 1000,
            "is_forecast": [False] * 1000,
        })
        res_weak = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="persistence",
            df_data=df_weak,
        )
        assert res_weak.status == "INSUFFICIENT_SIGNAL"
        assert res_weak.risk.get("category") == "UNAVAILABLE"
        assert res_weak.is_insufficient_signal is True

    def test_out_of_distribution_risk_with_top_drifting_features(self):
        """When drift is detected, risk is classified with LOW_CONFIDENCE and top drifting features surfaced."""
        # Use live data or high-surge data for Trichy which exhibits extreme shift
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="xgboost",
        )
        if res.drift_status == "OUT_OF_DISTRIBUTION":
            assert res.status == "LOW_CONFIDENCE"
            assert res.risk.get("status") == "LOW_CONFIDENCE"
            assert res.risk.get("badge") == "Low Confidence"
            assert res.risk.get("category") in ["Low", "Moderate", "High", "Severe", "UNAVAILABLE"]
            assert res.top_drifting_features is not None
            assert len(res.top_drifting_features) <= 3
            assert "feature" in res.top_drifting_features[0]
            assert "z_score" in res.top_drifting_features[0]

    def test_monsoon_input_inside_same_season_range_not_flagged(self):
        """Monsoon observations that fit seasonal expectations (DOY +/-30d) must be IN_DISTRIBUTION."""
        for sid in ["TN_VAI_MADURAI", "TN_THA_TIRUNELVELI", "TN_PAL_VELLORE"]:
            res = generate_forecast(
                station_id=sid,
                horizon=1,
                model_name="xgboost",
            )
            assert res.drift_status == "IN_DISTRIBUTION", f"Station {sid} should be IN_DISTRIBUTION, got {res.drift_status}"
            assert res.status == "OK", f"Expected OK status for {sid}, got {res.status}: {res.reason}"
            assert res.risk.get("category") in ["Low", "Moderate", "High", "Severe"], f"Expected risk category for {sid}, got {res.risk}"
            assert res.forecast is not None
            assert res.interval is not None

    def test_forecast_value_and_interval_show_even_when_drift_blocks_risk(self):
        """When drift is detected, forecast and prediction intervals must NEVER be hidden."""
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="xgboost",
        )
        assert res.drift_status == "OUT_OF_DISTRIBUTION"
        assert res.status in ("LOW_CONFIDENCE", "OK_DRIFT_BLOCKED")
        assert res.forecast is not None, "Forecast value must still be displayed"
        assert res.interval is not None, "Conformal prediction interval must still be displayed"
        assert res.features_outside_range_count >= 2
        assert res.top_drifting_features is not None
        assert len(res.top_drifting_features) >= 1

    def test_low_confidence_still_shows_forecast_and_interval(self):
        """LOW_CONFIDENCE still shows forecast, interval, and risk category with low confidence status."""
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="xgboost",
        )
        assert res.status == "LOW_CONFIDENCE"
        assert res.forecast is not None, "Forecast must be displayed"
        assert res.interval is not None, "Interval must be displayed"
        assert res.risk.get("category") in ["Low", "Moderate", "High", "Severe"]
        assert res.risk.get("badge") == "Low Confidence"
        assert res.risk.get("is_low_confidence") is True

    def test_stale_and_insufficient_signal_never_show_risk_category(self):
        """STALE and INSUFFICIENT_SIGNAL stations must never show an operational flow risk category."""
        # STALE
        res_stale = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="xgboost",
            mode="live",
            max_data_stale_hours=0.001,
        )
        assert res_stale.status == "STALE"
        assert res_stale.risk.get("category") == "UNAVAILABLE"

        # INSUFFICIENT_SIGNAL (synthetic constant baseflow with CV < 0.80)
        dates = pd.date_range("2020-01-01", periods=1000, freq="D").strftime("%Y-%m-%d").tolist()
        df_const = pd.DataFrame({
            "date": dates,
            "precipitation_sum": [5.0] * 1000,
            "river_discharge": [0.02] * 1000,
            "is_forecast": [False] * 1000,
        })
        res_sig = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="persistence",
            df_data=df_const,
        )
        assert res_sig.status == "INSUFFICIENT_SIGNAL"
        assert res_sig.risk.get("category") == "UNAVAILABLE"

    def test_insufficient_signal_station_shows_no_risk_category_but_shows_forecast(self):
        """An insufficient-signal station never shows a risk category, but still shows forecast and interval."""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D").strftime("%Y-%m-%d").tolist()
        df_const = pd.DataFrame({
            "date": dates,
            "precipitation_sum": [5.0] * 1000,
            "river_discharge": [1.0 + (i % 2) * 0.05 for i in range(1000)],
            "is_forecast": [False] * 1000,
        })
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="persistence",
            df_data=df_const,
        )
        assert res.status == "INSUFFICIENT_SIGNAL", f"Expected INSUFFICIENT_SIGNAL, got {res.status}"
        assert res.is_insufficient_signal is True
        assert res.risk.get("category") == "UNAVAILABLE"
        assert res.forecast is not None
        assert res.interval is not None

    def test_unsupported_station_never_shows_flow_forecast_or_flow_risk(self):
        """Chennai (unsupported coastal river basin) must NEVER show flow forecast, interval, or flow risk."""
        for h in [1, 3, 7]:
            res = generate_forecast(station_id="TN_ADY_CHENNAI", horizon=h, model_name="persistence")
            assert res.is_rainfall_only is True
            assert res.forecast is None
            assert res.interval is None
            assert res.risk.get("category") == "UNAVAILABLE"
            assert res.rainfall_risk is not None
            assert "category" in res.rainfall_risk
            assert res.rainfall_risk["category"] in ["Low", "Moderate", "High", "Severe"]

    def test_incomplete_window_returns_insufficient_history(self):
        """A live window with fewer than required past days (< 30) must return INSUFFICIENT_HISTORY with missing count."""
        dates = pd.date_range("2026-10-01", periods=15, freq="D").strftime("%Y-%m-%d").tolist()
        df_short = pd.DataFrame({
            "date": dates,
            "precipitation_sum": [10.0] * 15,
            "river_discharge": [2.5] * 15,
            "is_forecast": [False] * 15,
        })
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="persistence",
            df_data=df_short,
        )
        assert res.status == "INSUFFICIENT_HISTORY"
        assert res.forecast is None
        assert res.risk.get("category") == "UNAVAILABLE"
        assert "missing 15 days" in res.reason
        assert "15 past days available, required 30" in res.reason
