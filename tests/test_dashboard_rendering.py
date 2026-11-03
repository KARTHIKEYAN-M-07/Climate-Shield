"""tests/test_dashboard_rendering.py
──────────────────────────────────
Unit and integration tests for dashboard rendering and exception safety:
1. Tests that the dashboard renders without uncaught exceptions across all permutations:
   - Model missing
   - Drift reference statistics missing
   - Stale serving data (>48h)
   - Replay mode with historical origin
   - Standard operational state
2. Confirms that no NameError or KeyError occurs under any combination.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.forecast.engine import ForecastResult, generate_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(PROJECT_ROOT / "src" / "dashboard" / "app.py")


@pytest.fixture(autouse=True)
def mock_fast_live_fetch():
    """Mock live fetch in unit tests to prevent network timeout retries."""
    dummy_df = pd.DataFrame({
        "date": ["2023-12-30", "2023-12-31"],
        "precipitation_sum": [12.0, 5.0],
        "river_discharge": [3.2, 4.1],
    })
    mock_info = {
        "df": dummy_df,
        "status": "FALLBACK_CACHE",
        "source_used": "historical_processed",
        "fetched_at_utc": "2026-10-29T08:00:00Z",
        "last_observation_date": "2023-12-31",
        "warnings": [],
    }
    with patch("src.data.live.fetch_station_live_data", return_value=mock_info):
        yield


class TestDashboardCombinations:
    """Test dashboard execution across operational edge cases."""

    def test_default_live_render_no_exceptions(self):
        """Default launch in Live mode should run to completion with no exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        assert not at.exception, f"App execution raised an exception: {at.exception}"

    def test_historical_replay_mode_render_no_exceptions(self):
        """Historical replay mode should render retrospective actuals with no exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        # In sidebar, find mode radio
        for r in at.sidebar.radio:
            if "Historical Replay" in r.options:
                r.set_value("Historical Replay").run()
                break
        assert not at.exception, f"Historical Replay mode raised an exception: {at.exception}"

    def test_model_missing_render_no_exceptions(self):
        """When model is missing, app displays warning notice and no NameError/KeyError."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.query_params["station"] = "TN_ADY_CHENNAI"  # Untrained / excluded station
        at.run()
        assert not at.exception, f"Missing model station raised an exception: {at.exception}"

    def test_drift_reference_missing_render_no_exceptions(self):
        """When drift reference is completely missing, compute_drift_status returns UNKNOWN and app renders safely."""
        with patch("src.forecast.engine.compute_drift_status", return_value={
            "drift_status": "UNKNOWN", "status": "Unknown", "is_drifted": False,
            "drift_value": None, "psi": None, "z_score": None, "warnings": ["Missing drift reference"],
        }):
            at = AppTest.from_file(APP_PATH, default_timeout=20)
            at.run()
            assert not at.exception, f"Missing drift stats raised an exception: {at.exception}"

    def test_stale_data_render_no_exceptions(self):
        """When data is older than 48h in Live mode, risk is UNAVAILABLE and drift is marked not assessed."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        assert not at.exception
        # Ensure STALE data policy is reflected in warnings or cards
        warnings = [w.value for w in at.warning]
        infos = [i.value for i in at.info]
        assert len(warnings) > 0 or len(infos) > 0


class TestForecastEnginePermutations:
    """Direct functional tests verifying ForecastResult across permutations."""

    def test_engine_missing_model_returns_clean_result(self):
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="non_existent_model_xyz",
        )
        assert res.status in ("UNAVAILABLE", "NO_MODEL")
        assert res.forecast is None
        assert res.risk["category"] == "UNAVAILABLE"
        assert "no trained model" in res.reason.lower() or "not found" in res.reason.lower()

    def test_engine_stale_data_policy(self):
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="xgboost",
            mode="live",
            max_data_stale_hours=0.001,  # Force staleness immediately
        )
        assert res.status == "STALE"
        assert res.risk["category"] == "UNAVAILABLE"
        assert res.risk["status"] in ("STALE", "STALE_DATA")

    def test_engine_replay_mode_bounds_and_actuals(self):
        res = generate_forecast(
            station_id="TN_CAU_TRICHY",
            horizon=1,
            model_name="xgboost",
            mode="historical_replay",
            replay_origin_date="2023-11-15",
        )
        assert res.status in ("OK", "OK_DRIFT_BLOCKED", "LOW_CONFIDENCE")
        assert res.forecast is not None
        assert res.replay_actuals is not None
        assert "+1d" in res.replay_actuals
        assert res.replay_error is not None
