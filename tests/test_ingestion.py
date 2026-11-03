"""
tests/test_ingestion.py
───────────────────────
Phase-2 tests covering:
  • Successful API responses parsed correctly
  • HTTP timeout → IngestionError after max_retries
  • HTTP 429   → pause + retry → success
  • Bad JSON   → IngestionError
  • Offline (ConnectionError) → IngestionError
  • Missing JSON keys → IngestionError
  • save_raw writes untouched file + meta sidecar
  • save_raw is idempotent (second call skips write)
  • validate_dataframe: missing values, duplicates, range flags, report written
  • Extreme values are FLAGGED, not removed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
import requests

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import (
    IngestionError,
    _get_with_retry,
    fetch_discharge,
    fetch_rainfall,
    save_raw,
)
from src.data.validation import validate_dataframe
from src.utils.config import load_data_config

# =============================================================================
# Shared fixtures
# =============================================================================

STATION = {
    "id": "TN_CAU_TRICHY",
    "name": "Test Station",
    "latitude": 10.79,
    "longitude": 78.70,
    "timezone": "Asia/Kolkata",
}

MINIMAL_API = {
    "base_url": "https://example.com/api",
    "variables": ["precipitation_sum"],
    "timeout_s": 5,
    "max_retries": 2,
    "backoff_factor": 1,
    "rate_limit_pause_s": 1,
}

RAINFALL_JSON = {
    "daily": {
        "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "precipitation_sum": [5.0, 12.3, 0.0],
    }
}

DISCHARGE_JSON = {
    "daily": {
        "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "river_discharge": [100.0, 150.5, 90.2],
    }
}


def _make_response(json_data=None, status_code=200, raise_for=None):
    """Build a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = json.dumps(json_data) if json_data else "error"
    if json_data is not None:
        mock.json.return_value = json_data
    else:
        mock.json.side_effect = ValueError("bad json")
    if raise_for:
        mock.raise_for_status.side_effect = raise_for
    return mock


@pytest.fixture(scope="module")
def data_cfg():
    return load_data_config()


# =============================================================================
# _get_with_retry tests
# =============================================================================


class TestGetWithRetry:
    """Test the low-level HTTP retry helper in isolation."""

    def _call(self, side_effects, **kwargs):
        defaults = dict(
            url="https://example.com",
            params={},
            timeout_s=1,
            max_retries=2,
            backoff_factor=1,
            rate_limit_pause_s=1,
        )
        defaults.update(kwargs)
        with patch("requests.get", side_effect=side_effects) as mock_get:
            with patch("time.sleep"):  # don't actually sleep in tests
                result = _get_with_retry(**defaults)
        return result, mock_get

    def test_success_on_first_attempt(self):
        resp = _make_response({"key": "value"})
        result, mock_get = self._call([resp])
        assert result == {"key": "value"}
        assert mock_get.call_count == 1

    def test_timeout_then_success(self):
        """One timeout → should retry and succeed."""
        good = _make_response({"ok": True})
        result, mock_get = self._call(
            [requests.exceptions.Timeout, good]
        )
        assert result == {"ok": True}
        assert mock_get.call_count == 2

    def test_all_timeouts_raise_ingestion_error(self):
        """max_retries=2 means 3 total attempts; all timeout → IngestionError."""
        with patch("requests.get", side_effect=requests.exceptions.Timeout):
            with patch("time.sleep"):
                with pytest.raises(IngestionError, match="attempts failed"):
                    _get_with_retry(
                        url="https://x.com", params={},
                        timeout_s=1, max_retries=2,
                        backoff_factor=1, rate_limit_pause_s=1,
                    )

    def test_connection_error_raises_ingestion_error(self):
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("offline")):
            with patch("time.sleep"):
                with pytest.raises(IngestionError, match="attempts failed"):
                    _get_with_retry(
                        url="https://x.com", params={},
                        timeout_s=1, max_retries=1,
                        backoff_factor=1, rate_limit_pause_s=1,
                    )

    def test_http_429_retries_with_pause(self):
        """429 → sleep rate_limit_pause_s → retry and succeed."""
        rate_limited = _make_response(None, status_code=429)
        good = _make_response({"data": 1})
        with patch("requests.get", side_effect=[rate_limited, good]):
            with patch("time.sleep") as mock_sleep:
                result = _get_with_retry(
                    url="https://x.com", params={},
                    timeout_s=1, max_retries=2,
                    backoff_factor=1, rate_limit_pause_s=65,
                )
        assert result == {"data": 1}
        # Must have slept the rate-limit duration
        mock_sleep.assert_any_call(65)

    def test_bad_json_raises_ingestion_error(self):
        bad_resp = _make_response(None, status_code=200)  # json() raises ValueError
        with patch("requests.get", return_value=bad_resp):
            with pytest.raises(IngestionError, match="Invalid JSON"):
                _get_with_retry(
                    url="https://x.com", params={},
                    timeout_s=1, max_retries=0,
                    backoff_factor=1, rate_limit_pause_s=1,
                )

    def test_non_200_non_429_raises_ingestion_error(self):
        bad_resp = _make_response(None, status_code=500)
        with patch("requests.get", return_value=bad_resp):
            with pytest.raises(IngestionError, match="HTTP 500"):
                _get_with_retry(
                    url="https://x.com", params={},
                    timeout_s=1, max_retries=0,
                    backoff_factor=1, rate_limit_pause_s=1,
                )

    def test_empty_json_raises_ingestion_error(self):
        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.json.return_value = {}
        empty_resp.text = "{}"
        with patch("requests.get", return_value=empty_resp):
            with pytest.raises(IngestionError, match="Empty JSON"):
                _get_with_retry(
                    url="https://x.com", params={},
                    timeout_s=1, max_retries=0,
                    backoff_factor=1, rate_limit_pause_s=1,
                )


# =============================================================================
# fetch_rainfall tests
# =============================================================================


class TestFetchRainfall:
    def _cfg(self):
        return {
            "rainfall_api": {**MINIMAL_API, "variables": ["precipitation_sum"]},
            "period": {"start": "2023-01-01", "end": "2023-01-03"},
        }

    def test_returns_dataframe_with_correct_columns(self):
        with patch("src.data.ingestion._get_with_retry", return_value=RAINFALL_JSON):
            df = fetch_rainfall(STATION, self._cfg())
        assert list(df.columns) == ["date", "precipitation_sum"]

    def test_returns_correct_row_count(self):
        with patch("src.data.ingestion._get_with_retry", return_value=RAINFALL_JSON):
            df = fetch_rainfall(STATION, self._cfg())
        assert len(df) == 3

    def test_dates_are_sorted(self):
        shuffled = {
            "daily": {
                "time": ["2023-01-03", "2023-01-01", "2023-01-02"],
                "precipitation_sum": [0.0, 5.0, 12.3],
            }
        }
        with patch("src.data.ingestion._get_with_retry", return_value=shuffled):
            df = fetch_rainfall(STATION, self._cfg())
        assert list(df["date"]) == ["2023-01-01", "2023-01-02", "2023-01-03"]

    def test_missing_daily_key_raises(self):
        with patch("src.data.ingestion._get_with_retry", return_value={"x": 1}):
            with pytest.raises(IngestionError, match="Missing keys"):
                fetch_rainfall(STATION, self._cfg())

    def test_missing_precipitation_key_raises(self):
        bad = {"daily": {"time": ["2023-01-01"]}}
        with patch("src.data.ingestion._get_with_retry", return_value=bad):
            with pytest.raises(IngestionError, match="Missing keys"):
                fetch_rainfall(STATION, self._cfg())


# =============================================================================
# fetch_discharge tests
# =============================================================================


class TestFetchDischarge:
    def _cfg(self):
        return {
            "discharge_api": {**MINIMAL_API, "variables": ["river_discharge"]},
            "period": {"start": "2023-01-01", "end": "2023-01-03"},
        }

    def test_returns_dataframe_with_correct_columns(self):
        with patch("src.data.ingestion._get_with_retry", return_value=DISCHARGE_JSON):
            df = fetch_discharge(STATION, self._cfg())
        assert list(df.columns) == ["date", "river_discharge"]

    def test_values_are_numeric(self):
        with patch("src.data.ingestion._get_with_retry", return_value=DISCHARGE_JSON):
            df = fetch_discharge(STATION, self._cfg())
        assert pd.api.types.is_numeric_dtype(df["river_discharge"])

    def test_missing_river_discharge_key_raises(self):
        bad = {"daily": {"time": ["2023-01-01"]}}
        with patch("src.data.ingestion._get_with_retry", return_value=bad):
            with pytest.raises(IngestionError, match="Missing keys"):
                fetch_discharge(STATION, self._cfg())


# =============================================================================
# save_raw tests
# =============================================================================


class TestSaveRaw:
    def _cfg(self, tmp_path):
        return {
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "filename_templates": {
                "raw_rainfall": "{station_id}_rainfall_raw_{date}.csv",
            },
            "retrieval": {"source_credit": "Open-Meteo.com (CC BY 4.0)"},
        }

    def _df(self):
        return pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "precipitation_sum": [5.0, 12.3],
        })

    def test_csv_file_created(self, tmp_path):
        cfg = self._cfg(tmp_path)
        path = save_raw(self._df(), "TN_CAU_TRICHY", "rainfall", cfg,
                        retrieval_date="2023-06-01")
        assert path.exists()

    def test_csv_content_matches_dataframe(self, tmp_path):
        cfg = self._cfg(tmp_path)
        df = self._df()
        path = save_raw(df, "TN_CAU_TRICHY", "rainfall", cfg,
                        retrieval_date="2023-06-01")
        df_read = pd.read_csv(path)
        assert list(df_read.columns) == list(df.columns)
        assert len(df_read) == len(df)

    def test_meta_sidecar_created(self, tmp_path):
        cfg = self._cfg(tmp_path)
        path = save_raw(self._df(), "TN_CAU_TRICHY", "rainfall", cfg,
                        retrieval_date="2023-06-01")
        meta_path = path.with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["station_id"] == "TN_CAU_TRICHY"
        assert meta["dataset"] == "rainfall"
        assert meta["rows"] == 2

    def test_second_call_does_not_overwrite(self, tmp_path):
        """Raw files must be immutable; second save skips silently."""
        cfg = self._cfg(tmp_path)
        df1 = self._df()
        path = save_raw(df1, "TN_CAU_TRICHY", "rainfall", cfg,
                        retrieval_date="2023-06-01")
        original_mtime = path.stat().st_mtime

        df2 = pd.DataFrame({
            "date": ["2024-01-01"],
            "precipitation_sum": [99.0],
        })
        save_raw(df2, "TN_CAU_TRICHY", "rainfall", cfg,
                 retrieval_date="2023-06-01")  # same date → same filename

        assert path.stat().st_mtime == original_mtime  # file unchanged

    def test_filename_contains_station_and_date(self, tmp_path):
        cfg = self._cfg(tmp_path)
        path = save_raw(self._df(), "TN_CAU_TRICHY", "rainfall", cfg,
                        retrieval_date="2024-03-15")
        assert "TN_CAU_TRICHY" in path.name
        assert "2024-03-15" in path.name


# =============================================================================
# validate_dataframe tests
# =============================================================================


class TestValidateDataframe:
    def _cfg(self, tmp_path):
        return {
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "filename_templates": {
                "validation_report": "{station_id}_{dataset}_validation_{date}.json",
            },
            "period": {"start": "2023-01-01", "end": "2023-01-05"},
            "validation": {
                "rainfall_min_mm": 0.0,
                "rainfall_max_mm": 600.0,
                "discharge_min_m3s": 0.0,
                "discharge_max_m3s": 50000.0,
                "max_missing_pct": 10.0,
                "max_consecutive_missing": 2,
            },
        }

    def _make_df(self, values, dates=None):
        if dates is None:
            dates = [f"2023-01-0{i+1}" for i in range(len(values))]
        return pd.DataFrame({"date": dates, "precipitation_sum": values})

    def test_clean_data_passes(self, tmp_path):
        cfg = self._cfg(tmp_path)
        df = self._make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        assert report["passed"] is True
        assert report["issues"] == []

    def test_missing_values_counted(self, tmp_path):
        cfg = self._cfg(tmp_path)
        df = self._make_df([1.0, None, None, 4.0, 5.0])
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        assert report["missing_values"]["count"] == 2

    def test_high_missing_triggers_warning(self, tmp_path):
        cfg = self._cfg(tmp_path)
        # 3/5 = 60% missing > 10% threshold
        df = self._make_df([None, None, None, 4.0, 5.0])
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        codes = [w["code"] for w in report["warnings"]]
        assert "high_missing_pct" in codes

    def test_duplicate_timestamps_flagged(self, tmp_path):
        cfg = self._cfg(tmp_path)
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "precipitation_sum": [5.0, 6.0, 7.0],
        })
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        codes = [w["code"] for w in report["warnings"]]
        assert "duplicate_timestamps" in codes
        assert report["duplicate_timestamps"] == 2

    def test_extreme_values_flagged_not_removed(self, tmp_path):
        """Extreme values must be flagged in the report but the df is unchanged."""
        cfg = self._cfg(tmp_path)
        extreme_val = 650.0  # above 600 mm/day threshold
        df = self._make_df([1.0, extreme_val, 3.0, 4.0, 5.0])
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        # Must be flagged
        codes = [w["code"] for w in report["warnings"]]
        assert "values_above_maximum" in codes
        # Must still be 5 rows (not dropped)
        assert report["total_rows"] == 5

    def test_below_minimum_flagged(self, tmp_path):
        cfg = self._cfg(tmp_path)
        df = self._make_df([-1.0, 2.0, 3.0, 4.0, 5.0])
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        codes = [w["code"] for w in report["warnings"]]
        assert "values_below_minimum" in codes

    def test_summary_stats_present(self, tmp_path):
        cfg = self._cfg(tmp_path)
        df = self._make_df([10.0, 20.0, 30.0, 40.0, 50.0])
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        s = report["summary"]
        assert s["min"] == pytest.approx(10.0)
        assert s["max"] == pytest.approx(50.0)
        assert s["mean"] == pytest.approx(30.0)

    def test_report_json_written_to_disk(self, tmp_path):
        cfg = self._cfg(tmp_path)
        df = self._make_df([1.0, 2.0, 3.0])
        validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                           write_report=True, retrieval_date="2023-06-01")
        report_dir = tmp_path / "reports"
        json_files = list(report_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["station_id"] == "TN_CAU_TRICHY"

    def test_consecutive_missing_exceeding_threshold_warns(self, tmp_path):
        cfg = self._cfg(tmp_path)  # max_consecutive_missing = 2
        # 3 consecutive → should warn
        df = self._make_df([1.0, None, None, None, 5.0])
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        codes = [w["code"] for w in report["warnings"]]
        assert "consecutive_missing" in codes

    def test_empty_dataframe_flagged(self, tmp_path):
        cfg = self._cfg(tmp_path)
        df = pd.DataFrame({"date": [], "precipitation_sum": []})
        report = validate_dataframe(df, "rainfall", "TN_CAU_TRICHY", cfg,
                                    write_report=False, retrieval_date="2023-06-01")
        codes = [i["code"] for i in report["issues"]]
        assert "empty_dataframe" in codes
        assert report["passed"] is False
