"""
tests/test_config.py
Phase-1 tests: verify that YAML configs load correctly and contain all
required keys, station ids, and sensible defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# ── make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    get_station,
    load_data_config,
    load_thresholds_config,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def data_cfg():
    return load_data_config()


@pytest.fixture(scope="module")
def thresh_cfg():
    return load_thresholds_config()


# =============================================================================
# data.yaml tests
# =============================================================================


class TestDataConfig:
    def test_loads_without_error(self, data_cfg):
        assert isinstance(data_cfg, dict)

    def test_project_metadata_present(self, data_cfg):
        assert "project" in data_cfg
        assert "name" in data_cfg["project"]
        assert "version" in data_cfg["project"]

    def test_configured_stations(self, data_cfg):
        stations = data_cfg.get("stations", [])
        assert len(stations) >= 3, f"Expected at least 3 stations, got {len(stations)}"

    def test_station_ids_unique(self, data_cfg):
        ids = [s["id"] for s in data_cfg["stations"]]
        assert len(ids) == len(set(ids)), "Duplicate station IDs found"

    def test_required_station_fields(self, data_cfg):
        required = {"id", "name", "river", "latitude", "longitude", "timezone"}
        for station in data_cfg["stations"]:
            missing = required - set(station.keys())
            assert not missing, f"Station {station.get('id')} missing fields: {missing}"

    def test_latitude_range(self, data_cfg):
        for s in data_cfg["stations"]:
            assert 8.0 <= s["latitude"] <= 13.5, (
                f"Latitude {s['latitude']} for {s['id']} outside Tamil Nadu range"
            )

    def test_longitude_range(self, data_cfg):
        for s in data_cfg["stations"]:
            assert 76.0 <= s["longitude"] <= 80.5, (
                f"Longitude {s['longitude']} for {s['id']} outside Tamil Nadu range"
            )

    def test_default_station_exists(self, data_cfg):
        default_id = data_cfg["default_station_id"]
        ids = [s["id"] for s in data_cfg["stations"]]
        assert default_id in ids

    def test_period_start_before_end(self, data_cfg):
        from datetime import date
        start = date.fromisoformat(data_cfg["period"]["start"])
        end = date.fromisoformat(data_cfg["period"]["end"])
        assert start < end

    def test_period_minimum_length(self, data_cfg):
        from datetime import date
        start = date.fromisoformat(data_cfg["period"]["start"])
        end = date.fromisoformat(data_cfg["period"]["end"])
        years = (end - start).days / 365.25
        assert years >= 5, f"Data period {years:.1f} years is too short (need ≥ 5)"

    def test_rainfall_api_keys_present(self, data_cfg):
        api = data_cfg.get("rainfall_api", {})
        for key in ("base_url", "variables", "timeout_s", "max_retries"):
            assert key in api, f"rainfall_api missing key: {key}"

    def test_discharge_api_keys_present(self, data_cfg):
        api = data_cfg.get("discharge_api", {})
        for key in ("base_url", "variables", "timeout_s", "max_retries"):
            assert key in api, f"discharge_api missing key: {key}"

    def test_validation_rules_sensible(self, data_cfg):
        v = data_cfg.get("validation", {})
        assert v["rainfall_min_mm"] >= 0
        assert v["rainfall_max_mm"] > v["rainfall_min_mm"]
        assert v["discharge_min_m3s"] >= 0
        assert v["discharge_max_m3s"] > v["discharge_min_m3s"]

    def test_paths_keys_present(self, data_cfg):
        paths = data_cfg.get("paths", {})
        for key in ("raw_dir", "interim_dir", "processed_dir", "reports_dir"):
            assert key in paths, f"paths missing key: {key}"

    def test_forecast_horizons_in_thresholds(self, thresh_cfg):
        horizons = thresh_cfg.get("forecast_horizons_days", [])
        assert set(horizons) == {1, 3, 7}


# =============================================================================
# thresholds.yaml tests
# =============================================================================


class TestThresholdsConfig:
    def test_loads_without_error(self, thresh_cfg):
        assert isinstance(thresh_cfg, dict)

    def test_metadata_present(self, thresh_cfg):
        m = thresh_cfg.get("metadata", {})
        assert "source" in m
        assert "units" in m
        assert m["units"] == "m³/s"

    def test_source_is_prototype(self, thresh_cfg):
        """Ensure default source is prototype (not official CWC)."""
        assert thresh_cfg["metadata"]["source"] == "prototype_percentile"

    def test_all_station_ids_present(self, thresh_cfg, data_cfg):
        cfg_ids = {s["id"] for s in data_cfg["stations"]}
        thresh_ids = set(thresh_cfg.get("stations", {}).keys())
        assert cfg_ids == thresh_ids, (
            f"Threshold station IDs {thresh_ids} don't match config IDs {cfg_ids}"
        )

    def test_categories_for_each_station(self, thresh_cfg):
        expected = {"Low", "Moderate", "High", "Severe"}
        for sid, sdata in thresh_cfg["stations"].items():
            cats = set(sdata.get("categories", {}).keys())
            assert cats == expected, f"Station {sid} categories {cats} != {expected}"

    def test_low_category_lower_bound_is_zero(self, thresh_cfg):
        for sid, sdata in thresh_cfg["stations"].items():
            low_lb = sdata["categories"]["Low"]["lower_bound_m3s"]
            assert low_lb == 0.0, (
                f"Station {sid} Low lower_bound should be 0.0, got {low_lb}"
            )

    def test_staleness_policy_present(self, thresh_cfg):
        s = thresh_cfg.get("staleness", {})
        assert "max_data_age_hours" in s
        assert "max_forecast_age_hours" in s
        assert s["max_data_age_hours"] > 0


# =============================================================================
# get_station() helper tests
# =============================================================================


class TestGetStation:
    def test_returns_default_station(self, data_cfg):
        station = get_station(data_cfg)
        assert station["id"] == data_cfg["default_station_id"]

    def test_returns_named_station(self, data_cfg):
        station = get_station(data_cfg, "TN_CAU_METTUR")
        assert station["id"] == "TN_CAU_METTUR"

    def test_raises_on_unknown_station(self, data_cfg):
        with pytest.raises(KeyError, match="NOT_EXIST"):
            get_station(data_cfg, "NOT_EXIST")
