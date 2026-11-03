"""
tests/test_grid_snap.py
───────────────────────
Phase-12 tests:
  - Validates configs/stations.yaml schema, required fields, coordinate ranges.
  - Verifies "coordinates_verified_by" and "basin" fields on every station.
  - Tests evaluate_grid_cell:
      * Healthy channel -> viable
      * Near-zero / flat flow -> excluded with "River too small for model resolution"
  - Tests snap_to_best_channel:
      * Snaps to highest dynamic range channel cell
      * Excludes candidate where all 3x3 cells are below threshold
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.grid_snap import evaluate_grid_cell, snap_to_best_channel
from src.utils.config import load_stations_config, load_data_config, load_all_stations


class TestStationsConfig:
    def test_stations_yaml_loads_and_has_required_keys(self):
        cfg = load_stations_config()
        assert "stations" in cfg
        stations = cfg["stations"]
        assert len(stations) >= 5

        required_keys = {
            "id", "name", "river", "basin", "latitude", "longitude",
            "source", "coordinates_verified_by", "status"
        }
        for s in stations:
            for k in required_keys:
                assert k in s, f"Station {s.get('id')} missing required key '{k}'"
            # Tamil Nadu geographic coordinate bounds check
            assert 8.0 <= s["latitude"] <= 13.8, f"Invalid latitude: {s['latitude']}"
            assert 76.0 <= s["longitude"] <= 80.5, f"Invalid longitude: {s['longitude']}"

    def test_load_all_stations_merges_stations(self):
        stns = load_all_stations()
        station_ids = [s["id"] for s in stns]
        # Must include both original Cauvery stations and added candidate basins
        assert "TN_CAU_TRICHY" in station_ids
        assert "TN_VAI_MADURAI" in station_ids
        assert "TN_THA_TIRUNELVELI" in station_ids


class TestGridSnapLogic:
    def test_evaluate_grid_cell_viable(self):
        series = np.array([10.0, 50.0, 120.0, 30.0, 80.0])
        res = evaluate_grid_cell(series, min_mean=1.0, min_max=5.0)
        assert res["is_viable"] is True
        assert res["mean"] > 1.0

    def test_evaluate_grid_cell_excluded_too_small(self):
        # Series with near zero and flat discharge
        series = np.array([0.05, 0.08, 0.02, 0.04, 0.03])
        res = evaluate_grid_cell(series, min_mean=1.0, min_max=5.0)
        assert res["is_viable"] is False
        assert "River too small" in res["reason"]

    def test_snap_to_best_channel_selects_active_cell(self):
        # Mock function: cell (+0.05, 0.0) has high river discharge, while center has low
        station = {
            "id": "TN_TEST_STN",
            "latitude": 10.0,
            "longitude": 78.0,
        }

        def mock_fetch(lat, lon):
            if round(lat, 4) == 10.05 and round(lon, 4) == 78.0:
                # River channel located here
                return pd.DataFrame({"river_discharge": [50.0, 150.0, 200.0, 80.0]})
            else:
                # Non-channel bank cell
                return pd.DataFrame({"river_discharge": [0.2, 0.3, 0.1, 0.2]})

        result = snap_to_best_channel(station, mock_fetch, grid_step_deg=0.05)
        assert result["is_viable"] is True
        assert result["snapped_latitude"] == 10.05
        assert result["snapped_longitude"] == 78.0
        assert "Snapped to north cell" in result["selection_rationale"]

    def test_snap_to_best_channel_excludes_when_all_cells_flat(self):
        station = {
            "id": "TN_TEST_FLAT",
            "latitude": 10.0,
            "longitude": 78.0,
        }

        # All cells return dry or near-zero flow
        def mock_fetch(lat, lon):
            return pd.DataFrame({"river_discharge": [0.1, 0.2, 0.1, 0.0]})

        result = snap_to_best_channel(station, mock_fetch, grid_step_deg=0.05)
        assert result["is_viable"] is False
        assert result["status"] == "excluded"
        assert "River channel too small" in result["exclusion_reason"]
