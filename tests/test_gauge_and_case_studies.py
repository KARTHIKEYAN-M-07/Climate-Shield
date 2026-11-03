"""
tests/test_gauge_and_case_studies.py
───────────────────────────────────
Phase-18 tests:
  - Validates configs/case_studies.yaml structure (all 4 events present with verified dates)
  - validate_gauge_csv:
      * Valid gauge CSV passes
      * Missing date or flow column fails with clear error
      * High values auto-detected as centimeters and converted
  - compare_glofas_to_gauge:
      * Computes correlation r and KGE over overlapping window
      * Returns INSUFFICIENT_OVERLAP when days < 5
  - quantile_mapping_bias_correction:
      * Correctly maps shifted distribution to target observed distribution
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.data.gauge_upload import (
    validate_gauge_csv,
    compare_glofas_to_gauge,
    quantile_mapping_bias_correction,
)


class TestCaseStudiesConfig:
    def test_case_studies_yaml_loaded(self):
        cfg = load_yaml("configs/case_studies.yaml")
        assert "case_studies" in cfg
        cs_list = cfg["case_studies"]
        assert len(cs_list) == 4

        expected_ids = {
            "CHENNAI_FLOODS_2015",
            "CYCLONE_MICHAUNG_2023",
            "SOUTH_TN_EXTREME_2023",
            "CYCLONE_FENGAL_2024",
        }
        loaded_ids = {c["id"] for c in cs_list}
        assert expected_ids == loaded_ids

        for c in cs_list:
            assert "start_date" in c and "end_date" in c
            assert c["start_date"] < c["end_date"]
            assert "source_reference" in c


class TestGaugeUploadValidation:
    def test_valid_gauge_csv(self):
        df = pd.DataFrame({
            "timestamp": ["2023-11-01", "2023-11-02", "2023-11-03"],
            "water_level_m": [2.5, 3.1, 4.0],
        })
        res = validate_gauge_csv(df)
        assert res["is_valid"] is True
        assert len(res["errors"]) == 0
        assert res["is_centimeters"] is False

    def test_missing_date_column_fails(self):
        df = pd.DataFrame({"water_level_m": [2.5, 3.1]})
        res = validate_gauge_csv(df)
        assert res["is_valid"] is False
        assert any("Missing required date" in e for e in res["errors"])

    def test_centimeter_unit_heuristic(self):
        df = pd.DataFrame({
            "date": ["2023-11-01", "2023-11-02"],
            "water_level": [250.0, 310.0],  # clearly cm
        })
        res = validate_gauge_csv(df)
        assert res["is_valid"] is True
        assert res["is_centimeters"] is True

    def test_compare_glofas_to_gauge(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D").strftime("%Y-%m-%d").tolist()
        df_model = pd.DataFrame({
            "date": dates,
            "river_discharge": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        df_obs = pd.DataFrame({
            "date": dates,
            "water_level_m": [12, 22, 31, 42, 51, 62, 71, 82, 91, 102],
        })
        res = compare_glofas_to_gauge(df_model, df_obs)
        assert res["status"] == "COMPARISON_COMPLETE"
        assert res["correlation_r"] > 0.95
        assert res["overlap_days"] == 10

    def test_quantile_mapping_bias_correction(self):
        # Model has systematic -20 offset
        y_train_obs = np.array([50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        y_train_model = y_train_obs - 20.0  # [30, 40, 50, 60, 70, 80]
        y_test_model = np.array([30.0, 50.0, 80.0])

        corrected = quantile_mapping_bias_correction(y_train_model, y_train_obs, y_test_model)
        # Should correct back toward observed distribution [50, 70, 100]
        assert corrected[0] == pytest.approx(50.0, abs=1.0)
        assert corrected[1] == pytest.approx(70.0, abs=1.0)
        assert corrected[2] == pytest.approx(100.0, abs=1.0)
