"""
tests/test_features.py
──────────────────────
Phase-4 & Phase-13 unit and leakage tests:
  1. Hand-computed unit tests on small synthetic series:
     • Lags (1, 2, 3, 7, 14)
     • Cumulative rainfall (3, 7, 14 days)
     • Rolling mean & rolling max (7, 14, 30 days)
     • Rate of change (1, 3, 7 days)
     • Cyclical month encoding (sin/cos on unit circle)
     • Phase 13: Antecedent Precipitation Index (API decay)
     • Phase 13: Tamil Nadu Monsoon flags (monsoon_sw, monsoon_ne)
     • Phase 13: Days-since-heavy-rain
     • Phase 13: Wet-spell length
     • Phase 13: Catchment-areal rainfall weighted average
  2. Automated leakage tests:
     • Modifying future values (at t+1 or later) has ZERO effect on any feature at time t.
     • Validating get_feature_columns excludes raw target columns and target labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.temporal_features import (
    add_lag_features,
    add_cumulative_rainfall,
    add_rolling_mean,
    add_rolling_max,
    add_rate_of_change,
    add_cyclical_month,
    add_antecedent_precipitation_index,
    add_monsoon_flags,
    add_days_since_heavy_rain,
    add_wet_spell_length,
    add_catchment_areal_rainfall,
    add_targets,
    build_feature_matrix,
    get_feature_columns,
    get_target_column,
)


@pytest.fixture
def small_df():
    """Small hand-computed DataFrame (10 daily rows)."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D").strftime("%Y-%m-%d").tolist()
    rain = [0.0, 10.0, 20.0, 30.0, 0.0, 50.0, 0.0, 10.0, 0.0, 20.0]
    disc = [100.0, 110.0, 130.0, 160.0, 150.0, 200.0, 190.0, 180.0, 170.0, 160.0]
    return pd.DataFrame({
        "date": dates,
        "precipitation_sum": rain,
        "river_discharge": disc,
    })


class TestHandComputedFeatures:
    def test_lags_hand_computed(self, small_df):
        out = add_lag_features(small_df, "precipitation_sum", lags=[1, 2])
        assert pd.isna(out.loc[0, "precipitation_sum_lag_1"])
        assert out.loc[1, "precipitation_sum_lag_1"] == pytest.approx(0.0)
        assert out.loc[2, "precipitation_sum_lag_1"] == pytest.approx(10.0)
        assert out.loc[2, "precipitation_sum_lag_2"] == pytest.approx(0.0)

    def test_cumulative_rainfall_hand_computed(self, small_df):
        out = add_cumulative_rainfall(small_df, col="precipitation_sum", windows=[3])
        assert out.loc[0, "precipitation_sum_cumul_3d"] == pytest.approx(0.0)
        assert out.loc[2, "precipitation_sum_cumul_3d"] == pytest.approx(30.0)
        assert out.loc[3, "precipitation_sum_cumul_3d"] == pytest.approx(60.0)

    def test_rolling_mean_hand_computed(self, small_df):
        out = add_rolling_mean(small_df, col="precipitation_sum", windows=[3])
        assert out.loc[2, "precipitation_sum_roll_mean_3d"] == pytest.approx(10.0)
        assert out.loc[3, "precipitation_sum_roll_mean_3d"] == pytest.approx(20.0)

    def test_rolling_max_hand_computed(self, small_df):
        out = add_rolling_max(small_df, col="precipitation_sum", windows=[3])
        assert out.loc[2, "precipitation_sum_roll_max_3d"] == pytest.approx(20.0)
        assert out.loc[4, "precipitation_sum_roll_max_3d"] == pytest.approx(30.0)

    def test_rate_of_change_hand_computed(self, small_df):
        out = add_rate_of_change(small_df, col="river_discharge", lags=[1, 2])
        assert pd.isna(out.loc[0, "river_discharge_roc_1d"])
        assert out.loc[1, "river_discharge_roc_1d"] == pytest.approx(10.0)
        assert out.loc[2, "river_discharge_roc_1d"] == pytest.approx(20.0)
        assert out.loc[2, "river_discharge_roc_2d"] == pytest.approx(30.0)

    def test_cyclical_month(self):
        df = pd.DataFrame({"date": ["2023-01-15", "2023-06-15", "2023-12-15"]})
        out = add_cyclical_month(df)
        assert out.loc[0, "month_sin"] == pytest.approx(np.sin(2 * np.pi / 12), abs=1e-4)
        assert (out.loc[0, "month_sin"] ** 2 + out.loc[0, "month_cos"] ** 2) == pytest.approx(1.0, abs=1e-5)

    def test_api_decay_hand_computed(self):
        # rain: [10.0, 0.0, 20.0]
        # decay = 0.9
        # API[0] = 10.0
        # API[1] = 0.9 * 10 + 0 = 9.0
        # API[2] = 0.9 * 9.0 + 20.0 = 28.1
        df = pd.DataFrame({"precipitation_sum": [10.0, 0.0, 20.0]})
        out = add_antecedent_precipitation_index(df, decay_factors=[0.90])
        assert out.loc[0, "api_decay_90"] == pytest.approx(10.0)
        assert out.loc[1, "api_decay_90"] == pytest.approx(9.0)
        assert out.loc[2, "api_decay_90"] == pytest.approx(28.1)

    def test_monsoon_flags(self):
        df = pd.DataFrame({"date": ["2023-07-15", "2023-11-15", "2023-02-15"]})
        out = add_monsoon_flags(df)
        # July: SW Monsoon
        assert out.loc[0, "monsoon_sw"] == 1.0
        assert out.loc[0, "monsoon_ne"] == 0.0
        # November: NE Monsoon
        assert out.loc[1, "monsoon_sw"] == 0.0
        assert out.loc[1, "monsoon_ne"] == 1.0
        # February: Dry season
        assert out.loc[2, "monsoon_sw"] == 0.0
        assert out.loc[2, "monsoon_ne"] == 0.0

    def test_days_since_heavy_rain_hand_computed(self):
        # rain: [0.0, 40.0, 10.0, 0.0, 50.0] (threshold = 35.5)
        # t=0: dry (count=1000)
        # t=1: heavy rain -> 0
        # t=2: moderate -> 1
        # t=3: dry -> 2
        # t=4: heavy rain -> 0
        df = pd.DataFrame({"precipitation_sum": [0.0, 40.0, 10.0, 0.0, 50.0]})
        out = add_days_since_heavy_rain(df, threshold_mm=35.5)
        assert out.loc[1, "days_since_heavy_rain"] == 0.0
        assert out.loc[2, "days_since_heavy_rain"] == 1.0
        assert out.loc[3, "days_since_heavy_rain"] == 2.0
        assert out.loc[4, "days_since_heavy_rain"] == 0.0

    def test_wet_spell_length_hand_computed(self):
        # rain: [0.0, 5.0, 10.0, 2.0, 0.0, 1.0] (threshold = 1.0)
        # t=0: 0
        # t=1: 1
        # t=2: 2
        # t=3: 3
        # t=4: 0
        # t=5: 1
        df = pd.DataFrame({"precipitation_sum": [0.0, 5.0, 10.0, 2.0, 0.0, 1.0]})
        out = add_wet_spell_length(df, threshold_mm=1.0)
        assert out.loc[0, "wet_spell_length_days"] == 0.0
        assert out.loc[1, "wet_spell_length_days"] == 1.0
        assert out.loc[2, "wet_spell_length_days"] == 2.0
        assert out.loc[3, "wet_spell_length_days"] == 3.0
        assert out.loc[4, "wet_spell_length_days"] == 0.0
        assert out.loc[5, "wet_spell_length_days"] == 1.0

    def test_catchment_areal_rainfall_hand_computed(self):
        df = pd.DataFrame({
            "stn1_rain": [10.0, 20.0],
            "stn2_rain": [30.0, 40.0],
        })
        out = add_catchment_areal_rainfall(df, ["stn1_rain", "stn2_rain"], weights=[0.25, 0.75])
        # t=0: 0.25*10 + 0.75*30 = 2.5 + 22.5 = 25.0
        assert out.loc[0, "catchment_areal_rain_mm"] == pytest.approx(25.0)


class TestLeakage:
    def test_feature_causality_leakage(self):
        """Verify that mutating future data (t >= cutoff) does NOT alter any feature at t < cutoff."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist()
        df_base = pd.DataFrame({
            "date": dates,
            "precipitation_sum": np.random.uniform(0, 50, n),
            "river_discharge": np.random.uniform(50, 500, n),
        })

        out_base = build_feature_matrix(df_base, horizons=[1, 3], include_rich_features=True)
        feat_cols = get_feature_columns(out_base, horizons=[1, 3])

        cutoff = 25
        df_perturbed = df_base.copy()
        df_perturbed.loc[cutoff:, "precipitation_sum"] += 1000.0
        df_perturbed.loc[cutoff:, "river_discharge"] += 50000.0

        out_perturbed = build_feature_matrix(df_perturbed, horizons=[1, 3], include_rich_features=True)

        for t in range(cutoff - 3):  # account for tail drop
            for col in feat_cols:
                v_orig = out_base.loc[t, col]
                v_pert = out_perturbed.loc[t, col]
                if pd.isna(v_orig):
                    assert pd.isna(v_pert)
                else:
                    assert v_orig == pytest.approx(v_pert), (
                        f"Leakage detected in feature {col} at t={t}! "
                        f"Orig: {v_orig}, Perturbed: {v_pert}"
                    )

    def test_column_exclusion(self, small_df):
        mat = build_feature_matrix(small_df, horizons=[1, 3, 7])
        feats = get_feature_columns(mat, horizons=[1, 3, 7])

        for h in [1, 3, 7]:
            assert f"target_discharge_h{h}d" not in feats

        assert "date" not in feats
        assert "precipitation_sum" not in feats
        assert "river_discharge" not in feats
        assert "api_decay_90" in feats
        assert "monsoon_sw" in feats
        assert "monsoon_ne" in feats
        assert "days_since_heavy_rain" in feats
        assert "wet_spell_length_days" in feats
        assert "river_discharge_roll_std_7d" in feats
        assert "precip_max_3d" in feats
        assert "days_since_significant_rain" in feats

    def test_new_features_hand_computed(self, small_df):
        from src.features.temporal_features import (
            add_rolling_std,
            add_rainfall_intensity,
            add_days_since_significant_rain,
        )
        df_std = add_rolling_std(small_df, "river_discharge", windows=[3])
        assert df_std.loc[0, "river_discharge_roll_std_3d"] == pytest.approx(0.0)
        assert df_std.loc[2, "river_discharge_roll_std_3d"] > 0.0

        df_intensity = add_rainfall_intensity(small_df, "precipitation_sum", windows=[3])
        # rain: [0.0, 10.0, 20.0, 30.0, ...]
        assert df_intensity.loc[2, "precip_max_3d"] == pytest.approx(20.0)
        assert df_intensity.loc[3, "precip_max_3d"] == pytest.approx(30.0)

        df_since = add_days_since_significant_rain(small_df, "precipitation_sum", threshold_mm=5.0)
        assert df_since.loc[0, "days_since_significant_rain"] == pytest.approx(999.0)
        assert df_since.loc[1, "days_since_significant_rain"] == pytest.approx(0.0)  # rain=10.0 >= 5.0
        assert df_since.loc[4, "days_since_significant_rain"] == pytest.approx(1.0)  # rain=0.0
