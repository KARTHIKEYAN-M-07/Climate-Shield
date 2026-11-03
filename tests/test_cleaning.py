"""
tests/test_cleaning.py
──────────────────────
Phase-3 tests: cleaning, alignment, and EDA.

Covers:
  Cleaning
  ─────────
  • Duplicate dates removed (first kept)
  • Full daily reindex introduces NaN for gaps
  • Short gaps (≤ max_consec) forward-filled + flag set
  • Long gaps (> max_consec) left as NaN + gap_flag True
  • Negative rainfall zeroed + neg_flag set
  • Extreme values preserved + extreme_flag set  (NOT removed)
  • Interim CSV written with versioned filename

  Alignment
  ─────────
  • Inner join yields intersection rows only
  • Outer join preserves all dates
  • Mismatched dates reported correctly
  • Aligned CSV and JSON sidecar saved

  EDA
  ───
  • All six plot functions return a valid Path that exists on disk
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import clean_rainfall, clean_discharge
from src.data.alignment import align_datasets, save_processed
from src.data.eda import (
    plot_time_series,
    plot_monthly_box,
    plot_rolling_stats,
    plot_scatter_lag,
    plot_missing_heatmap,
    plot_correlation,
)


# =============================================================================
# Shared helpers
# =============================================================================

def _cfg(tmp_path):
    return {
        "paths": {
            "raw_dir":       str(tmp_path / "raw"),
            "interim_dir":   str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
            "reports_dir":   str(tmp_path / "reports"),
        },
        "validation": {
            "rainfall_min_mm":         0.0,
            "rainfall_max_mm":         600.0,
            "discharge_min_m3s":       0.0,
            "discharge_max_m3s":       50000.0,
            "max_missing_pct":         10.0,
            "max_consecutive_missing": 2,
        },
    }


def _make_rain(dates, values):
    return pd.DataFrame({"date": dates, "precipitation_sum": values})


def _make_disc(dates, values):
    return pd.DataFrame({"date": dates, "river_discharge": values})


def _daily_range(start, n):
    return pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d").tolist()


# =============================================================================
# Cleaning — rainfall
# =============================================================================

class TestCleanRainfall:

    def test_clean_returns_dataframe(self, tmp_path):
        df = _make_rain(_daily_range("2023-01-01", 5), [1.0, 2.0, 3.0, 4.0, 5.0])
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        assert isinstance(result, pd.DataFrame)

    def test_date_column_present(self, tmp_path):
        df = _make_rain(_daily_range("2023-01-01", 5), [1.0, 2.0, 3.0, 4.0, 5.0])
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        assert "date" in result.columns

    def test_duplicate_dates_removed_keep_first(self, tmp_path):
        dates = ["2023-01-01", "2023-01-01", "2023-01-02"]
        values = [10.0, 99.0, 5.0]
        df = _make_rain(dates, values)
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        date_counts = result["date"].astype(str).value_counts()
        assert date_counts.get("2023-01-01", 0) == 1
        # First value (10.0) should be kept
        val = result.loc[result["date"].astype(str) == "2023-01-01",
                         "precipitation_sum"].iloc[0]
        assert val == pytest.approx(10.0)

    def test_full_reindex_fills_missing_dates(self, tmp_path):
        """Gap of 1 day should appear in output with NaN (if too long for ffill)."""
        # max_consec=2, gap=3 days → should remain NaN
        dates = ["2023-01-01", "2023-01-05"]  # 3-day gap in between
        df = _make_rain(dates, [10.0, 20.0])
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        assert len(result) == 5  # 2023-01-01 through 2023-01-05

    def test_short_gap_forward_filled(self, tmp_path):
        """Gap of 1 day ≤ max_consec(2) → forward-filled."""
        dates = ["2023-01-01", "2023-01-03"]
        df = _make_rain(dates, [8.0, 12.0])
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        # Jan 2 should be filled with Jan 1 value (8.0)
        jan2 = result.loc[result["date"].astype(str) == "2023-01-02",
                          "precipitation_sum"]
        assert float(jan2.iloc[0]) == pytest.approx(8.0)
        ffill_flag = result.loc[result["date"].astype(str) == "2023-01-02",
                                "precipitation_sum_ffill_flag"]
        assert bool(ffill_flag.iloc[0]) is True

    def test_long_gap_stays_nan_with_gap_flag(self, tmp_path):
        """Gap of 3 days > max_consec(2) → some NaN rows remain."""
        dates = ["2023-01-01", "2023-01-05"]
        df = _make_rain(dates, [5.0, 15.0])
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        gap_rows = result[result["precipitation_sum_gap_flag"] == True]
        assert len(gap_rows) > 0

    def test_negative_rainfall_zeroed_with_flag(self, tmp_path):
        df = _make_rain(_daily_range("2023-01-01", 3), [-5.0, 2.0, 3.0])
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        neg_rows = result[result["precipitation_sum_neg_flag"] == True]
        assert len(neg_rows) == 1
        assert float(neg_rows["precipitation_sum"].iloc[0]) == pytest.approx(0.0)

    def test_extreme_value_preserved_not_removed(self, tmp_path):
        """Value > 600 mm must stay in df; only extreme_flag set."""
        extreme = 650.0
        df = _make_rain(_daily_range("2023-01-01", 3), [extreme, 2.0, 3.0])
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        assert len(result) == 3
        extreme_rows = result[result["precipitation_sum_extreme_flag"] == True]
        assert len(extreme_rows) == 1
        assert float(extreme_rows["precipitation_sum"].iloc[0]) == pytest.approx(extreme)

    def test_interim_csv_written(self, tmp_path):
        df = _make_rain(_daily_range("2023-01-01", 5), [1.0]*5)
        clean_rainfall(df, _cfg(tmp_path), station_id="ST",
                       retrieval_date="2023-06-01", write_interim=True)
        interim_dir = tmp_path / "interim"
        csvs = list(interim_dir.glob("*.csv"))
        assert len(csvs) == 1

    def test_all_flag_columns_present(self, tmp_path):
        df = _make_rain(_daily_range("2023-01-01", 5), [1.0]*5)
        result = clean_rainfall(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        expected_flags = [
            "precipitation_sum_ffill_flag",
            "precipitation_sum_gap_flag",
            "precipitation_sum_neg_flag",
            "precipitation_sum_extreme_flag",
        ]
        for flag in expected_flags:
            assert flag in result.columns, f"Missing flag column: {flag}"


# =============================================================================
# Cleaning — discharge
# =============================================================================

class TestCleanDischarge:

    def test_clean_returns_dataframe(self, tmp_path):
        df = _make_disc(_daily_range("2023-01-01", 5), [100.0]*5)
        result = clean_discharge(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        assert isinstance(result, pd.DataFrame)

    def test_negative_discharge_zeroed_with_flag(self, tmp_path):
        df = _make_disc(_daily_range("2023-01-01", 3), [-10.0, 200.0, 300.0])
        result = clean_discharge(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        neg_rows = result[result["river_discharge_neg_flag"] == True]
        assert len(neg_rows) == 1
        assert float(neg_rows["river_discharge"].iloc[0]) == pytest.approx(0.0)

    def test_extreme_discharge_preserved(self, tmp_path):
        extreme = 55000.0  # above 50000 threshold
        df = _make_disc(_daily_range("2023-01-01", 3), [extreme, 200.0, 300.0])
        result = clean_discharge(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        extreme_rows = result[result["river_discharge_extreme_flag"] == True]
        assert len(extreme_rows) == 1
        assert float(extreme_rows["river_discharge"].iloc[0]) == pytest.approx(extreme)

    def test_all_flag_columns_present(self, tmp_path):
        df = _make_disc(_daily_range("2023-01-01", 5), [100.0]*5)
        result = clean_discharge(df, _cfg(tmp_path), station_id="ST", write_interim=False)
        for flag in [
            "river_discharge_ffill_flag",
            "river_discharge_gap_flag",
            "river_discharge_neg_flag",
            "river_discharge_extreme_flag",
        ]:
            assert flag in result.columns, f"Missing: {flag}"


# =============================================================================
# Alignment
# =============================================================================

class TestAlignDatasets:

    def _rain(self, dates):
        return pd.DataFrame({"date": dates, "precipitation_sum": [5.0]*len(dates)})

    def _disc(self, dates):
        return pd.DataFrame({"date": dates, "river_discharge": [100.0]*len(dates)})

    def test_inner_join_yields_intersection(self):
        rain = self._rain(["2023-01-01", "2023-01-02", "2023-01-03"])
        disc = self._disc(["2023-01-02", "2023-01-03", "2023-01-04"])
        aligned, report = align_datasets(rain, disc, how="inner")
        assert len(aligned) == 2
        assert set(aligned["date"].astype(str)) == {"2023-01-02", "2023-01-03"}

    def test_outer_join_preserves_all_dates(self):
        rain = self._rain(["2023-01-01", "2023-01-02"])
        disc = self._disc(["2023-01-02", "2023-01-03"])
        aligned, report = align_datasets(rain, disc, how="outer")
        assert len(aligned) == 3

    def test_report_counts_mismatches(self):
        rain = self._rain(["2023-01-01", "2023-01-02", "2023-01-03"])
        disc = self._disc(["2023-01-02", "2023-01-03"])
        _, report = align_datasets(rain, disc, how="inner")
        assert report["dates_only_in_rainfall"] == 1

    def test_sorted_by_date(self):
        rain = self._rain(["2023-01-03", "2023-01-01", "2023-01-02"])
        disc = self._disc(["2023-01-01", "2023-01-02", "2023-01-03"])
        aligned, _ = align_datasets(rain, disc)
        dates = aligned["date"].astype(str).tolist()
        assert dates == sorted(dates)

    def test_both_columns_present(self):
        rain = self._rain(["2023-01-01", "2023-01-02"])
        disc = self._disc(["2023-01-01", "2023-01-02"])
        aligned, _ = align_datasets(rain, disc)
        assert "precipitation_sum" in aligned.columns
        assert "river_discharge" in aligned.columns

    def test_report_has_discharge_label(self):
        rain = self._rain(["2023-01-01"])
        disc = self._disc(["2023-01-01"])
        _, report = align_datasets(rain, disc)
        assert "modelled river discharge" in report["discharge_label"]

    def test_save_processed_writes_csv_and_json(self, tmp_path):
        rain = self._rain(["2023-01-01", "2023-01-02"])
        disc = self._disc(["2023-01-01", "2023-01-02"])
        aligned, report = align_datasets(rain, disc)
        cfg = {
            "paths": {"processed_dir": str(tmp_path / "processed")},
        }
        path = save_processed(aligned, "ST", cfg,
                              retrieval_date="2023-06-01",
                              alignment_report=report)
        assert path.exists()
        json_path = path.with_suffix(".alignment.json")
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "aligned_rows_out" in data

    def test_empty_inner_join_produces_empty_df(self):
        rain = self._rain(["2023-01-01", "2023-01-02"])
        disc = self._disc(["2023-01-10", "2023-01-11"])
        aligned, report = align_datasets(rain, disc, how="inner")
        assert len(aligned) == 0
        assert report["aligned_rows_out"] == 0


# =============================================================================
# EDA plots
# =============================================================================

class TestEdaPlots:
    """Verify every plot function runs without error and saves a file."""

    def _df(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=365 * 3, freq="D")
        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "precipitation_sum": np.random.exponential(5, len(dates)),
            "river_discharge": np.random.gamma(5, 50, len(dates)),
        })

    def _cfg(self, tmp_path):
        return {
            "paths": {"reports_dir": str(tmp_path / "reports")},
        }

    def test_time_series_saved(self, tmp_path):
        path = plot_time_series(self._df(), "TEST", self._cfg(tmp_path))
        assert path.exists()
        assert path.suffix == ".png"

    def test_monthly_box_saved(self, tmp_path):
        paths = plot_monthly_box(self._df(), "TEST", self._cfg(tmp_path))
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_rolling_stats_saved(self, tmp_path):
        path = plot_rolling_stats(self._df(), "TEST", self._cfg(tmp_path))
        assert path.exists()

    def test_scatter_lag_saved(self, tmp_path):
        path = plot_scatter_lag(self._df(), "TEST", self._cfg(tmp_path))
        assert path.exists()

    def test_missing_heatmap_saved(self, tmp_path):
        df = self._df()
        # inject some missing
        df.loc[10:20, "precipitation_sum"] = np.nan
        path = plot_missing_heatmap(df, "TEST", self._cfg(tmp_path))
        assert path.exists()

    def test_correlation_saved(self, tmp_path):
        path = plot_correlation(self._df(), "TEST", self._cfg(tmp_path))
        assert path.exists()

    def test_plots_have_nonzero_size(self, tmp_path):
        path = plot_time_series(self._df(), "TEST", self._cfg(tmp_path))
        assert path.stat().st_size > 5_000  # at least 5 KB
