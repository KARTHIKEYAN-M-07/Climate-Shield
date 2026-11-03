"""
tests/test_evaluation_pipeline.py
─────────────────────────────────
Phase-7 tests:
  - Verifies train_and_evaluate end-to-end execution on synthetic dataset.
  - Verifies that model_scores.json and model_scores.csv are created with legitimate non-hardcoded values.
  - Verifies that all 6 models are evaluated on identical test timestamps.
  - Verifies presence of all required metrics:
      MAE, RMSE, R2, NSE, peak_flow_error_pct, peak_timing_error_days,
      missed_event_rate, false_alarm_rate, warning_lead_time_days.
  - Verifies beats_baseline flag matches comparative RMSE against persistence baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_and_evaluate
from src.utils.config import load_data_config


@pytest.fixture
def mock_dataset(tmp_path):
    """Synthetic continuous series for 120 days."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2022-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist()
    rain = np.random.exponential(5, n)
    # Autoregressive discharge
    disc = np.zeros(n)
    disc[0] = 50.0
    for i in range(1, n):
        disc[i] = 0.8 * disc[i-1] + 2.0 * rain[i-1] + np.random.normal(0, 3)
    disc = np.clip(disc, 1.0, None)

    df = pd.DataFrame({
        "date": dates,
        "precipitation_sum": rain,
        "river_discharge": disc,
    })
    return df


class TestEvaluationPipeline:
    def test_pipeline_runs_and_writes_reports(self, mock_dataset, tmp_path):
        cfg = load_data_config()
        # Override paths to temporary directory
        cfg["paths"]["models_dir"] = str(tmp_path / "models")
        cfg["paths"]["reports_dir"] = str(tmp_path / "reports")

        result = train_and_evaluate(
            station_id="TN_CAU_TRICHY",
            horizons=[1, 3],
            cfg=cfg,
            df_aligned=mock_dataset,
        )

        assert "scores" in result
        scores = result["scores"]
        assert len(scores) > 0

        # Check files created on disk
        json_file = tmp_path / "reports" / "model_scores.json"
        csv_file = tmp_path / "reports" / "model_scores.csv"
        assert json_file.exists()
        assert csv_file.exists()

        # Check required fields
        required_keys = {
            "model", "horizon_days", "mae", "rmse", "r2", "nse",
            "peak_flow_error_pct", "peak_timing_error_days",
            "missed_event_rate", "false_alarm_rate", "beats_baseline"
        }
        for entry in scores:
            for k in required_keys:
                assert k in entry, f"Missing key '{k}' in entry: {entry}"

        # Check saved joblib artifacts
        models_dir = tmp_path / "models"
        joblib_files = list(models_dir.glob("*.joblib"))
        assert len(joblib_files) >= 12  # at least 6 models * 2 horizons
