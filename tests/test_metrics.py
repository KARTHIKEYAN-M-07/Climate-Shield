"""
tests/test_metrics.py
─────────────────────
Phase-5 tests:
  - Splitting boundaries (chronological order preserved, exact 70/15/15 fractions, no overlap).
  - Persistence baseline interface and predictions.
  - Metrics formulas against hand-calculated values:
      * MAE, RMSE, R²
      * NSE unclipped negative values
  - Event metrics:
      * Peak flow error
      * Peak timing error
      * Missed event rate, false alarm rate
      * Contingency table calculations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import chronological_split
from src.models.persistence import PersistenceBaseline
from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    nash_sutcliffe_efficiency,
    compute_standard_metrics,
)
from src.evaluation.event_metrics import (
    peak_flow_error,
    peak_timing_error,
    contingency_table,
    missed_event_rate,
    false_alarm_rate,
    compute_event_metrics,
)


# =============================================================================
# 1. Chronological Split Tests
# =============================================================================

class TestChronologicalSplit:
    def test_split_proportions_and_continuity(self):
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({"date": dates, "val": range(n)})

        train, val, test = chronological_split(df, 0.70, 0.15, 0.15)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert len(train) + len(val) + len(test) == n

        # Time continuity and zero overlap
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()

    def test_split_raises_on_too_small_df(self):
        df = pd.DataFrame({"val": range(5)})
        with pytest.raises(ValueError, match="too small"):
            chronological_split(df)


# =============================================================================
# 2. Persistence Baseline Tests
# =============================================================================

class TestPersistenceBaseline:
    def test_persistence_predicts_current_value(self):
        df = pd.DataFrame({
            "river_discharge": [10.0, 25.0, 50.0, 100.0],
            "feature_x": [1, 2, 3, 4],
        })
        model = PersistenceBaseline(discharge_col="river_discharge")
        model.fit(df)
        preds = model.predict(df)
        np.testing.assert_array_equal(preds, np.array([10.0, 25.0, 50.0, 100.0]))

    def test_persistence_missing_col_raises(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        model = PersistenceBaseline(discharge_col="river_discharge")
        with pytest.raises(KeyError, match="requires column 'river_discharge'"):
            model.predict(df)


# =============================================================================
# 3. Standard Metrics Tests vs Hand-Calculated Values
# =============================================================================

class TestStandardMetrics:
    def test_mae_hand_calculated(self):
        # true: [10, 20, 30]
        # pred: [12, 18, 33]
        # abs diffs: [2, 2, 3] -> mean = 7 / 3 ~ 2.3333
        y_true = [10.0, 20.0, 30.0]
        y_pred = [12.0, 18.0, 33.0]
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(7.0 / 3.0)

    def test_rmse_hand_calculated(self):
        # sq diffs: [4, 4, 9] -> mean = 17 / 3 -> sqrt ~ 2.38047
        y_true = [10.0, 20.0, 30.0]
        y_pred = [12.0, 18.0, 33.0]
        expected = np.sqrt(17.0 / 3.0)
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(expected)

    def test_r2_hand_calculated(self):
        # y_true: [10, 20, 30], mean = 20
        # ss_tot = (10-20)^2 + 0 + (30-20)^2 = 100 + 100 = 200
        # ss_res = 4 + 4 + 9 = 17
        # R2 = 1 - (17 / 200) = 1 - 0.085 = 0.915
        y_true = [10.0, 20.0, 30.0]
        y_pred = [12.0, 18.0, 33.0]
        assert r2_score(y_true, y_pred) == pytest.approx(0.915)

    def test_nse_negative_not_clipped(self):
        # NSE must never clip negative values!
        # If predictions are terribly far from true, NSE < 0.
        y_true = [10.0, 20.0, 30.0]
        # pred very far off: [100, 200, 300]
        # ss_res = (90^2 + 180^2 + 270^2) = 8100 + 32400 + 72900 = 113400
        # ss_tot = 200
        # NSE = 1 - (113400 / 200) = 1 - 567 = -566.0
        y_pred = [100.0, 200.0, 300.0]
        nse = nash_sutcliffe_efficiency(y_true, y_pred)
        assert nse < 0.0
        assert nse == pytest.approx(-566.0)

    def test_compute_standard_metrics_dict(self):
        res = compute_standard_metrics([1, 2], [1, 2])
        assert res["mae"] == pytest.approx(0.0)
        assert res["rmse"] == pytest.approx(0.0)
        assert res["r2"] == pytest.approx(1.0)
        assert res["nse"] == pytest.approx(1.0)


# =============================================================================
# 4. Event Metrics Tests
# =============================================================================

class TestEventMetrics:
    def test_peak_flow_error(self):
        # true peak: 200.0, pred peak: 250.0 -> (250-200)/200 * 100 = +25.0%
        y_true = [50.0, 100.0, 200.0, 150.0]
        y_pred = [60.0, 90.0, 250.0, 140.0]
        assert peak_flow_error(y_true, y_pred) == pytest.approx(25.0)

    def test_peak_timing_error(self):
        dates = ["2023-08-01", "2023-08-02", "2023-08-03", "2023-08-04"]
        # True peak on index 1 ("2023-08-02")
        y_true = [10.0, 100.0, 50.0, 20.0]
        # Pred peak on index 3 ("2023-08-04") -> +2 days late
        y_pred = [10.0, 50.0, 60.0, 90.0]
        assert peak_timing_error(dates, y_true, y_pred) == 2

    def test_contingency_and_event_rates(self):
        threshold = 100.0
        # Index 0: 50, 40 -> TN
        # Index 1: 120, 130 -> TP
        # Index 2: 150, 80 -> FN (miss)
        # Index 3: 70, 110 -> FP (false alarm)
        y_true = [50.0, 120.0, 150.0, 70.0]
        y_pred = [40.0, 130.0, 80.0, 110.0]

        tbl = contingency_table(y_true, y_pred, threshold)
        assert tbl["TP"] == 1
        assert tbl["FP"] == 1
        assert tbl["FN"] == 1
        assert tbl["TN"] == 1

        # Missed event rate = FN / (TP + FN) = 1 / 2 = 0.5 (50%)
        assert missed_event_rate(y_true, y_pred, threshold) == pytest.approx(0.5)

        # False alarm rate = FP / (FP + TN) = 1 / 2 = 0.5 (50%)
        assert false_alarm_rate(y_true, y_pred, threshold) == pytest.approx(0.5)

    def test_compute_event_metrics(self):
        y_true = [50.0, 150.0]
        y_pred = [50.0, 150.0]
        res = compute_event_metrics(y_true, y_pred, threshold=100.0, horizon_days=3)
        assert res["missed_event_rate"] == pytest.approx(0.0)
        assert res["false_alarm_rate"] == pytest.approx(0.0)
        assert res["warning_lead_time_days"] == 3
        assert res["true_positives"] == 1
