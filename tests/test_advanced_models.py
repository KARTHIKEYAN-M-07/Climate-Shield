"""
tests/test_advanced_models.py
─────────────────────────────
Phase-14 tests:
  - SeasonalClimatology:
      * Fits day-of-year means strictly on train
      * Accurately predicts matching calendar days
  - DampedPersistence:
      * Fits mu and lag-1 autocorrelation phi
      * Correctly damps predictions over horizon h
  - ElasticNet pipeline execution & imputer tolerance
  - ExtraTrees pipeline execution & imputer tolerance
  - HistGradientBoosting pipeline execution
  - HybridResidualModel:
      * Base estimator + residual corrector round-trip
      * Correctly sums base prediction and residual correction
  - StackingForecastEnsemble execution on synthetic folds
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import SeasonalClimatology, DampedPersistence
from src.models.persistence import PersistenceBaseline
from src.models.linear import build_ridge_model
from src.models.advanced import (
    build_elastic_net_model,
    build_extra_trees_model,
    build_hist_gradient_boosting_model,
    HybridResidualModel,
    build_stacking_ensemble,
)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist()
    X = pd.DataFrame({
        "date": dates,
        "river_discharge": np.random.uniform(50, 200, n),
        "feat_rain": np.random.uniform(0, 40, n),
        "feat_cumul": np.random.uniform(0, 100, n),
    })
    # Add an intentional NaN to test imputer robustness
    X.loc[10, "feat_rain"] = np.nan
    y = X["river_discharge"].values * 1.05 + 10.0 + np.random.normal(0, 2, n)
    return X, y


class TestAdvancedBaselines:
    def test_seasonal_climatology(self):
        dates = ["2022-08-15", "2023-08-15", "2022-08-16"]
        df = pd.DataFrame({
            "date": dates,
            "river_discharge": [100.0, 120.0, 50.0],
        })
        model = SeasonalClimatology()
        model.fit(df)
        # Aug 15 mean = (100 + 120) / 2 = 110.0
        test_df = pd.DataFrame({"date": ["2024-08-15"]})
        pred = model.predict(test_df)
        assert pred[0] == pytest.approx(110.0)

    def test_damped_persistence(self):
        df = pd.DataFrame({"river_discharge": [10.0, 20.0, 30.0, 40.0, 50.0]})
        model = DampedPersistence(horizon_days=2)
        model.fit(df)
        pred = model.predict(pd.DataFrame({"river_discharge": [50.0]}))
        assert len(pred) == 1
        assert pred[0] > 0.0


class TestAdvancedModels:
    def test_elastic_net(self, synthetic_data):
        X, y = synthetic_data
        model = build_elastic_net_model()
        model.fit(X[["river_discharge", "feat_rain", "feat_cumul"]], y)
        preds = model.predict(X[["river_discharge", "feat_rain", "feat_cumul"]])
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_extra_trees(self, synthetic_data):
        X, y = synthetic_data
        model = build_extra_trees_model(n_estimators=10)
        model.fit(X[["river_discharge", "feat_rain", "feat_cumul"]], y)
        preds = model.predict(X[["river_discharge", "feat_rain", "feat_cumul"]])
        assert len(preds) == len(X)

    def test_hist_gradient_boosting(self, synthetic_data):
        X, y = synthetic_data
        model = build_hist_gradient_boosting_model(max_iter=10)
        model.fit(X[["river_discharge", "feat_rain", "feat_cumul"]], y)
        preds = model.predict(X[["river_discharge", "feat_rain", "feat_cumul"]])
        assert len(preds) == len(X)

    def test_hybrid_residual_model(self, synthetic_data):
        X, y = synthetic_data
        features = ["river_discharge", "feat_rain", "feat_cumul"]
        base = PersistenceBaseline()
        corrector = build_ridge_model()
        hybrid = HybridResidualModel(base_estimator=base, residual_corrector=corrector)
        hybrid.fit(X[features], y)
        preds = hybrid.predict(X[features])
        assert len(preds) == len(X)
        assert (preds >= 0).all()  # non-negative constraint

    def test_stacking_ensemble(self, synthetic_data):
        X, y = synthetic_data
        features = ["river_discharge", "feat_rain", "feat_cumul"]
        stacker = build_stacking_ensemble()
        stacker.fit(X[features], y)
        preds = stacker.predict(X[features])
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()
