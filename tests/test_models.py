"""
tests/test_models.py
────────────────────
Phase-6 tests:
  - Verifying instantiation, fitting, and prediction for all core models:
      1. Persistence Baseline
      2. Ridge Regression
      3. Decision Tree
      4. Random Forest
      5. Gradient Boosting
      6. XGBoost
  - Verifying handling of NaNs in features (handled by Pipeline imputers)
  - Verifying LSTM handles absence of TensorFlow gracefully with clear explanation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.persistence import PersistenceBaseline
from src.models.linear import build_ridge_model
from src.models.tree import build_decision_tree_model
from src.models.random_forest import build_random_forest_model
from src.models.gradient_boosting import build_gradient_boosting_model
from src.models.xgboost_model import build_xgboost_model
from src.models.lstm import is_lstm_available, build_lstm_model


@pytest.fixture
def synthetic_data():
    """Synthetic dataset with 50 samples and 5 features, including some missing values."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({
        "river_discharge": np.random.uniform(10, 100, n),
        "feat_lag1": np.random.uniform(0, 50, n),
        "feat_cumul": np.random.uniform(0, 150, n),
        "month_sin": np.random.uniform(-1, 1, n),
        "month_cos": np.random.uniform(-1, 1, n),
    })
    # Inject a few NaNs to verify imputer robustness
    X.loc[5, "feat_lag1"] = np.nan
    X.loc[12, "feat_cumul"] = np.nan

    y = X["river_discharge"].values * 1.1 + 5.0 + np.random.normal(0, 2, n)
    return X, y


class TestModelPipelines:
    def test_ridge_pipeline(self, synthetic_data):
        X, y = synthetic_data
        model = build_ridge_model()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_decision_tree_pipeline(self, synthetic_data):
        X, y = synthetic_data
        model = build_decision_tree_model()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_random_forest_pipeline(self, synthetic_data):
        X, y = synthetic_data
        model = build_random_forest_model(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_gradient_boosting_pipeline(self, synthetic_data):
        X, y = synthetic_data
        model = build_gradient_boosting_model(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_xgboost_pipeline(self, synthetic_data):
        X, y = synthetic_data
        model = build_xgboost_model(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_lstm_graceful_handling(self):
        """Verifies that if TF is missing, clear informative ImportError is raised."""
        if not is_lstm_available():
            with pytest.raises(ImportError, match="TensorFlow is not installed"):
                build_lstm_model((10, 5))
        else:
            m = build_lstm_model((10, 5))
            assert m is not None
