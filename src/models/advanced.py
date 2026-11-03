"""
src/models/advanced.py
──────────────────────
Additional ML models and ensembles for flood forecasting:
  1. ElasticNet (StandardScaler + SimpleImputer + ElasticNet)
  2. ExtraTreesRegressor (ExtraTrees ensemble)
  3. HistGradientBoostingRegressor (Histogram-based gradient boosting)
  4. HybridResidualModel (ML predicts the residual: y - y_hat_base)
  5. StackingForecastEnsemble (Ridge, Random Forest, HistGB with Ridge meta-learner)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_elastic_net_model(alpha: float = 1.0, l1_ratio: float = 0.5, random_state: int = 42) -> Pipeline:
    """Build ElasticNet pipeline with median imputation and standardization."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=2000)),
    ])


def build_extra_trees_model(
    n_estimators: int = 100,
    max_depth: int | None = 8,
    random_state: int = 42,
) -> Pipeline:
    """Build ExtraTrees regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def build_hist_gradient_boosting_model(
    max_iter: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Build HistGradientBoosting regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )),
    ])


class HybridResidualModel(BaseEstimator, RegressorMixin):
    """Hybrid model: base empirical model (e.g. Persistence) + ML residual corrector.

    Forecast:
        y_hat = base_model(X) + ml_corrector(X)
    where ml_corrector is trained to predict the residual: y_true - base_model(X).
    """

    def __init__(self, base_estimator: Any, residual_corrector: Any):
        self.base_estimator = base_estimator
        self.residual_corrector = residual_corrector

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "HybridResidualModel":
        y_arr = np.asarray(y, dtype=float)
        self.base_estimator.fit(X, y_arr)
        base_preds = self.base_estimator.predict(X)
        residual = y_arr - base_preds
        self.residual_corrector.fit(X, residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = self.base_estimator.predict(X)
        corr_preds = self.residual_corrector.predict(X)
        combined = base_preds + corr_preds
        return np.asarray(np.clip(combined, 0.0, None), dtype=float)


def build_stacking_ensemble(random_state: int = 42) -> StackingRegressor:
    """Build Stacking Regressor combining Ridge, ExtraTrees, and HistGB with Ridge meta-learner.

    Meta-learner weights are fit strictly on cross-validation folds of the training set.
    """
    estimators = [
        ("ridge", build_elastic_net_model(alpha=0.5, random_state=random_state)),
        ("extra_trees", build_extra_trees_model(n_estimators=50, random_state=random_state)),
        ("hist_gb", build_hist_gradient_boosting_model(max_iter=50, random_state=random_state)),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1,
    )


class VotingAverageEnsemble(BaseEstimator, RegressorMixin):
    """Ensemble that averages predictions from multiple estimators."""

    def __init__(self, estimators: list[tuple[str, Any]] | None = None):
        self.estimators = estimators or []

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "VotingAverageEnsemble":
        for name, est in self.estimators:
            est.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = [est.predict(X) for _, est in self.estimators]
        avg = np.mean(preds, axis=0)
        return np.asarray(np.clip(avg, 0.0, None), dtype=float)


def build_average_ensemble(random_state: int = 42) -> VotingAverageEnsemble:
    """Build simple average ensemble of top models (ElasticNet, ExtraTrees, HistGB)."""
    return VotingAverageEnsemble([
        ("elastic_net", build_elastic_net_model(alpha=0.5, random_state=random_state)),
        ("extra_trees", build_extra_trees_model(n_estimators=50, random_state=random_state)),
        ("hist_gb", build_hist_gradient_boosting_model(max_iter=50, random_state=random_state)),
    ])
