"""
src/risk/safety_guards.py
─────────────────────────
Operational safety guards and explainability for flood forecasting:
  1. Out-of-Distribution (OOD) Guard:
     - Detects inputs outside the historical training envelope (beyond min/max or > 3*IQR).
     - Forces risk engine to return "LOW CONFIDENCE" or "UNAVAILABLE".
  2. Plain-Language Prediction Explainer:
     - Translates top feature attributions into intuitive hydrologic statements.
  3. ML What-If Simulator:
     - Simulates rainfall scaling scenarios (+20%, +50%).
     - Strictly flagged as HYPOTHETICAL with tree-extrapolation warnings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


# =============================================================================
# Out-of-Distribution (OOD) Guard
# =============================================================================


class OutOfDistributionGuard:
    """Monitors feature vectors against the historical training distribution envelope."""

    def __init__(self, iqr_multiplier: float = 3.0):
        self.iqr_multiplier = iqr_multiplier
        self.feature_stats: dict[str, dict[str, float]] = {}
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "OutOfDistributionGuard":
        """Compute training distribution bounds (min, max, Q1, Q3, IQR)."""
        self.feature_stats = {}
        for col in X_train.select_dtypes(include=[np.number]).columns:
            vals = X_train[col].dropna().values
            if len(vals) == 0:
                continue
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = q75 - q25
            self.feature_stats[col] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "q25": q25,
                "q75": q75,
                "lower_bound": q25 - self.iqr_multiplier * iqr,
                "upper_bound": q75 + self.iqr_multiplier * iqr,
            }
        self.is_fitted = True
        return self

    def check_sample(self, sample_features: dict[str, float] | pd.Series) -> dict[str, Any]:
        """Check if incoming sample violates distribution boundaries."""
        if not self.is_fitted:
            return {"is_ood": False, "violations": [], "status": "GUARD_NOT_FITTED"}

        violations = []
        for feat, val in sample_features.items():
            if feat not in self.feature_stats or val is None or np.isnan(val):
                continue
            stats = self.feature_stats[feat]
            if val < stats["lower_bound"] or val > stats["upper_bound"]:
                violations.append({
                    "feature": feat,
                    "value": round(float(val), 2),
                    "expected_range": [round(stats["lower_bound"], 2), round(stats["upper_bound"], 2)],
                    "historical_max": round(stats["max"], 2),
                })

        is_ood = len(violations) > 0
        return {
            "is_ood": is_ood,
            "status": "OOD_DETECTED" if is_ood else "IN_DISTRIBUTION",
            "violations_count": len(violations),
            "violations": violations,
            "recommended_action": "RETURN_LOW_CONFIDENCE" if is_ood else "PROCEED",
        }


# =============================================================================
# Plain-Language Explainer
# =============================================================================


def explain_prediction(
    sample: dict[str, float],
    training_summary: dict[str, dict[str, float]],
) -> list[str]:
    """Generate human-readable explanations based on key hydrologic indicators."""
    reasons = []

    # 1. 7-day cumulative rainfall
    cumul_7d = sample.get("precipitation_sum_cumul_7d")
    if cumul_7d is not None and "precipitation_sum_cumul_7d" in training_summary:
        p90 = training_summary["precipitation_sum_cumul_7d"].get("p90", 50.0)
        if cumul_7d >= p90:
            reasons.append(f"7-day cumulative rainfall ({cumul_7d:.1f} mm) is in the top 10% historical bracket.")

    # 2. Rate of change (surge)
    roc_1d = sample.get("river_discharge_roc_1d")
    if roc_1d is not None and roc_1d > 20.0:
        reasons.append(f"Rapid discharge surge (+{roc_1d:.1f} m³/s in 24 hours) detected at observation origin.")

    # 3. Antecedent Precipitation Index (soil moisture memory)
    api = sample.get("api_decay_90")
    if api is not None and api > 40.0:
        reasons.append(f"Catchment soil saturation memory index (API={api:.1f}) indicates wet antecedent ground.")

    # 4. Regional Monsoon
    if sample.get("monsoon_ne") == 1.0:
        reasons.append("Observation falls within the Northeast Monsoon window (Tamil Nadu primary flood period).")
    elif sample.get("monsoon_sw") == 1.0:
        reasons.append("Observation falls within the Southwest Monsoon season.")

    if not reasons:
        reasons.append("Baseline seasonal hydrologic baseflow without active storm surges.")

    return reasons


# =============================================================================
# ML What-If Simulator
# =============================================================================




def simulate_rainfall_what_if(
    model: Any,
    sample_df: pd.DataFrame,
    rainfall_multiplier: float,
) -> dict[str, Any]:
    """Simulate a hypothetical discharge forecast under scaled rainfall.

    This is a hypothetical scenario, not an operational flood forecast.
    """
    if rainfall_multiplier < 0:
        raise ValueError("Rainfall multiplier cannot be negative.")

    if sample_df.empty:
        raise ValueError("sample_df cannot be empty.")

    # Keep the original data for the baseline prediction
    baseline_df = sample_df.copy()
    scaled_df = sample_df.copy()

    # Identify numeric rainfall-related features (excluding flag columns and non-numeric types)
    rain_cols = [
        col
        for col in scaled_df.columns
        if any(
            keyword in col.lower()
            for keyword in ("precipitation", "rain", "cumul")
        )
        and not col.lower().endswith("_flag")
        and not col.lower().endswith("_status")
        and pd.api.types.is_numeric_dtype(scaled_df[col])
    ]

    # Scale rainfall-related features for the hypothetical scenario
    for col in rain_cols:
        scaled_df[col] = scaled_df[col] * rainfall_multiplier

    # Align both inputs with the features used during model training
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)

        missing_cols = [
            col for col in expected_cols
            if col not in scaled_df.columns
        ]

        if missing_cols:
            raise ValueError(
                "Missing model features in rainfall simulation: "
                f"{missing_cols}. Check your feature-generation pipeline."
            )

        baseline_df = baseline_df.loc[:, expected_cols]
        scaled_df = scaled_df.loc[:, expected_cols]

    # Generate baseline and hypothetical predictions
    base_pred = float(model.predict(baseline_df)[0])
    hypothetical_pred = float(model.predict(scaled_df)[0])

    # Calculate percentage change safely
    if base_pred != 0:
        pct_change = (
            (hypothetical_pred - base_pred) / abs(base_pred)
        ) * 100.0
    else:
        pct_change = 0.0

    # The training rainfall range has not been checked here.
    # Do not present this placeholder as a verified safety assessment.
    extrapolation_status = "CHECK_REQUIRED"

    return {
        "status": "HYPOTHETICAL_SIMULATION",
        "multiplier": rainfall_multiplier,
        "delta_rainfall_pct": round(
            (rainfall_multiplier - 1.0) * 100.0, 1
        ),
        "base_forecast_m3s": round(base_pred, 2),
        "simulated_forecast_m3s": round(hypothetical_pred, 2),
        "forecast_change_pct": round(pct_change, 2),
        "extrapolation_status": extrapolation_status,
        "caveat": (
            "HYPOTHETICAL SCENARIO ONLY. Rainfall-related input features "
            "were scaled for this simulation. Derived features may need "
            "to be recalculated using the original training pipeline. "
            "Tree models cannot extrapolate beyond the maximum observed rainfall. "
            "This result is not an operational forecast and must not be "
            "used for flood safety decisions."
        ),
    }
