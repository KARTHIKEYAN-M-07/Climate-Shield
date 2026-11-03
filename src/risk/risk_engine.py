"""
src/risk/risk_engine.py
───────────────────────
Decision-support risk classification layer.
Separate from the ML models.

Takes:
  (predicted_level, station_id, thresholds_cfg, horizon_days, data_freshness_hours)

Returns:
  - category: "Low" | "Moderate" | "High" | "Severe" | "UNAVAILABLE"
  - rationale: Explanation of decision
  - threshold_source: Documented source provenance
  - is_stale: Boolean indicator of input freshness violation
  - warning_message: Prototype disclaimer

Rules:
  - Returns "UNAVAILABLE" if inputs are stale (age > max_data_age_hours), missing, or negative.
  - Returns "UNAVAILABLE" if station thresholds are null / uncomputed.
  - Never fabricate or guess risk categories.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "PROTOTYPE STATISTICAL RISK CATEGORY ONLY — NOT AN OFFICIAL CWC/IMD FLOOD WARNING. "
    "Derived from historical modelled discharge percentiles. Follow official disaster management guidance."
)


def evaluate_risk(
    predicted_level: float | None,
    station_id: str,
    thresholds_cfg: dict[str, Any],
    horizon_days: int = 1,
    data_freshness_hours: float | None = 0.0,
    current_level: float | None = None,
    model_exists: bool = True,
) -> dict[str, Any]:
    """Classify forecast river discharge into prototype risk category.

    Parameters
    ----------
    predicted_level: Forecast river discharge (m³/s) for the horizon.
    station_id: Station ID.
    thresholds_cfg: Threshold configuration loaded from thresholds.yaml.
    horizon_days: Forecast horizon in days (1, 3, or 7).
    data_freshness_hours: Age of observed data in hours.
    current_level: Optional latest observed river discharge (m³/s) for trend context.
    model_exists: Whether a trained model artifact exists for this station/horizon.

    Returns
    -------
    dict with category, rationale, threshold_crossed, threshold_source, etc.
    """
    source_info = (
        "Prototype statistical thresholds (training-set percentiles of Open-Meteo GloFAS modelled discharge). "
        "NOT official CWC warning or danger flood levels."
    )
    staleness_rules = thresholds_cfg.get("staleness", {})
    max_data_age = staleness_rules.get("max_data_age_hours", 48)

    # ── Check model existence ─────────────────────────────────────────────────
    if not model_exists:
        return {
            "status": "UNAVAILABLE",
            "category": "UNAVAILABLE",
            "predicted_discharge_m3s": None,
            "current_discharge_m3s": current_level,
            "station_id": station_id,
            "horizon_days": horizon_days,
            "threshold_crossed": "None (No trained model)",
            "threshold_value": None,
            "rationale": f"No trained model artifact found for station '{station_id}'. Risk assessment is unavailable.",
            "threshold_source": source_info,
            "is_stale": False,
            "season_context": "Seasonal timing is contextual only and does not drive risk.",
            "disclaimer": DISCLAIMER,
        }

    # ── Check staleness ───────────────────────────────────────────────────────
    if data_freshness_hours is not None and data_freshness_hours > max_data_age:
        return {
            "status": "UNAVAILABLE",
            "category": "UNAVAILABLE",
            "predicted_discharge_m3s": predicted_level,
            "current_discharge_m3s": current_level,
            "station_id": station_id,
            "horizon_days": horizon_days,
            "threshold_crossed": "None (Stale data)",
            "threshold_value": None,
            "rationale": (
                f"Data freshness ({data_freshness_hours:.1f} hours) exceeds maximum allowed age "
                f"({max_data_age} hours). Decision layer cannot certify stale inputs."
            ),
            "threshold_source": source_info,
            "is_stale": True,
            "season_context": "Seasonal timing is contextual only and does not drive risk.",
            "disclaimer": DISCLAIMER,
        }

    # ── Check valid numeric input ─────────────────────────────────────────────
    if predicted_level is None or predicted_level < 0:
        return {
            "status": "UNAVAILABLE",
            "category": "UNAVAILABLE",
            "predicted_discharge_m3s": predicted_level,
            "current_discharge_m3s": current_level,
            "station_id": station_id,
            "horizon_days": horizon_days,
            "threshold_crossed": "None (Invalid input)",
            "threshold_value": None,
            "rationale": "Missing or negative predicted discharge input.",
            "threshold_source": source_info,
            "is_stale": False,
            "season_context": "Seasonal timing is contextual only and does not drive risk.",
            "disclaimer": DISCLAIMER,
        }

    # ── Check station thresholds available ────────────────────────────────────
    station_thresh = thresholds_cfg.get("stations", {}).get(station_id)
    if not station_thresh or "categories" not in station_thresh:
        return {
            "status": "UNAVAILABLE",
            "category": "UNAVAILABLE",
            "predicted_discharge_m3s": predicted_level,
            "current_discharge_m3s": current_level,
            "station_id": station_id,
            "horizon_days": horizon_days,
            "threshold_crossed": "None (Missing station thresholds)",
            "threshold_value": None,
            "rationale": f"No threshold configuration found for station '{station_id}'.",
            "threshold_source": source_info,
            "is_stale": False,
            "season_context": "Seasonal timing is contextual only and does not drive risk.",
            "disclaimer": DISCLAIMER,
        }

    cats = station_thresh["categories"]
    p50 = cats.get("Moderate", {}).get("lower_bound_m3s")
    p75 = cats.get("High", {}).get("lower_bound_m3s")
    p90 = cats.get("Severe", {}).get("lower_bound_m3s")

    # If any threshold is null or uncalibrated
    if p50 is None or p75 is None or p90 is None:
        return {
            "status": "UNAVAILABLE",
            "category": "UNAVAILABLE",
            "predicted_discharge_m3s": predicted_level,
            "current_discharge_m3s": current_level,
            "station_id": station_id,
            "horizon_days": horizon_days,
            "threshold_crossed": "None (Uncalibrated thresholds)",
            "threshold_value": None,
            "rationale": "Station thresholds have not been calibrated yet from training data.",
            "threshold_source": source_info,
            "is_stale": False,
            "season_context": "Seasonal timing is contextual only and does not drive risk.",
            "disclaimer": DISCLAIMER,
        }

    # ── Map category strictly from FORECAST ───────────────────────────────────
    if predicted_level >= p90:
        category = "Severe"
        threshold_crossed = f"Severe (≥ {p90:.2f} m³/s, 90th percentile)"
        threshold_value = p90
        rationale = f"Forecast flow ({predicted_level:.2f} m³/s) meets or exceeds 90th percentile Severe threshold ({p90:.2f} m³/s)."
    elif predicted_level >= p75:
        category = "High"
        threshold_crossed = f"High (≥ {p75:.2f} m³/s, 75th percentile)"
        threshold_value = p75
        rationale = f"Forecast flow ({predicted_level:.2f} m³/s) exceeds 75th percentile High threshold ({p75:.2f} m³/s)."
    elif predicted_level >= p50:
        category = "Moderate"
        threshold_crossed = f"Moderate (≥ {p50:.2f} m³/s, median)"
        threshold_value = p50
        rationale = f"Forecast flow ({predicted_level:.2f} m³/s) exceeds historical median threshold ({p50:.2f} m³/s)."
    else:
        category = "Low"
        threshold_crossed = f"None (normal baseline < {p50:.2f} m³/s median)"
        threshold_value = p50
        rationale = f"Forecast flow ({predicted_level:.2f} m³/s) is within normal baseline range (< median {p50:.2f} m³/s)."

    # Trend context when current level is provided
    if current_level is not None:
        if predicted_level < current_level:
            rationale += f" Flow is receding from current observed level ({current_level:.2f} m³/s) to forecast {predicted_level:.2f} m³/s."
        elif predicted_level > current_level:
            rationale += f" Flow is rising from current observed level ({current_level:.2f} m³/s) to forecast {predicted_level:.2f} m³/s."
        else:
            rationale += f" Flow is steady at current observed level ({current_level:.2f} m³/s)."

    # Season context is strictly educational context text, never a risk driver
    season_text = (
        "Tamil Nadu Northeast Monsoon (Oct-Dec) is the primary regional flood window. "
        "Seasonal context is contextual only (informational); risk category is determined strictly by forecast flow."
    )

    return {
        "status": "OK",
        "category": category,
        "predicted_discharge_m3s": round(predicted_level, 2),
        "current_discharge_m3s": round(current_level, 2) if current_level is not None else None,
        "station_id": station_id,
        "horizon_days": horizon_days,
        "threshold_crossed": threshold_crossed,
        "threshold_value": threshold_value,
        "rationale": rationale,
        "threshold_source": source_info,
        "season_context": season_text,
        "thresholds_applied": {
            "Low": [0.0, p50],
            "Moderate": [p50, p75],
            "High": [p75, p90],
            "Severe": [p90, float("inf")],
        },
        "is_stale": False,
        "disclaimer": DISCLAIMER,
    }
