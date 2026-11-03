"""src/risk/rainfall_risk.py
─────────────────────────
Rule-based rainfall risk indicator for unsupported or coastal river stations.

Evaluates:
  - 24h rainfall (mm)
  - 72h cumulative rainfall (mm)
  - Peak hourly intensity (mm/h)
  - Precipitation probability (%)
Categorises risk based on IMD meteorological rainfall standards.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def compute_rainfall_risk_indicator(
    df: Optional[pd.DataFrame] = None,
    live_weather_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute rule-based rainfall risk indicators from historical/live rainfall.

    Categories (IMD Meteorological Thresholds):
      - Severe (Warning): 24h >= 115.6 mm (Very Heavy/Extremely Heavy) or 72h >= 150.0 mm
      - High (Alert): 24h >= 64.5 mm (Heavy Rain) or 72h >= 100.0 mm
      - Moderate (Advisory): 24h >= 15.6 mm (Moderate Rain) or 72h >= 50.0 mm
      - Low (Normal): 24h < 15.6 mm and 72h < 50.0 mm

    Returns
    -------
    dict
        Contains 24h, 72h, peak intensity, probability, category, and disclaimer.
    """
    rain_24h = 0.0
    rain_72h = 0.0
    peak_intensity = 0.0
    precip_prob = 0.0

    if df is not None and not df.empty and "precipitation_sum" in df.columns:
        s = pd.to_numeric(df["precipitation_sum"], errors="coerce").dropna()
        if not s.empty:
            rain_24h = float(s.iloc[-1])
            rain_72h = float(s.iloc[-3:].sum()) if len(s) >= 3 else float(s.sum())
            # Approximate peak hourly intensity as daily / 4 during storm peak
            peak_intensity = round(max(rain_24h / 6.0, 0.0), 1)

    if live_weather_data:
        daily = live_weather_data.get("daily", {})
        if "precipitation_sum" in daily and len(daily["precipitation_sum"]) > 0:
            rain_24h = float(daily["precipitation_sum"][0] or rain_24h)
        if "precipitation_probability_max" in daily and len(daily["precipitation_probability_max"]) > 0:
            precip_prob = float(daily["precipitation_probability_max"][0] or 0.0)
        hourly = live_weather_data.get("hourly", {})
        if "precipitation" in hourly and len(hourly["precipitation"]) > 0:
            h_vals = [float(v) for v in hourly["precipitation"][:24] if v is not None]
            if h_vals:
                peak_intensity = round(max(h_vals), 1)

    # Risk categorization
    if rain_24h >= 115.6 or rain_72h >= 150.0:
        category = "Severe"
        badge_class = "risk-severe"
        rationale = f"Very Heavy rainfall detected (24h: {rain_24h:.1f} mm, 72h: {rain_72h:.1f} mm) — severe surface waterlogging risk"
    elif rain_24h >= 64.5 or rain_72h >= 100.0:
        category = "High"
        badge_class = "risk-high"
        rationale = f"Heavy rainfall detected (24h: {rain_24h:.1f} mm, 72h: {rain_72h:.1f} mm) — significant localized flood risk"
    elif rain_24h >= 15.6 or rain_72h >= 50.0:
        category = "Moderate"
        badge_class = "risk-moderate"
        rationale = f"Moderate rainfall detected (24h: {rain_24h:.1f} mm, 72h: {rain_72h:.1f} mm) — monitor low-lying drainage corridors"
    else:
        category = "Low"
        badge_class = "risk-low"
        rationale = f"Light or normal rainfall (24h: {rain_24h:.1f} mm, 72h: {rain_72h:.1f} mm) — no active rainfall warning"

    return {
        "is_rainfall_only": True,
        "mode_label": "Rainfall risk only: no river-flow model for this station",
        "category": category,
        "badge_class": badge_class,
        "rain_24h_mm": round(rain_24h, 1),
        "rain_72h_mm": round(rain_72h, 1),
        "peak_intensity_mmh": round(peak_intensity, 1),
        "precip_probability_pct": round(precip_prob, 1),
        "rationale": rationale,
        "disclaimer": "Rainfall risk only: no river-flow model for this station. Flow forecasts and flow thresholds disabled.",
    }
