"""
src/risk/thresholds.py
──────────────────────
Thresholds computation and management for prototype decision layer.

⚠ PROTOTYPE STATISTICAL THRESHOLDS — NOT OFFICIAL CWC WARNING/DANGER LEVELS
These percentile thresholds (50th, 75th, 90th, 95th) are derived from the training set
of Open-Meteo modelled river discharge (m³/s) exclusively for academic demonstration.
They are NOT official warning levels issued by CWC, IMD, or any State Disaster Management Authority.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils.config import load_thresholds_config

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_station_thresholds(
    training_discharge: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Compute statistical prototype thresholds from training-set river discharge.

    Categories:
      Low:      [0, p50)
      Moderate: [p50, p75)
      High:     [p75, p90)
      Severe:   [p90, inf)
      Event:    p95 (used for event metrics)
    """
    s = pd.Series(training_discharge).dropna()
    if len(s) == 0:
        raise ValueError("Cannot compute thresholds on empty discharge series.")

    p50 = float(np.percentile(s, 50))
    p75 = float(np.percentile(s, 75))
    p90 = float(np.percentile(s, 90))
    p95 = float(np.percentile(s, 95))

    return {
        "percentile_basis": {
            "p50": round(p50, 2),
            "p75": round(p75, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
        },
        "categories": {
            "Low": {
                "description": "Below median historical discharge. Normal flow.",
                "lower_bound_m3s": 0.0,
            },
            "Moderate": {
                "description": "Above median. Elevated; monitor closely.",
                "lower_bound_m3s": round(p50, 2),
            },
            "High": {
                "description": "Above 75th percentile. Significant risk; prepare.",
                "lower_bound_m3s": round(p75, 2),
            },
            "Severe": {
                "description": "Above 90th percentile. Extreme risk; act immediately.",
                "lower_bound_m3s": round(p90, 2),
            },
        },
    }


def update_thresholds_config(
    station_id: str,
    threshold_data: dict[str, Any],
    config_path: Path | None = None,
) -> None:
    """Update configs/thresholds.yaml with computed station thresholds."""
    if config_path is None:
        config_path = _PROJECT_ROOT / "configs" / "thresholds.yaml"

    cfg = load_thresholds_config()
    if "stations" not in cfg:
        cfg["stations"] = {}

    cfg["stations"][station_id] = threshold_data
    cfg["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    logger.info("Updated thresholds in %s for station %s", config_path, station_id)
