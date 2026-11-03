"""
src/data/grid_snap.py
─────────────────────
GloFAS Grid-Snap & Resolution Check:
  GloFAS river discharge is produced on an approximate 0.05° (~5 km) hydrological grid.
  Because physical river channels often meander across grid boundaries, the nominal
  geographic coordinate of a station may sit on an adjacent non-channel cell with zero
  or muted flow.

This module:
  1. Searches the nominal cell and its 8 adjacent grid cells (+/- 0.05°).
  2. Evaluates the modelled discharge series for each cell.
  3. Selects the cell with the highest hydrologic signal (discharge dynamic range & mean).
  4. Excludes candidate stations where all neighbouring cells are near zero (< 1.0 m³/s)
     or flat, returning an explicit status: "River too small for model resolution".
  5. Caches the snapped coordinate and the selection rationale.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_grid_cell(
    series: pd.Series | np.ndarray,
    min_mean: float = 1.0,
    min_max: float = 5.0,
) -> dict[str, Any]:
    """Evaluate whether a discharge series represents a viable GloFAS river channel."""
    vals = pd.Series(series).dropna().values
    if len(vals) == 0:
        return {
            "is_viable": False,
            "mean": 0.0,
            "max": 0.0,
            "std": 0.0,
            "reason": "Empty or missing discharge series.",
        }

    mean_q = float(np.mean(vals))
    max_q = float(np.max(vals))
    std_q = float(np.std(vals))

    if mean_q < min_mean and max_q < min_max:
        return {
            "is_viable": False,
            "mean": round(mean_q, 3),
            "max": round(max_q, 3),
            "std": round(std_q, 3),
            "reason": (
                f"River too small for GloFAS model resolution (~5 km grid). "
                f"Mean={mean_q:.2f} m³/s, Max={max_q:.2f} m³/s."
            ),
        }

    return {
        "is_viable": True,
        "mean": round(mean_q, 3),
        "max": round(max_q, 3),
        "std": round(std_q, 3),
        "reason": f"Active river channel cell (Mean={mean_q:.1f} m³/s, Max={max_q:.1f} m³/s).",
    }


def snap_to_best_channel(
    station: dict[str, Any],
    fetch_discharge_fn: Callable[[float, float], pd.DataFrame],
    grid_step_deg: float = 0.05,
    min_mean: float = 1.0,
    min_max: float = 5.0,
) -> dict[str, Any]:
    """Query 3x3 surrounding cells and snap to the most representative GloFAS channel.

    Parameters
    ----------
    station: Station dict with latitude and longitude.
    fetch_discharge_fn: Function (lat, lon) -> DataFrame with 'river_discharge'.
    grid_step_deg: Spatial resolution of GloFAS (~0.05 degrees).
    min_mean: Minimum allowable mean flow to qualify as viable channel.
    min_max: Minimum allowable peak flow to qualify as viable channel.

    Returns
    -------
    dict
        Selection result containing snapped_lat, snapped_lon, is_viable, and selection_rationale.
    """
    nom_lat = station["latitude"]
    nom_lon = station["longitude"]
    station_id = station.get("id", "UNKNOWN")

    candidates = []
    # Test 3x3 window: center, cardinal, and diagonals
    offsets = [
        (0.0, 0.0, "nominal_center"),
        (grid_step_deg, 0.0, "north"),
        (-grid_step_deg, 0.0, "south"),
        (0.0, grid_step_deg, "east"),
        (0.0, -grid_step_deg, "west"),
        (grid_step_deg, grid_step_deg, "north_east"),
        (grid_step_deg, -grid_step_deg, "north_west"),
        (-grid_step_deg, grid_step_deg, "south_east"),
        (-grid_step_deg, -grid_step_deg, "south_west"),
    ]

    best_candidate = None
    highest_score = -1.0

    for d_lat, d_lon, label in offsets:
        test_lat = round(nom_lat + d_lat, 4)
        test_lon = round(nom_lon + d_lon, 4)

        try:
            df = fetch_discharge_fn(test_lat, test_lon)
            if "river_discharge" in df.columns:
                series = df["river_discharge"]
            else:
                continue

            eval_res = evaluate_grid_cell(series, min_mean=min_mean, min_max=min_max)
            # Channel score rewards higher mean flow and dynamic range
            score = eval_res["mean"] + 0.5 * eval_res["std"]

            cand_info = {
                "latitude": test_lat,
                "longitude": test_lon,
                "offset_label": label,
                "is_nominal": (d_lat == 0.0 and d_lon == 0.0),
                "eval": eval_res,
                "score": score,
            }
            candidates.append(cand_info)

            if score > highest_score:
                highest_score = score
                best_candidate = cand_info
        except Exception as e:
            logger.warning("Grid-snap query failed at (%.4f, %.4f): %s", test_lat, test_lon, e)
            continue

    if not best_candidate or not best_candidate["eval"]["is_viable"]:
        return {
            "station_id": station_id,
            "nominal_latitude": nom_lat,
            "nominal_longitude": nom_lon,
            "snapped_latitude": nom_lat,
            "snapped_longitude": nom_lon,
            "is_viable": False,
            "status": "excluded",
            "exclusion_reason": "River channel too small for GloFAS model resolution (~5 km grid).",
            "all_candidates": candidates,
        }

    snapped_lat = best_candidate["latitude"]
    snapped_lon = best_candidate["longitude"]
    is_shifted = (snapped_lat != nom_lat or snapped_lon != nom_lon)

    rationale = (
        f"Snapped to {best_candidate['offset_label']} cell (lat={snapped_lat}, lon={snapped_lon}) "
        f"yielding highest hydrologic river channel discharge signal (mean={best_candidate['eval']['mean']} m³/s, "
        f"max={best_candidate['eval']['max']} m³/s)."
        if is_shifted
        else "Nominal coordinates directly align with the main GloFAS river channel cell."
    )

    return {
        "station_id": station_id,
        "nominal_latitude": nom_lat,
        "nominal_longitude": nom_lon,
        "snapped_latitude": snapped_lat,
        "snapped_longitude": snapped_lon,
        "is_viable": True,
        "status": "active",
        "snapped_offset": best_candidate["offset_label"],
        "selection_rationale": rationale,
        "metrics": best_candidate["eval"],
        "all_candidates": candidates,
    }
