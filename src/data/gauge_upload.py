"""src/data/gauge_upload.py
────────────────────────
External CWC / India-WRIS gauge data validation and ingestion:
  1. Validates uploaded gauge CSV schema (timestamp/date, station_id, value, unit).
  2. Timezone validation & normalization (assumes Asia/Kolkata if tz-naive).
  3. Automatic unit detection & conversion (cm -> m, cusec -> m³/s).
  4. Aligns physical observations with historical rainfall to allow model training.
  5. Computes agreement metrics between GloFAS modelled discharge and observed gauge data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.evaluation.metrics import kling_gupta_efficiency

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
KOLKATA_TZ = timezone(timedelta(hours=5, minutes=30))


def validate_cwc_gauge_csv(
    df: pd.DataFrame,
    target_station_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate external CWC / India-WRIS gauge CSV schema, units, timezone, and physical values.

    Expected Schema:
      - timestamp (or date / datetime)
      - station_id
      - value (or water_level_m / discharge_m3s)
      - unit ('m', 'm3/s', 'cumec', 'cm', 'cusec')

    Returns
    -------
    dict
        Dictionary containing is_valid, cleaned_df, errors, and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df.empty:
        return {"is_valid": False, "errors": ["Uploaded gauge CSV is empty."], "warnings": []}

    # Normalize column names for flexible detection
    col_map = {}
    for c in df.columns:
        clow = c.strip().lower()
        if clow in ["timestamp", "date", "datetime", "time"]:
            col_map["timestamp"] = c
        elif clow in ["station_id", "station", "gauge_id", "stationid"]:
            col_map["station_id"] = c
        elif clow in ["value", "water_level_m", "discharge_m3s", "discharge", "level", "flow", "water_level"]:
            col_map["value"] = c
        elif clow in ["unit", "units", "uom"]:
            col_map["unit"] = c

    # 1. Schema Validation
    if "timestamp" not in col_map:
        errors.append("Missing required date or timestamp column. Expected columns: timestamp (or date), value, station_id, unit.")
    if "value" not in col_map:
        errors.append("Missing required value column. Expected water_level_m, discharge_m3s, or value.")

    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings, "cleaned_df": None, "is_centimeters": False}

    df_clean = df.copy()
    ts_col = col_map["timestamp"]
    val_col = col_map["value"]
    unit_col = col_map.get("unit")

    # Station handling (optional in input CSV; defaults if missing)
    if "station_id" in col_map:
        sid_col = col_map["station_id"]
    else:
        sid_col = "station_id"
        df_clean["station_id"] = target_station_id or "STATION_GAUGE"
        warnings.append(f"No station_id column found: assigned default '{df_clean['station_id'].iloc[0]}'.")

    # 2. Timezone and Date Validation
    try:
        raw_dates = pd.to_datetime(df_clean[ts_col], errors="coerce")
        if raw_dates.isna().all():
            errors.append(f"Could not parse any timestamps in column '{ts_col}'.")
            return {"is_valid": False, "errors": errors, "warnings": warnings, "cleaned_df": None}

        # Check timezone; if naive, localize to Asia/Kolkata
        if raw_dates.dt.tz is None:
            warnings.append("Timestamps are timezone-naive: assumed Indian Standard Time (Asia/Kolkata, UTC+05:30).")
            # Localize to IST then convert to date string
            clean_dates = raw_dates.dt.tz_localize(KOLKATA_TZ).dt.strftime("%Y-%m-%d")
        else:
            clean_dates = raw_dates.dt.tz_convert(KOLKATA_TZ).dt.strftime("%Y-%m-%d")

        df_clean["date"] = clean_dates
    except Exception as exc:
        errors.append(f"Timestamp parsing error: {exc}")
        return {"is_valid": False, "errors": errors, "warnings": warnings, "cleaned_df": None}

    # 3. Station Validation
    unique_sids = df_clean[sid_col].astype(str).unique().tolist()
    if target_station_id and target_station_id not in unique_sids:
        warnings.append(f"Provided station_id '{target_station_id}' not found in CSV station IDs: {unique_sids[:5]}.")

    # 4. Value and Unit Validation
    raw_vals = pd.to_numeric(df_clean[val_col], errors="coerce")
    valid_mask = raw_vals.notna()
    if valid_mask.sum() == 0:
        errors.append(f"No valid numeric readings found in value column '{val_col}'.")
        return {"is_valid": False, "errors": errors, "warnings": warnings, "cleaned_df": None}

    # Check negative values
    if (raw_vals[valid_mask] < 0).any():
        n_neg = int((raw_vals[valid_mask] < 0).sum())
        warnings.append(f"Found {n_neg} negative gauge readings; these will be set to zero.")
        raw_vals = raw_vals.clip(lower=0.0)

    # Unit Conversion
    unit_str = "m"
    is_centimeters = False
    if unit_col and unit_col in df_clean.columns:
        u_sample = str(df_clean[unit_col].dropna().iloc[0]).strip().lower() if not df_clean[unit_col].dropna().empty else "m"
        if u_sample in ["cm", "centimeter", "centimeters"]:
            is_centimeters = True
            raw_vals = raw_vals * 0.01
            unit_str = "m"
            warnings.append("Unit 'cm' detected and converted to meters (scaled by 0.01).")
        elif u_sample in ["cusec", "cfs"]:
            raw_vals = raw_vals * 0.0283168
            unit_str = "m³/s"
            warnings.append("Unit 'cusec' detected and converted to m³/s (scaled by 0.0283168).")
        elif u_sample in ["cumec", "m3/s", "m³/s"]:
            unit_str = "m³/s"
        else:
            unit_str = u_sample
    else:
        # Heuristic detection if unit column omitted
        if raw_vals.median() > 50.0:
            is_centimeters = True
            raw_vals = raw_vals * 0.01
            unit_str = "m"
            warnings.append("Median value > 50 with no unit column: assumed centimeters and converted to meters.")
        else:
            unit_str = "m"
            warnings.append("No unit column found: assumed standard SI unit (meters / m³/s).")

    # Standardized output DataFrame
    output_df = pd.DataFrame({
        "date": df_clean["date"],
        "station_id": df_clean[sid_col].astype(str),
        "gauge_value": raw_vals,
        "gauge_unit": unit_str,
    }).dropna(subset=["date", "gauge_value"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return {
        "is_valid": True,
        "errors": [],
        "warnings": warnings,
        "cleaned_df": output_df,
        "is_centimeters": is_centimeters,
        "total_rows": len(df),
        "valid_rows": len(output_df),
        "date_range": (str(output_df["date"].iloc[0]), str(output_df["date"].iloc[-1])),
        "detected_unit": unit_str,
        "mean_value": round(float(output_df["gauge_value"].mean()), 3),
        "max_value": round(float(output_df["gauge_value"].max()), 3),
    }


def validate_gauge_csv(df: pd.DataFrame) -> dict[str, Any]:
    """Backward-compatible wrapper around validate_cwc_gauge_csv."""
    res = validate_cwc_gauge_csv(df)
    return {
        "is_valid": res["is_valid"],
        "errors": res["errors"],
        "warnings": res["warnings"],
        "date_column": "date",
        "value_column": "gauge_value",
        "is_centimeters": res.get("is_centimeters", False),
        "total_rows": res.get("total_rows", 0),
        "valid_numeric_rows": res.get("valid_rows", 0),
    }


def save_and_align_gauge_dataset(
    df_gauge: pd.DataFrame,
    station_id: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save validated gauge data and align with station rainfall for training."""
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    proc_dir = _PROJECT_ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Save raw gauge CSV
    raw_path = raw_dir / f"{station_id}_gauge_raw.csv"
    df_gauge.to_csv(raw_path, index=False)

    # Find rainfall data for alignment
    proc_files = sorted(proc_dir.glob(f"{station_id}_aligned_*.csv"))
    if proc_files:
        df_base = pd.read_csv(proc_files[-1])
    else:
        # Fallback to interim or raw rainfall
        rain_files = sorted(raw_dir.glob(f"{station_id}_rainfall_raw_*.csv"))
        if not rain_files:
            raise FileNotFoundError(f"No rainfall data found for {station_id} to align with gauge data.")
        df_base = pd.read_csv(rain_files[-1])

    # Merge gauge values into aligned dataset
    df_gauge_sub = df_gauge[["date", "gauge_value"]].copy()
    df_gauge_sub = df_gauge_sub.rename(columns={"gauge_value": "river_discharge"})

    merged = pd.merge(df_base.drop(columns=["river_discharge"], errors="ignore"), df_gauge_sub, on="date", how="inner")
    out_path = proc_dir / f"{station_id}_gauge_aligned.csv"
    merged.to_csv(out_path, index=False)
    logger.info("Saved gauge-aligned training dataset to %s (%d rows)", out_path, len(merged))
    return out_path


def compare_glofas_to_gauge(
    df_glofas: pd.DataFrame,
    df_gauge: pd.DataFrame,
    glofas_col: str = "river_discharge",
    gauge_col: str = "gauge_value",
) -> dict[str, Any]:
    """Align GloFAS modelled discharge with physical gauge observations and compute agreement."""
    g_col = glofas_col if glofas_col in df_glofas.columns else "river_discharge"
    v_col = gauge_col if gauge_col in df_gauge.columns else ("water_level_m" if "water_level_m" in df_gauge.columns else "value")

    df_g = df_glofas[["date", g_col]].copy()
    df_obs = df_gauge[["date", v_col]].copy()

    df_g["date"] = pd.to_datetime(df_g["date"]).dt.strftime("%Y-%m-%d")
    df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.strftime("%Y-%m-%d")

    merged = pd.merge(df_g, df_obs, on="date", how="inner").dropna()
    if len(merged) < 5:
        return {
            "status": "INSUFFICIENT_OVERLAP",
            "overlap_days": len(merged),
            "correlation_r": float("nan"),
            "kge": float("nan"),
            "message": "Fewer than 5 overlapping days between GloFAS and uploaded gauge data.",
        }

    q_model = merged[g_col].values.astype(float)
    q_obs = merged[v_col].values.astype(float)

    r = float(np.corrcoef(q_model, q_obs)[0, 1]) if len(q_model) > 1 else 0.0
    kge = float(kling_gupta_efficiency(q_obs, q_model))

    return {
        "status": "COMPARISON_COMPLETE",
        "overlap_days": len(merged),
        "correlation_r": round(r, 4),
        "kge": round(kge, 4),
        "glofas_mean": round(float(np.mean(q_model)), 2),
        "gauge_mean": round(float(np.mean(q_obs)), 2),
        "message": f"Successfully evaluated agreement over {len(merged)} overlapping observation days.",
    }


def quantile_mapping_bias_correction(
    y_train_model: np.ndarray,
    y_train_obs: np.ndarray,
    y_test_model: np.ndarray,
    n_quantiles: int = 20,
) -> np.ndarray:
    """Non-parametric empirical quantile mapping bias correction."""
    percentiles = np.linspace(0, 100, n_quantiles)
    model_quantiles = np.percentile(y_train_model, percentiles)
    obs_quantiles = np.percentile(y_train_obs, percentiles)
    corrected = np.interp(y_test_model, model_quantiles, obs_quantiles)
    return np.asarray(np.maximum(0.0, corrected), dtype=float)
