"""
src/data/cleaning.py
────────────────────
Clean raw rainfall and discharge DataFrames and write versioned interim files.

Design rules
────────────
* Raw files are NEVER touched.  Every cleaning step writes a NEW versioned file
  to data/interim/.
* Duplicate timestamps: keep first occurrence; log all dropped rows.
* Short gaps (≤ cfg max_consecutive_missing): forward-fill with a flag column.
* Long gaps (> threshold): leave as NaN and flag in a separate boolean column.
* Extreme values are NEVER deleted; they are preserved and a flag column is
  set to True so downstream steps can inspect them.
* Non-negative constraint for rainfall: negative values set to 0 with a flag.
* Returns a cleaned DataFrame; caller is responsible for persisting.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── version counter (bumped each time clean_* is called in a session) ─────────
# For file naming we use retrieval_date + "_v1", "_v2", etc.
_VERSION_TAG = "v1"


# =============================================================================
# Public API
# =============================================================================


def clean_rainfall(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    station_id: str = "UNKNOWN",
    retrieval_date: str | None = None,
    write_interim: bool = True,
) -> pd.DataFrame:
    """Clean a raw rainfall DataFrame.

    Steps (in order):
    1. Ensure 'date' is a proper datetime index.
    2. Remove duplicate dates (keep first; log dropped rows).
    3. Reindex to a full daily range; gaps become NaN.
    4. Forward-fill short gaps (≤ max_consecutive_missing); set flag column.
    5. Flag remaining NaN rows (long gaps) — NOT filled.
    6. Enforce non-negative: set negative values to 0 + flag.
    7. Flag extreme values (> max_mm) — NOT removed.
    8. Write versioned CSV to data/interim/ if write_interim=True.

    Parameters
    ----------
    df:             Raw DataFrame with columns [date, precipitation_sum].
    cfg:            Full data.yaml config dict.
    station_id:     For filename and provenance.
    retrieval_date: YYYY-MM-DD string; defaults to today UTC.
    write_interim:  If True, write cleaned file to data/interim/.

    Returns
    -------
    pd.DataFrame with added boolean flag columns:
        precipitation_sum_ffill_flag  – True where forward-fill was applied.
        precipitation_sum_gap_flag    – True where NaN remains (long gap).
        precipitation_sum_neg_flag    – True where negative was zeroed.
        precipitation_sum_extreme_flag – True where value > max threshold.
    """
    if retrieval_date is None:
        retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rules = cfg.get("validation", {})
    max_consec = rules.get("max_consecutive_missing", 7)
    max_mm = rules.get("rainfall_max_mm", 600.0)

    df = df.copy()
    df = _ensure_date_index(df, "precipitation_sum")

    # ── 1 deduplicate ─────────────────────────────────────────────────────────
    n_before = len(df)
    df = _deduplicate(df)
    if len(df) < n_before:
        logger.info("Rainfall: removed %d duplicate dates.", n_before - len(df))

    # ── 2 full daily reindex ──────────────────────────────────────────────────
    df = _reindex_daily(df)

    # ── 3 forward-fill short gaps ─────────────────────────────────────────────
    df, n_ffill = _forward_fill_short(df, "precipitation_sum", max_consec)
    if n_ffill:
        logger.info("Rainfall: forward-filled %d short-gap rows.", n_ffill)

    # ── 4 flag remaining gaps (long) ──────────────────────────────────────────
    df["precipitation_sum_gap_flag"] = df["precipitation_sum"].isna()

    # ── 5 non-negative enforcement ────────────────────────────────────────────
    neg_mask = df["precipitation_sum"].notna() & (df["precipitation_sum"] < 0)
    df["precipitation_sum_neg_flag"] = neg_mask
    df.loc[neg_mask, "precipitation_sum"] = 0.0
    if neg_mask.sum():
        logger.warning("Rainfall: set %d negative values to 0.", int(neg_mask.sum()))

    # ── 6 flag extremes (NOT removed) ─────────────────────────────────────────
    extreme_mask = df["precipitation_sum"].notna() & (df["precipitation_sum"] > max_mm)
    df["precipitation_sum_extreme_flag"] = extreme_mask
    if extreme_mask.sum():
        logger.warning(
            "Rainfall: flagged %d extreme values (> %.0f mm) — NOT removed.",
            int(extreme_mask.sum()), max_mm,
        )

    df = df.reset_index()  # date → column

    if write_interim:
        _write_interim(df, station_id, "rainfall", cfg, retrieval_date)

    return df


def clean_discharge(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    station_id: str = "UNKNOWN",
    retrieval_date: str | None = None,
    write_interim: bool = True,
) -> pd.DataFrame:
    """Clean a raw discharge DataFrame.

    Same pipeline as clean_rainfall but for river_discharge.
    Discharge must be ≥ 0; no upper-clip (extremes flagged only).

    Returns
    -------
    pd.DataFrame with flag columns:
        river_discharge_ffill_flag
        river_discharge_gap_flag
        river_discharge_neg_flag
        river_discharge_extreme_flag
    """
    if retrieval_date is None:
        retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rules = cfg.get("validation", {})
    max_consec = rules.get("max_consecutive_missing", 7)
    max_q = rules.get("discharge_max_m3s", 50000.0)

    df = df.copy()
    df = _ensure_date_index(df, "river_discharge")

    n_before = len(df)
    df = _deduplicate(df)
    if len(df) < n_before:
        logger.info("Discharge: removed %d duplicate dates.", n_before - len(df))

    df = _reindex_daily(df)

    df, n_ffill = _forward_fill_short(df, "river_discharge", max_consec)
    if n_ffill:
        logger.info("Discharge: forward-filled %d short-gap rows.", n_ffill)

    df["river_discharge_gap_flag"] = df["river_discharge"].isna()

    neg_mask = df["river_discharge"].notna() & (df["river_discharge"] < 0)
    df["river_discharge_neg_flag"] = neg_mask
    df.loc[neg_mask, "river_discharge"] = 0.0
    if neg_mask.sum():
        logger.warning("Discharge: set %d negative values to 0.", int(neg_mask.sum()))

    extreme_mask = df["river_discharge"].notna() & (df["river_discharge"] > max_q)
    df["river_discharge_extreme_flag"] = extreme_mask
    if extreme_mask.sum():
        logger.warning(
            "Discharge: flagged %d extreme values (> %.0f m³/s) — NOT removed.",
            int(extreme_mask.sum()), max_q,
        )

    df = df.reset_index()

    if write_interim:
        _write_interim(df, station_id, "discharge", cfg, retrieval_date)

    return df


# =============================================================================
# Private helpers
# =============================================================================


def _ensure_date_index(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Parse 'date' column and set it as a named DatetimeIndex."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.name = "date"          # ← reset_index() restores column name as "date"
    df = df[[value_col] + [c for c in df.columns if c != value_col]]
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep first occurrence of each duplicate index (date)."""
    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        logger.debug("Dropping %d duplicate index rows.", n_dupes)
        df = df[~df.index.duplicated(keep="first")]
    return df


def _reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Extend to a complete daily DatetimeIndex; introduce NaN for missing dates."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    return df.reindex(full_idx)


def _forward_fill_short(
    df: pd.DataFrame,
    col: str,
    max_consec: int,
) -> tuple[pd.DataFrame, int]:
    """Forward-fill runs of NaN that are ≤ max_consec in length.

    Returns the modified DataFrame and the count of filled cells.
    """
    flag_col = f"{col}_ffill_flag"
    df[flag_col] = False

    series = df[col].copy()
    filled = series.ffill(limit=max_consec)
    ffilled_mask = series.isna() & filled.notna()

    df.loc[ffilled_mask, col] = filled[ffilled_mask]
    df.loc[ffilled_mask, flag_col] = True

    return df, int(ffilled_mask.sum())


def _write_interim(
    df: pd.DataFrame,
    station_id: str,
    dataset: str,
    cfg: dict[str, Any],
    retrieval_date: str,
) -> Path:
    """Write cleaned DataFrame to data/interim/ with a versioned filename."""
    interim_dir = _PROJECT_ROOT / cfg["paths"]["interim_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{station_id}_{dataset}_clean_{retrieval_date}_{_VERSION_TAG}.csv"
    out_path = interim_dir / filename
    df.to_csv(out_path, index=False)
    logger.info("Interim %s written to %s (%d rows).", dataset, out_path, len(df))
    return out_path
