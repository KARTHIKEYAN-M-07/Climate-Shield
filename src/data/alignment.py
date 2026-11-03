"""
src/data/alignment.py
─────────────────────
Align cleaned rainfall and discharge DataFrames on a shared daily date index
and write the result to data/processed/.

Design rules
────────────
* Use an INNER join by default so every row has both rainfall and discharge.
* Log and report count of dates present in one but not the other.
* The processed file is the single input for feature engineering (Phase 4).
* Column name "river_discharge" is labelled throughout as
  "modelled river discharge (m³/s)" in metadata—not gauge water level.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Public API
# =============================================================================


def align_datasets(
    df_rain: pd.DataFrame,
    df_disc: pd.DataFrame,
    *,
    how: str = "inner",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge cleaned rainfall and discharge on a shared date index.

    Parameters
    ----------
    df_rain:  Cleaned rainfall DataFrame (must have 'date' column).
    df_disc:  Cleaned discharge DataFrame (must have 'date' column).
    how:      Merge strategy — ``"inner"`` (default) | ``"outer"`` | ``"left"``.

    Returns
    -------
    (aligned_df, alignment_report)
        aligned_df  – merged DataFrame sorted by date.
        alignment_report – dict summarising date coverage, row counts, mismatches.
    """
    rain = df_rain.copy()
    disc = df_disc.copy()

    rain["date"] = pd.to_datetime(rain["date"])
    disc["date"] = pd.to_datetime(disc["date"])

    rain_dates = set(rain["date"])
    disc_dates = set(disc["date"])

    only_rain = sorted(str(d.date()) for d in rain_dates - disc_dates)
    only_disc = sorted(str(d.date()) for d in disc_dates - rain_dates)

    merged = pd.merge(rain, disc, on="date", how=how)
    merged = merged.sort_values("date").reset_index(drop=True)

    # ── Convert date back to plain string for downstream compat ───────────────
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    report: dict[str, Any] = {
        "aligned_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_type": how,
        "rainfall_rows_in": len(rain),
        "discharge_rows_in": len(disc),
        "aligned_rows_out": len(merged),
        "dates_only_in_rainfall": len(only_rain),
        "dates_only_in_discharge": len(only_disc),
        "date_range_start": str(merged["date"].min()) if len(merged) else None,
        "date_range_end": str(merged["date"].max()) if len(merged) else None,
        "sample_only_rain_dates": only_rain[:10],
        "sample_only_disc_dates": only_disc[:10],
        "discharge_label": "modelled river discharge (m³/s) — NOT gauge water level",
        "source_credit": "Weather and flood data by Open-Meteo.com (CC BY 4.0)",
    }

    if only_rain:
        logger.warning(
            "Alignment: %d dates in rainfall but NOT in discharge.", len(only_rain)
        )
    if only_disc:
        logger.warning(
            "Alignment: %d dates in discharge but NOT in rainfall.", len(only_disc)
        )

    logger.info(
        "Alignment complete (%s join): %d rainfall + %d discharge → %d rows.",
        how, len(rain), len(disc), len(merged),
    )
    return merged, report


def save_processed(
    df: pd.DataFrame,
    station_id: str,
    cfg: dict[str, Any],
    *,
    retrieval_date: str | None = None,
    alignment_report: dict[str, Any] | None = None,
) -> Path:
    """Write the aligned DataFrame to data/processed/ with a report sidecar.

    Parameters
    ----------
    df:               Aligned DataFrame.
    station_id:       e.g. ``"TN_CAU_TRICHY"``.
    cfg:              Full data.yaml config dict.
    retrieval_date:   YYYY-MM-DD; defaults to today UTC.
    alignment_report: Dict written alongside as a JSON sidecar.

    Returns
    -------
    Path to the saved CSV.
    """
    if retrieval_date is None:
        retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    processed_dir = _PROJECT_ROOT / cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{station_id}_aligned_{retrieval_date}.csv"
    out_path = processed_dir / filename
    df.to_csv(out_path, index=False)
    logger.info("Processed dataset saved to %s (%d rows).", out_path, len(df))

    if alignment_report is not None:
        report_path = out_path.with_suffix(".alignment.json")
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(alignment_report, fh, indent=2, default=str)

    return out_path


def load_processed(
    station_id: str,
    cfg: dict[str, Any],
    *,
    retrieval_date: str | None = None,
) -> pd.DataFrame:
    """Load the most recent aligned processed CSV for *station_id*.

    If *retrieval_date* is None, picks the lexicographically latest file.

    Raises
    ------
    FileNotFoundError
        If no processed file exists for *station_id*.
    """
    processed_dir = _PROJECT_ROOT / cfg["paths"]["processed_dir"]
    pattern = f"{station_id}_aligned_*.csv"
    files = sorted(processed_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No processed file found for station '{station_id}' "
            f"in {processed_dir}. Run ingestion + cleaning first."
        )

    if retrieval_date:
        target = processed_dir / f"{station_id}_aligned_{retrieval_date}.csv"
        if not target.exists():
            raise FileNotFoundError(f"Processed file not found: {target}")
        chosen = target
    else:
        chosen = files[-1]  # latest by name (date-stamped)

    logger.info("Loading processed dataset from %s", chosen)
    df = pd.read_csv(chosen, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df
