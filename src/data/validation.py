"""
src/data/validation.py
──────────────────────
Validate a raw DataFrame (rainfall or discharge) against the rules in
configs/data.yaml and write a JSON report to reports/.

Design rules
────────────
* NEVER delete or clip extreme values.  Flag them in the report only.
* Duplicate timestamps are flagged and counted.
* Missing values are counted globally and per consecutive run.
* Out-of-range values are flagged with their index positions.
* The report is the single source of truth for data quality in the pipeline.
* A WARNING is logged if any threshold is exceeded; data is still returned.
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


def validate_dataframe(
    df: pd.DataFrame,
    dataset: str,
    station_id: str,
    cfg: dict[str, Any],
    *,
    write_report: bool = True,
    retrieval_date: str | None = None,
) -> dict[str, Any]:
    """Validate *df* against rules in *cfg* and optionally write a JSON report.

    Parameters
    ----------
    df:             Raw DataFrame (columns: date + value column).
    dataset:        ``"rainfall"`` or ``"discharge"``.
    station_id:     e.g. ``"TN_CAU_TRICHY"``.
    cfg:            Full data.yaml config dict.
    write_report:   If True, write JSON report to reports/.
    retrieval_date: Date string for the filename (defaults to today UTC).

    Returns
    -------
    dict
        Validation report (mirrors the JSON written to disk).
    """
    if retrieval_date is None:
        retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rules = cfg.get("validation", {})
    total_rows = len(df)

    report: dict[str, Any] = {
        "station_id": station_id,
        "dataset": dataset,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval_date": retrieval_date,
        "total_rows": total_rows,
        "issues": [],
        "warnings": [],
        "summary": {},
    }

    if total_rows == 0:
        _add_issue(report, "CRITICAL", "empty_dataframe",
                   "DataFrame has zero rows. Ingestion may have failed.")
        _finalize_and_write(report, dataset, station_id, cfg, retrieval_date,
                            write_report)
        return report

    # ── 1. Date column ─────────────────────────────────────────────────────────
    _check_date_column(df, report)

    # ── 2. Duplicates ──────────────────────────────────────────────────────────
    _check_duplicates(df, report)

    # ── 3. Missing values ──────────────────────────────────────────────────────
    value_col = _value_column(df, dataset)
    _check_missing(df, value_col, rules, report)

    # ── 4. Range checks ────────────────────────────────────────────────────────
    _check_range(df, value_col, dataset, rules, report)

    # ── 5. Temporal coverage ──────────────────────────────────────────────────
    _check_coverage(df, cfg, report)

    # ── 6. Summary stats ──────────────────────────────────────────────────────
    if value_col and value_col in df.columns:
        series = df[value_col].dropna()
        report["summary"] = {
            "value_column": value_col,
            "count_non_null": int(series.count()),
            "mean": round(float(series.mean()), 4) if len(series) else None,
            "std": round(float(series.std()), 4) if len(series) > 1 else None,
            "min": round(float(series.min()), 4) if len(series) else None,
            "p25": round(float(series.quantile(0.25)), 4) if len(series) else None,
            "p50": round(float(series.quantile(0.50)), 4) if len(series) else None,
            "p75": round(float(series.quantile(0.75)), 4) if len(series) else None,
            "p95": round(float(series.quantile(0.95)), 4) if len(series) else None,
            "max": round(float(series.max()), 4) if len(series) else None,
        }

    report["passed"] = len(report["issues"]) == 0

    _finalize_and_write(report, dataset, station_id, cfg, retrieval_date,
                        write_report)
    return report


# =============================================================================
# Private checkers
# =============================================================================


def _value_column(df: pd.DataFrame, dataset: str) -> str | None:
    """Return the primary value column name for a dataset."""
    mapping = {
        "rainfall": "precipitation_sum",
        "discharge": "river_discharge",
    }
    col = mapping.get(dataset)
    if col and col in df.columns:
        return col
    # Fall back to first non-date column
    for c in df.columns:
        if c != "date":
            return c
    return None


def _check_date_column(df: pd.DataFrame, report: dict) -> None:
    if "date" not in df.columns:
        _add_issue(report, "CRITICAL", "missing_date_column",
                   "No 'date' column found in DataFrame.")
        return

    # Try parsing dates
    bad_dates = []
    for i, val in enumerate(df["date"]):
        try:
            pd.Timestamp(val)
        except Exception:
            bad_dates.append({"row": i, "value": str(val)})

    if bad_dates:
        _add_issue(report, "ERROR", "unparseable_dates",
                   f"{len(bad_dates)} date(s) could not be parsed.",
                   detail=bad_dates[:10])


def _check_duplicates(df: pd.DataFrame, report: dict) -> None:
    if "date" not in df.columns:
        return
    dupes = df[df.duplicated(subset=["date"], keep=False)]
    n_dupes = len(dupes)
    report["duplicate_timestamps"] = n_dupes
    if n_dupes > 0:
        dupe_dates = sorted(dupes["date"].astype(str).unique().tolist())
        _add_issue(report, "WARNING", "duplicate_timestamps",
                   f"{n_dupes} duplicate date rows found.",
                   detail=dupe_dates[:20])
        logger.warning("Validation: %d duplicate timestamps in dataset.", n_dupes)


def _check_missing(
    df: pd.DataFrame,
    value_col: str | None,
    rules: dict,
    report: dict,
) -> None:
    if not value_col or value_col not in df.columns:
        return

    series = df[value_col]
    n_missing = int(series.isna().sum())
    total = len(series)
    pct_missing = round(100.0 * n_missing / total, 2) if total else 0.0

    report["missing_values"] = {
        "count": n_missing,
        "pct": pct_missing,
    }

    max_pct = rules.get("max_missing_pct", 10.0)
    if pct_missing > max_pct:
        _add_issue(report, "WARNING", "high_missing_pct",
                   f"{pct_missing}% missing values exceed threshold {max_pct}%.")
        logger.warning("Validation: %.1f%% missing values.", pct_missing)

    # Consecutive missing runs
    max_consec = rules.get("max_consecutive_missing", 7)
    consec = _longest_consecutive_missing(series)
    report["missing_values"]["max_consecutive"] = consec
    if consec > max_consec:
        _add_issue(report, "WARNING", "consecutive_missing",
                   f"Longest consecutive missing run = {consec} rows "
                   f"(threshold: {max_consec}).")
        logger.warning("Validation: %d consecutive missing rows.", consec)


def _longest_consecutive_missing(series: pd.Series) -> int:
    """Return the longest run of consecutive NaN values."""
    max_run = 0
    current = 0
    for val in series:
        if pd.isna(val):
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _check_range(
    df: pd.DataFrame,
    value_col: str | None,
    dataset: str,
    rules: dict,
    report: dict,
) -> None:
    if not value_col or value_col not in df.columns:
        return

    series = df[value_col].dropna()

    if dataset == "rainfall":
        lo = rules.get("rainfall_min_mm", 0.0)
        hi = rules.get("rainfall_max_mm", 600.0)
        unit = "mm/day"
    elif dataset == "discharge":
        lo = rules.get("discharge_min_m3s", 0.0)
        hi = rules.get("discharge_max_m3s", 50000.0)
        unit = "m³/s"
    else:
        return

    below = series[series < lo]
    above = series[series > hi]

    report["out_of_range"] = {
        "below_min": int(len(below)),
        "above_max": int(len(above)),
        "min_bound": lo,
        "max_bound": hi,
        "unit": unit,
    }

    if len(below):
        _add_issue(report, "WARNING", "values_below_minimum",
                   f"{len(below)} value(s) below physical minimum {lo} {unit}. "
                   "Values FLAGGED but NOT removed.",
                   detail={"indices": below.index.tolist()[:20],
                           "values": below.tolist()[:20]})
        logger.warning("Validation: %d values below minimum.", len(below))

    if len(above):
        _add_issue(report, "WARNING", "values_above_maximum",
                   f"{len(above)} value(s) above physical maximum {hi} {unit}. "
                   "Values FLAGGED but NOT removed (extreme values preserved).",
                   detail={"indices": above.index.tolist()[:20],
                           "values": above.tolist()[:20]})
        logger.warning("Validation: %d values above maximum.", len(above))


def _check_coverage(
    df: pd.DataFrame,
    cfg: dict,
    report: dict,
) -> None:
    if "date" not in df.columns:
        return

    period = cfg.get("period", {})
    expected_start = period.get("start")
    expected_end = period.get("end")

    if not expected_start or not expected_end:
        return

    actual_start = str(df["date"].min())
    actual_end = str(df["date"].max())

    report["coverage"] = {
        "expected_start": expected_start,
        "expected_end": expected_end,
        "actual_start": actual_start,
        "actual_end": actual_end,
    }

    if actual_start > expected_start:
        _add_issue(report, "WARNING", "coverage_start_late",
                   f"Data starts {actual_start}, expected {expected_start}.")

    if actual_end < expected_end:
        _add_issue(report, "WARNING", "coverage_end_early",
                   f"Data ends {actual_end}, expected {expected_end}.")


# =============================================================================
# Report helpers
# =============================================================================


def _add_issue(
    report: dict,
    level: str,
    code: str,
    message: str,
    detail: Any = None,
) -> None:
    entry: dict[str, Any] = {"level": level, "code": code, "message": message}
    if detail is not None:
        entry["detail"] = detail
    if level == "WARNING":
        report["warnings"].append(entry)
    else:
        report["issues"].append(entry)


def _finalize_and_write(
    report: dict,
    dataset: str,
    station_id: str,
    cfg: dict,
    retrieval_date: str,
    write_report: bool,
) -> None:
    report.setdefault("passed", len(report.get("issues", [])) == 0)

    if not write_report:
        return

    reports_dir = _PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    template = cfg.get("filename_templates", {}).get(
        "validation_report",
        "{station_id}_{dataset}_validation_{date}.json",
    )
    filename = template.format(
        station_id=station_id,
        dataset=dataset,
        date=retrieval_date,
    )
    out_path = reports_dir / filename

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    logger.info("Validation report written to %s", out_path)
