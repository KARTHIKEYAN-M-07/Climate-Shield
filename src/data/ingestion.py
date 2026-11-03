"""
src/data/ingestion.py
─────────────────────
Fetch daily rainfall (precipitation_sum) and modelled river discharge
(river_discharge) from Open-Meteo APIs.

Design rules
────────────
* Raw files are NEVER modified after writing.  One file per (station, variable,
  retrieval-date).
* HTTP 429  → wait `rate_limit_pause_s` seconds, then retry.
* Timeout / connection error → exponential back-off up to `max_retries`.
* Bad / empty JSON → raise `IngestionError`; never silently return empty data.
* All timestamps normalised to UTC-naive ISO-8601 date strings.

Credit: Weather and flood data by Open-Meteo.com (CC BY 4.0)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

# ── project root (two levels up from this file) ───────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Custom exception
# =============================================================================


class IngestionError(RuntimeError):
    """Raised when data cannot be fetched or parsed."""


# =============================================================================
# Internal HTTP helper
# =============================================================================


def _get_with_retry(
    url: str,
    params: dict[str, Any],
    timeout_s: int,
    max_retries: int,
    backoff_factor: int,
    rate_limit_pause_s: int,
) -> dict[str, Any]:
    """GET *url* with query *params*, retry on transient errors.

    Parameters
    ----------
    url:               API base URL.
    params:            Query-string parameters.
    timeout_s:         Per-attempt read/connect timeout in seconds.
    max_retries:       Maximum number of retry attempts after initial failure.
    backoff_factor:    Multiplier for exponential back-off (seconds).
    rate_limit_pause_s: Extra pause (seconds) on HTTP 429 before retrying.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    IngestionError
        On exhausted retries, bad JSON, or an unexpected HTTP error.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            logger.debug("GET %s attempt %d/%d params=%s", url, attempt + 1,
                         max_retries + 1, params)
            resp = requests.get(url, params=params, timeout=timeout_s)

            if resp.status_code == 429:
                wait = rate_limit_pause_s
                logger.warning("HTTP 429 – rate limited. Waiting %ds …", wait)
                time.sleep(wait)
                continue  # retry without counting as a back-off attempt

            if resp.status_code != 200:
                raise IngestionError(
                    f"HTTP {resp.status_code} from {url}: {resp.text[:300]}"
                )

            try:
                data = resp.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise IngestionError(
                    f"Invalid JSON from {url}: {exc}  body={resp.text[:200]}"
                ) from exc

            if not data:
                raise IngestionError(f"Empty JSON response from {url}")

            return data

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            wait = backoff_factor ** attempt
            logger.warning(
                "Timeout on attempt %d/%d – retrying in %ds …",
                attempt + 1, max_retries + 1, wait,
            )
            time.sleep(wait)

        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            wait = backoff_factor ** attempt
            logger.warning(
                "Connection error on attempt %d/%d – retrying in %ds …",
                attempt + 1, max_retries + 1, wait,
            )
            time.sleep(wait)

    raise IngestionError(
        f"All {max_retries + 1} attempts failed for {url}. "
        f"Last error: {last_exc}"
    ) from last_exc


# =============================================================================
# Public fetch functions
# =============================================================================


def fetch_rainfall(
    station: dict[str, Any],
    cfg: dict[str, Any],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch daily precipitation from Open-Meteo Historical Weather API.

    Parameters
    ----------
    station:    Station dict from data.yaml (must have latitude, longitude).
    cfg:        Full data.yaml config dict.
    start_date: Override period start (YYYY-MM-DD).  Falls back to cfg period.
    end_date:   Override period end   (YYYY-MM-DD).  Falls back to cfg period.

    Returns
    -------
    pd.DataFrame
        Columns: date (object, YYYY-MM-DD), precipitation_sum (float, mm/day).
        Sorted ascending by date.

    Raises
    ------
    IngestionError
        On HTTP failure, bad JSON, or missing expected keys.
    """
    api = cfg["rainfall_api"]
    period = cfg["period"]

    params: dict[str, Any] = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": start_date or period["start"],
        "end_date": end_date or period["end"],
        "daily": ",".join(api["variables"]),
        "timezone": station.get("timezone", "UTC"),
    }

    logger.info(
        "Fetching rainfall for %s (%s → %s) …",
        station["id"], params["start_date"], params["end_date"],
    )

    data = _get_with_retry(
        url=api["base_url"],
        params=params,
        timeout_s=api["timeout_s"],
        max_retries=api["max_retries"],
        backoff_factor=api["backoff_factor"],
        rate_limit_pause_s=api["rate_limit_pause_s"],
    )

    _check_keys(data, ["daily"], source="rainfall API")
    daily = data["daily"]
    _check_keys(daily, ["time", "precipitation_sum"], source="rainfall daily block")

    df = pd.DataFrame({
        "date": daily["time"],
        "precipitation_sum": daily["precipitation_sum"],
    })
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Rainfall fetched: %d rows for %s", len(df), station["id"])
    return df


def fetch_discharge(
    station: dict[str, Any],
    cfg: dict[str, Any],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch daily modelled river discharge from Open-Meteo Flood API.

    IMPORTANT: values are *modelled* river discharge (m³/s), NOT measured
    gauge water levels.  Always label them as "modelled river discharge (m³/s)".

    Parameters
    ----------
    station:    Station dict from data.yaml.
    cfg:        Full data.yaml config dict.
    start_date: Override period start (YYYY-MM-DD).
    end_date:   Override period end   (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Columns: date (object, YYYY-MM-DD),
                 river_discharge (float, m³/s – modelled).
        Sorted ascending by date.

    Raises
    ------
    IngestionError
        On HTTP failure, bad JSON, or missing expected keys.
    """
    api = cfg["discharge_api"]
    period = cfg["period"]

    params: dict[str, Any] = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": start_date or period["start"],
        "end_date": end_date or period["end"],
        "daily": ",".join(api["variables"]),
        "ensemble": "false",
    }

    logger.info(
        "Fetching discharge for %s (%s → %s) …",
        station["id"], params["start_date"], params["end_date"],
    )

    data = _get_with_retry(
        url=api["base_url"],
        params=params,
        timeout_s=api["timeout_s"],
        max_retries=api["max_retries"],
        backoff_factor=api["backoff_factor"],
        rate_limit_pause_s=api["rate_limit_pause_s"],
    )

    _check_keys(data, ["daily"], source="discharge API")
    daily = data["daily"]
    _check_keys(daily, ["time", "river_discharge"], source="discharge daily block")

    df = pd.DataFrame({
        "date": daily["time"],
        "river_discharge": daily["river_discharge"],
    })
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Discharge fetched: %d rows for %s", len(df), station["id"])
    return df


# =============================================================================
# Raw-file persistence
# =============================================================================


def save_raw(
    df: pd.DataFrame,
    station_id: str,
    dataset: str,
    cfg: dict[str, Any],
    retrieval_date: str | None = None,
) -> Path:
    """Write *df* to data/raw/ as a versioned CSV and update provenance.

    The file is written once and never modified.  A companion ``_meta.json``
    records the retrieval provenance.

    Parameters
    ----------
    df:             DataFrame to save.
    station_id:     e.g. ``"TN_CAU_TRICHY"``.
    dataset:        ``"rainfall"`` or ``"discharge"``.
    cfg:            Full data.yaml config dict.
    retrieval_date: YYYY-MM-DD string; defaults to today (UTC).

    Returns
    -------
    Path
        Absolute path to the saved CSV.
    """
    if retrieval_date is None:
        retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    raw_dir = _PROJECT_ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    template = cfg["filename_templates"].get(
        f"raw_{dataset}",
        "{station_id}_{dataset}_raw_{date}.csv",
    )
    filename = template.format(
        station_id=station_id,
        dataset=dataset,
        date=retrieval_date,
    )
    csv_path = raw_dir / filename

    if csv_path.exists():
        logger.warning(
            "Raw file %s already exists – skipping write (raw files are immutable).",
            csv_path,
        )
        return csv_path

    df.to_csv(csv_path, index=False)
    logger.info("Saved raw %s to %s (%d rows)", dataset, csv_path, len(df))

    # ── provenance sidecar ────────────────────────────────────────────────────
    meta = {
        "station_id": station_id,
        "dataset": dataset,
        "retrieval_date_utc": retrieval_date,
        "rows": len(df),
        "columns": list(df.columns),
        "source_credit": cfg.get("retrieval", {}).get(
            "source_credit", "Open-Meteo.com (CC BY 4.0)"
        ),
        "retrieved_by": "src/data/ingestion.py",
    }
    meta_path = csv_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return csv_path


# =============================================================================
# Convenience: ingest one station end-to-end
# =============================================================================


def ingest_station(
    station_id: str,
    cfg: dict[str, Any],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    retrieval_date: str | None = None,
) -> dict[str, Path]:
    """Fetch rainfall + discharge for *station_id* and save raw CSVs.

    Parameters
    ----------
    station_id:     Must match a station id in cfg.
    cfg:            Full data.yaml dict.
    start_date:     Override period start.
    end_date:       Override period end.
    retrieval_date: Date string for filenames (defaults to today UTC).

    Returns
    -------
    dict with keys ``"rainfall"`` and ``"discharge"`` mapping to saved Paths.
    """
    from src.utils.config import get_station  # local import avoids circular dep

    station = get_station(cfg, station_id)

    df_rain = fetch_rainfall(station, cfg, start_date=start_date, end_date=end_date)
    df_disc = fetch_discharge(station, cfg, start_date=start_date, end_date=end_date)

    paths = {
        "rainfall": save_raw(df_rain, station_id, "rainfall", cfg, retrieval_date),
        "discharge": save_raw(df_disc, station_id, "discharge", cfg, retrieval_date),
    }
    return paths


# =============================================================================
# Private helpers
# =============================================================================


def _check_keys(obj: dict, keys: list[str], *, source: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise IngestionError(
            f"Missing keys {missing} in {source}. Got: {list(obj.keys())}"
        )
