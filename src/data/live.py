"""src/data/live.py
─────────────────
Near-real-time live data retrieval and caching layer for ClimateShield AI.

Features:
  1. Fetches recent precipitation (Open-Meteo Forecast API) and modelled river discharge (Open-Meteo Flood API).
  2. Isolates future forecast rainfall into a separate display-only column: `forecast_precipitation_sum`.
  3. Splits series into PAST/OBSERVED (date < today, Asia/Kolkata) and FUTURE/FORECAST (date >= today, incomplete day) via `is_forecast`.
  4. Merges live window with stored historical archive for complete lag coverage.
  5. Resolves overlapping dates (live observations overwrite historical records) and tracks provenance in `source` column.
  6. Defensive retry, timeout, rate-limit, and connection-error handling with graceful fallback to `data/live_cache/`.
  7. Saves every successful fetch to `data/live_cache/<station_id>_<UTC_timestamp>.csv`.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

from src.data.ingestion import IngestionError, _get_with_retry

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
KOLKATA_TZ = timezone(timedelta(hours=5, minutes=30))
_NETWORK_AVAILABLE: Optional[bool] = None


def is_network_available() -> bool:
    """Quick check for external HTTP connectivity with memoization."""
    global _NETWORK_AVAILABLE
    if _NETWORK_AVAILABLE is not None:
        return _NETWORK_AVAILABLE
    try:
        resp = requests.get("https://archive-api.open-meteo.com", timeout=1.0)
        _NETWORK_AVAILABLE = True
    except Exception:
        _NETWORK_AVAILABLE = False
    return _NETWORK_AVAILABLE

FLAG_COLUMNS = [
    "precipitation_sum_ffill_flag",
    "precipitation_sum_gap_flag",
    "precipitation_sum_neg_flag",
    "precipitation_sum_extreme_flag",
    "river_discharge_ffill_flag",
    "river_discharge_gap_flag",
    "river_discharge_neg_flag",
    "river_discharge_extreme_flag",
]


def load_live_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration for live data ingestion and caching."""
    cfg_path = config_path or (_PROJECT_ROOT / "configs" / "live.yaml")
    if not cfg_path.exists():
        # Sensible offline defaults
        return {
            "live_weather_api": {
                "base_url": "https://api.open-meteo.com/v1/forecast",
                "daily_variables": ["precipitation_sum"],
                "past_days": 92,
                "forecast_days": 7,
                "timezone": "Asia/Kolkata",
                "timeout_s": 20,
                "max_retries": 3,
                "backoff_factor": 2,
            },
            "live_discharge_api": {
                "base_url": "https://flood-api.open-meteo.com/v1/flood",
                "daily_variables": ["river_discharge"],
                "past_days": 92,
                "forecast_days": 7,
                "timeout_s": 20,
                "max_retries": 3,
                "backoff_factor": 2,
            },
            "cache": {
                "ttl_seconds": 1800,
                "cache_dir": "data/live_cache",
                "max_data_stale_hours": 48.0,
                "max_cached_files_per_station": 5,
            },
        }

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_live_rainfall(
    station: Dict[str, Any],
    api_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Fetch recent daily precipitation from Open-Meteo Weather Forecast API.

    Timezone synchronization note:
    ------------------------------
    Uses the station's configured timezone from configs/data.yaml (default 'Asia/Kolkata').
    This guarantees exact temporal parity with the historical training archive, ensuring
    daily aggregation windows (00:00 - 23:59 IST) align perfectly and eliminating
    spurious phase-shift drift.
    """
    tz = station.get("timezone") or api_cfg.get("timezone", "Asia/Kolkata")
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "daily": ",".join(api_cfg.get("daily_variables", ["precipitation_sum"])),
        "past_days": api_cfg.get("past_days", 92),
        "forecast_days": api_cfg.get("forecast_days", 7),
        "timezone": tz,
    }

    data = _get_with_retry(
        url=api_cfg["base_url"],
        params=params,
        timeout_s=api_cfg.get("timeout_s", 3),
        max_retries=api_cfg.get("max_retries", 1),
        backoff_factor=api_cfg.get("backoff_factor", 1),
        rate_limit_pause_s=api_cfg.get("rate_limit_pause_s", 65),
    )

    daily = data.get("daily", {})
    if "time" not in daily or "precipitation_sum" not in daily:
        raise IngestionError("Open-Meteo weather response missing required daily precipitation fields")

    times = daily["time"]
    precip = daily["precipitation_sum"]

    df = pd.DataFrame({"date": times, "raw_precip": precip})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Treat today as incomplete: split into PAST/OBSERVED (date < today, Asia/Kolkata) and FUTURE/FORECAST (date >= today)
    today_kolkata_str = datetime.now(KOLKATA_TZ).strftime("%Y-%m-%d")
    df["is_forecast"] = (df["date"] >= today_kolkata_str).astype("boolean").astype(bool)
    df["precipitation_sum"] = df["raw_precip"].where(~df["is_forecast"], np.nan)
    df["forecast_precipitation_sum"] = df["raw_precip"].where(df["is_forecast"], np.nan)
    df = df.drop(columns=["raw_precip"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_live_discharge(
    station: Dict[str, Any],
    api_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Fetch recent daily modelled river discharge from Open-Meteo Flood API."""
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "daily": ",".join(api_cfg.get("daily_variables", ["river_discharge"])),
        "past_days": api_cfg.get("past_days", 92),
        "forecast_days": api_cfg.get("forecast_days", 7),
    }

    data = _get_with_retry(
        url=api_cfg["base_url"],
        params=params,
        timeout_s=api_cfg.get("timeout_s", 3),
        max_retries=api_cfg.get("max_retries", 1),
        backoff_factor=api_cfg.get("backoff_factor", 1),
        rate_limit_pause_s=api_cfg.get("rate_limit_pause_s", 65),
    )

    daily = data.get("daily", {})
    if "time" not in daily or "river_discharge" not in daily:
        raise IngestionError("Open-Meteo flood response missing required daily river_discharge fields")

    times = daily["time"]
    discharge = daily["river_discharge"]

    df = pd.DataFrame({"date": times, "raw_discharge": discharge})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Treat today as incomplete: split into PAST/OBSERVED (date < today, Asia/Kolkata) and FUTURE/FORECAST (date >= today)
    today_kolkata_str = datetime.now(KOLKATA_TZ).strftime("%Y-%m-%d")
    df["is_forecast"] = (df["date"] >= today_kolkata_str).astype("boolean").astype(bool)
    df["river_discharge"] = df["raw_discharge"].where(~df["is_forecast"], np.nan)
    df["api_forecast_river_discharge"] = df["raw_discharge"].where(df["is_forecast"], np.nan)
    df = df.drop(columns=["raw_discharge"]).sort_values("date").reset_index(drop=True)
    return df


def merge_live_with_history(
    live_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge live observations with historical dataset.

    Rule: Only overwrite historical values with live values when the live value is not null.
    Provenance recorded in 'source' column ('live_api' vs 'historical_archive').
    """
    if history_df.empty:
        res = live_df.copy()
        if "source" not in res.columns:
            res["source"] = "live_api"
        return res
    if live_df.empty:
        res = history_df.copy()
        if "source" not in res.columns:
            res["source"] = "historical_archive"
        return res

    hist = history_df.copy()
    if "source" not in hist.columns:
        hist["source"] = "historical_archive"

    live = live_df.copy()
    if "source" not in live.columns:
        live["source"] = "live_api"

    hist = hist.drop_duplicates(subset=["date"]).set_index("date")
    live = live.drop_duplicates(subset=["date"]).set_index("date")

    all_dates = hist.index.union(live.index).sort_values()
    merged = hist.reindex(all_dates)

    # Only overwrite historical values where live value is not null (notna)
    for col in live.columns:
        if col not in merged.columns:
            merged[col] = live[col]
        else:
            live_notna = live[col].dropna()
            if not live_notna.empty:
                merged.loc[live_notna.index, col] = live_notna

    # Track provenance: set source to live_api for dates where live had valid observations
    live_has_data = live.dropna(how="all", subset=[c for c in ["precipitation_sum", "river_discharge"] if c in live.columns])
    if not live_has_data.empty:
        merged.loc[live_has_data.index, "source"] = "live_api"

    merged = merged.reset_index()
    return merged


def check_live_source_consistency(
    live_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    station_meta: Optional[Dict[str, Any]] = None,
    live_cfg: Optional[Dict[str, Any]] = None,
    max_bias_mae_mm: float = 5.0,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit agreement between live API observations and historical archive source over last 60 complete days.

    Compares the last 60 days of complete observations between live data and archive endpoints/data:
      - Relative bias per variable: |mean_live - mean_archive| / max(mean_archive, 1e-4)
      - 14-day-sum difference per variable: max absolute difference of rolling 14-day cumulative sums
    Evaluates against separate tolerances defined in configs/live.yaml under `source_consistency`.
    Saves and updates results to reports/source_consistency.csv.
    """
    station_id = (station_meta.get("id") if station_meta else None) or "UNKNOWN"
    cfg = live_cfg or load_live_config()
    sc_cfg = cfg.get("source_consistency", {})
    comparison_days = int(sc_cfg.get("comparison_days", 60))

    rain_tol = sc_cfg.get("rainfall", {})
    max_rel_bias_rain = float(rain_tol.get("max_relative_bias", 0.15))
    max_14d_diff_rain = float(rain_tol.get("max_14d_sum_diff_mm", 20.0))

    dis_tol = sc_cfg.get("discharge", {})
    max_rel_bias_dis = float(dis_tol.get("max_relative_bias", 0.15))
    max_14d_diff_dis = float(dis_tol.get("max_14d_sum_diff_m3s", 50.0))

    rep_dir = reports_dir or (_PROJECT_ROOT / "reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    report_csv = rep_dir / "source_consistency.csv"

    if live_df.empty or "date" not in live_df.columns:
        return {
            "has_overlap": False,
            "has_comparison": False,
            "comparison_days": 0,
            "precip_mae": 0.0,
            "discharge_mae": 0.0,
            "precip_consistent": True,
            "discharge_consistent": True,
            "is_consistent": True,
        }

    # Slice live to completed past rows only
    if "is_forecast" in live_df.columns:
        is_fc = live_df["is_forecast"].astype("boolean").fillna(False).astype(bool)
        live_past = live_df[~is_fc].copy()
    else:
        live_past = live_df.copy()

    if live_past.empty:
        return {
            "has_overlap": False,
            "has_comparison": False,
            "comparison_days": 0,
            "precip_mae": 0.0,
            "discharge_mae": 0.0,
            "precip_consistent": True,
            "discharge_consistent": True,
            "is_consistent": True,
        }

    # Try direct endpoint query for historical archive if network is online and station_meta is present
    arch_df = pd.DataFrame()
    if is_network_available() and station_meta and "latitude" in station_meta and "longitude" in station_meta:
        try:
            start_date_str = str(live_past["date"].iloc[-min(comparison_days, len(live_past))])
            end_date_str = str(live_past["date"].iloc[-1])
            data_cfg_path = _PROJECT_ROOT / "configs" / "data.yaml"
            if data_cfg_path.exists():
                with open(data_cfg_path, "r", encoding="utf-8") as f:
                    full_data_cfg = yaml.safe_load(f)
                fast_cfg = copy.deepcopy(full_data_cfg)
                fast_cfg.setdefault("rainfall_api", {})["max_retries"] = 0
                fast_cfg["rainfall_api"]["timeout_s"] = 2
                fast_cfg.setdefault("discharge_api", {})["max_retries"] = 0
                fast_cfg["discharge_api"]["timeout_s"] = 2
                from src.data.ingestion import fetch_rainfall, fetch_discharge
                hist_rain = fetch_rainfall(station_meta, fast_cfg, start_date=start_date_str, end_date=end_date_str)
                hist_dis = fetch_discharge(station_meta, fast_cfg, start_date=start_date_str, end_date=end_date_str)
                arch_df = pd.merge(hist_rain, hist_dis, on="date", how="outer").sort_values("date").reset_index(drop=True)
        except Exception as exc:
            logger.debug("Direct archive endpoint query not available (%s); using local history_df", exc)

    if arch_df.empty and history_df is not None and not history_df.empty and "date" in history_df.columns:
        arch_df = history_df.copy()

    if arch_df.empty:
        return {
            "has_overlap": False,
            "has_comparison": False,
            "comparison_days": 0,
            "precip_mae": 0.0,
            "discharge_mae": 0.0,
            "precip_consistent": True,
            "discharge_consistent": True,
            "is_consistent": True,
        }

    # Restrict to last comparison_days of live observations
    target_dates = set(live_past["date"].iloc[-comparison_days:])
    live_cols = [c for c in ["date", "precipitation_sum", "river_discharge"] if c in live_past.columns]
    arch_cols = [c for c in ["date", "precipitation_sum", "river_discharge"] if c in arch_df.columns]
    live_sub = live_past[live_past["date"].isin(target_dates)][live_cols].dropna(subset=["date"])
    arch_sub = arch_df[arch_df["date"].isin(target_dates)][arch_cols].dropna(subset=["date"])

    overlap = pd.merge(live_sub, arch_sub, on="date", suffixes=("_live", "_arch")).sort_values("date").reset_index(drop=True)

    if overlap.empty:
        return {
            "has_overlap": False,
            "has_comparison": False,
            "comparison_days": 0,
            "precip_mae": 0.0,
            "discharge_mae": 0.0,
            "precip_consistent": True,
            "discharge_consistent": True,
            "is_consistent": True,
        }

    # Rainfall metrics
    if "precipitation_sum_live" in overlap.columns and "precipitation_sum_arch" in overlap.columns:
        p_l = pd.to_numeric(overlap["precipitation_sum_live"], errors="coerce").fillna(0.0)
        p_a = pd.to_numeric(overlap["precipitation_sum_arch"], errors="coerce").fillna(0.0)
        m_pl = float(p_l.mean())
        m_pa = float(p_a.mean())
        precip_rel_bias = float(abs(m_pl - m_pa) / max(m_pa, 1e-4)) if (m_pa > 1e-4 or m_pl > 1e-4) else 0.0
        p_l_14 = p_l.rolling(14, min_periods=1).sum()
        p_a_14 = p_a.rolling(14, min_periods=1).sum()
        precip_14d_sum_diff = float((p_l_14 - p_a_14).abs().max())
        precip_mae = float((p_l - p_a).abs().mean())
        precip_consistent = (precip_rel_bias <= max_rel_bias_rain) and (precip_14d_sum_diff <= max_14d_diff_rain)
    else:
        m_pl, m_pa, precip_rel_bias, precip_14d_sum_diff, precip_mae = 0.0, 0.0, 0.0, 0.0, 0.0
        precip_consistent = True

    # Discharge metrics
    if "river_discharge_live" in overlap.columns and "river_discharge_arch" in overlap.columns:
        d_l = pd.to_numeric(overlap["river_discharge_live"], errors="coerce").fillna(0.0)
        d_a = pd.to_numeric(overlap["river_discharge_arch"], errors="coerce").fillna(0.0)
        m_dl = float(d_l.mean())
        m_da = float(d_a.mean())
        discharge_rel_bias = float(abs(m_dl - m_da) / max(m_da, 1e-4)) if (m_da > 1e-4 or m_dl > 1e-4) else 0.0
        d_l_14 = d_l.rolling(14, min_periods=1).sum()
        d_a_14 = d_a.rolling(14, min_periods=1).sum()
        discharge_14d_sum_diff = float((d_l_14 - d_a_14).abs().max())
        discharge_mae = float((d_l - d_a).abs().mean())
        discharge_consistent = (discharge_rel_bias <= max_rel_bias_dis) and (discharge_14d_sum_diff <= max_14d_diff_dis)
    else:
        m_dl, m_da, discharge_rel_bias, discharge_14d_sum_diff, discharge_mae = 0.0, 0.0, 0.0, 0.0, 0.0
        discharge_consistent = True

    is_consistent = precip_consistent and discharge_consistent

    result = {
        "station_id": station_id,
        "has_overlap": True,
        "has_comparison": True,
        "comparison_days": len(overlap),
        "start_date": str(overlap["date"].iloc[0]),
        "end_date": str(overlap["date"].iloc[-1]),
        "precip_mean_live": round(m_pl, 3),
        "precip_mean_archive": round(m_pa, 3),
        "precip_rel_bias": round(precip_rel_bias, 4),
        "precip_14d_sum_diff": round(precip_14d_sum_diff, 3),
        "precip_consistent": precip_consistent,
        "discharge_mean_live": round(m_dl, 3),
        "discharge_mean_archive": round(m_da, 3),
        "discharge_rel_bias": round(discharge_rel_bias, 4),
        "discharge_14d_sum_diff": round(discharge_14d_sum_diff, 3),
        "discharge_consistent": discharge_consistent,
        "is_consistent": is_consistent,
        "precip_mae": round(precip_mae, 4),
        "discharge_mae": round(discharge_mae, 4),
        "overlap_days": len(overlap),
    }

    if station_id != "UNKNOWN":
        try:
            row_df = pd.DataFrame([{
                "station_id": station_id,
                "comparison_days": len(overlap),
                "start_date": str(overlap["date"].iloc[0]),
                "end_date": str(overlap["date"].iloc[-1]),
                "precip_mean_live": round(m_pl, 3),
                "precip_mean_archive": round(m_pa, 3),
                "precip_rel_bias": round(precip_rel_bias, 4),
                "precip_14d_sum_diff": round(precip_14d_sum_diff, 3),
                "precip_consistent": precip_consistent,
                "discharge_mean_live": round(m_dl, 3),
                "discharge_mean_archive": round(m_da, 3),
                "discharge_rel_bias": round(discharge_rel_bias, 4),
                "discharge_14d_sum_diff": round(discharge_14d_sum_diff, 3),
                "discharge_consistent": discharge_consistent,
                "is_consistent": is_consistent,
            }])
            if report_csv.exists():
                existing_df = pd.read_csv(report_csv)
                existing_df = existing_df[existing_df["station_id"] != station_id]
                updated_df = pd.concat([existing_df, row_df], ignore_index=True)
            else:
                updated_df = row_df
            updated_df.to_csv(report_csv, index=False)
        except Exception as save_exc:
            logger.warning("Could not update %s: %s", report_csv, save_exc)

    return result


def generate_all_stations_source_consistency_report() -> pd.DataFrame:
    """Audit source consistency for all configured stations and save to reports/source_consistency.csv."""
    from src.utils.config import load_all_stations
    stations = load_all_stations()
    cfg = load_live_config()
    rep_dir = _PROJECT_ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    report_csv = rep_dir / "source_consistency.csv"

    rows = []
    for s in stations:
        sid = s["id"]
        cache_dir = _PROJECT_ROOT / cfg.get("cache", {}).get("cache_dir", "data/live_cache")
        latest_c = get_latest_cached_live_file(sid, cache_dir)
        live_df = pd.read_csv(latest_c) if latest_c else pd.DataFrame()

        proc_files = sorted((_PROJECT_ROOT / "data" / "processed").glob(f"{sid}_aligned_*.csv"))
        hist_df = pd.read_csv(proc_files[-1]) if proc_files else pd.DataFrame()

        res = check_live_source_consistency(
            live_df=live_df,
            history_df=hist_df,
            station_meta=s,
            live_cfg=cfg,
            reports_dir=rep_dir,
        )
        if res.get("has_comparison"):
            rows.append({
                "station_id": sid,
                "comparison_days": res["comparison_days"],
                "start_date": res["start_date"],
                "end_date": res["end_date"],
                "precip_mean_live": res["precip_mean_live"],
                "precip_mean_archive": res["precip_mean_archive"],
                "precip_rel_bias": res["precip_rel_bias"],
                "precip_14d_sum_diff": res["precip_14d_sum_diff"],
                "precip_consistent": res["precip_consistent"],
                "discharge_mean_live": res["discharge_mean_live"],
                "discharge_mean_archive": res["discharge_mean_archive"],
                "discharge_rel_bias": res["discharge_rel_bias"],
                "discharge_14d_sum_diff": res["discharge_14d_sum_diff"],
                "discharge_consistent": res["discharge_consistent"],
                "is_consistent": res["is_consistent"],
            })

    df_rep = pd.DataFrame(rows)
    if not df_rep.empty:
        df_rep.to_csv(report_csv, index=False)
    return df_rep


def prune_station_cache(station_id: str, cache_dir: Path, max_keep: int = 5) -> None:
    """Retain only the newest N cache CSV files for a given station."""
    if not cache_dir.exists():
        return
    matches = sorted(cache_dir.glob(f"{station_id}_*.csv"), key=lambda p: p.stat().st_mtime)
    if len(matches) > max_keep:
        for old_f in matches[:-max_keep]:
            try:
                old_f.unlink()
            except Exception as e:
                logger.debug("Could not remove old cache file %s: %s", old_f, e)


def get_latest_cached_live_file(station_id: str, cache_dir: Path) -> Optional[Path]:
    """Retrieve the newest cached live CSV file for a station."""
    if not cache_dir.exists():
        return None
    matches = sorted(cache_dir.glob(f"{station_id}_*.csv"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def fetch_station_live_data(
    station_id: str,
    station_meta: Dict[str, Any],
    live_cfg: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Retrieve near-real-time observations with offline/sandbox fallback.

    Returns
    -------
    dict:
      - df: pd.DataFrame with merged data, strictly tagged with 'is_forecast'
      - status: 'OK' | 'FALLBACK_CACHE' | 'UNAVAILABLE'
      - source_used: 'live_api' | 'live_cache' | 'historical_processed'
      - fetched_at_utc: str
      - last_observation_date: str (guaranteed <= today Asia/Kolkata)
      - warnings: list[str]
    """
    cfg = live_cfg or load_live_config()
    cache_dir = _PROJECT_ROOT / cfg.get("cache", {}).get("cache_dir", "data/live_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today_kolkata_str = datetime.now(KOLKATA_TZ).strftime("%Y-%m-%d")

    # Prune old cache files to keep newest N
    max_keep = int(cfg.get("cache", {}).get("max_cached_files_per_station", 5))
    prune_station_cache(station_id, cache_dir, max_keep=max_keep)

    # 1. Check TTL cache if force_refresh is False
    latest_cache = get_latest_cached_live_file(station_id, cache_dir)
    ttl_seconds = float(cfg.get("cache", {}).get("ttl_seconds", 1800))
    if latest_cache and not force_refresh:
        cache_age_s = time.time() - latest_cache.stat().st_mtime
        if cache_age_s < ttl_seconds:
            cache_mtime_utc = datetime.fromtimestamp(
                latest_cache.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            df_cached = pd.read_csv(latest_cache)
            df_cached["is_forecast"] = (df_cached["date"] >= today_kolkata_str).astype("boolean").astype(bool)
            for fc in FLAG_COLUMNS:
                if fc not in df_cached.columns:
                    df_cached[fc] = 0.0
                else:
                    df_cached[fc] = pd.to_numeric(df_cached[fc], errors="coerce").fillna(0.0)

            is_fc = df_cached["is_forecast"].astype("boolean").fillna(False).astype(bool)
            past_cached = df_cached[(~is_fc) & (df_cached["river_discharge"].notna())]
            last_obs = str(past_cached["date"].iloc[-1]) if not past_cached.empty else None
            return {
                "df": df_cached,
                "status": "OK",
                "source_used": "live_cache",
                "fetched_at_utc": cache_mtime_utc,
                "last_observation_date": last_obs,
                "cache_file": str(latest_cache),
                "warnings": warnings,
            }

    # Load base historical archive
    proc_files = sorted(
        (_PROJECT_ROOT / "data" / "processed").glob(f"{station_id}_aligned_*.csv")
    )
    df_history = pd.read_csv(proc_files[-1]) if proc_files else pd.DataFrame()

    # Attempt live network query
    try:
        rain_df = fetch_live_rainfall(station_meta, cfg["live_weather_api"])
        dis_df = fetch_live_discharge(station_meta, cfg["live_discharge_api"])

        # Outer join live rainfall and discharge on date
        live_merged = pd.merge(rain_df, dis_df, on=["date", "is_forecast"], how="outer").sort_values("date").reset_index(drop=True)

        bias_report: Dict[str, Any] = {}
        if not df_history.empty:
            bias_report = check_live_source_consistency(
                live_merged,
                df_history,
                station_meta=station_meta,
                live_cfg=cfg,
            )
            if not bias_report.get("is_consistent", True):
                reasons = []
                if not bias_report.get("precip_consistent", True):
                    reasons.append(
                        f"rainfall (rel_bias={bias_report.get('precip_rel_bias', 0.0):.1%}, 14d_diff={bias_report.get('precip_14d_sum_diff', 0.0)}mm)"
                    )
                if not bias_report.get("discharge_consistent", True):
                    reasons.append(
                        f"discharge (rel_bias={bias_report.get('discharge_rel_bias', 0.0):.1%}, 14d_diff={bias_report.get('discharge_14d_sum_diff', 0.0)}m3/s)"
                    )
                warnings.append(
                    f"Direct source consistency check exceeded tolerance: {', '.join(reasons)}"
                )
            final_df = merge_live_with_history(live_merged, df_history)
        else:
            final_df = live_merged
            final_df["source"] = "live_api"

        final_df["is_forecast"] = (final_df["date"] >= today_kolkata_str).astype("boolean").astype(bool)

        # Fill missing cleaning flags with 0.0 to prevent NaNs
        for fc in FLAG_COLUMNS:
            if fc not in final_df.columns:
                final_df[fc] = 0.0
            else:
                final_df[fc] = pd.to_numeric(final_df[fc], errors="coerce").fillna(0.0)

        # Save to live_cache on successful query
        ts_filename = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = cache_dir / f"{station_id}_{ts_filename}.csv"
        final_df.to_csv(out_path, index=False)
        prune_station_cache(station_id, cache_dir, max_keep=max_keep)

        is_fc = final_df["is_forecast"].astype("boolean").fillna(False).astype(bool)
        past_obs = final_df[(~is_fc) & (final_df["river_discharge"].notna())]
        last_obs = str(past_obs["date"].iloc[-1]) if not past_obs.empty else None

        return {
            "df": final_df,
            "status": "OK",
            "source_used": "live_api",
            "fetched_at_utc": now_utc_str,
            "last_observation_date": last_obs,
            "cache_file": str(out_path),
            "warnings": warnings,
        }

    except (requests.RequestException, IngestionError) as exc:
        # Fall back gracefully to cached live file or historical processed data
        warnings.append(f"Live API unavailable ({exc}); falling back to local cached data")

        latest_cache = get_latest_cached_live_file(station_id, cache_dir)
        if latest_cache:
            cache_mtime_utc = datetime.fromtimestamp(
                latest_cache.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            df_cached = pd.read_csv(latest_cache)
            df_cached["is_forecast"] = (df_cached["date"] >= today_kolkata_str).astype("boolean").astype(bool)
            for fc in FLAG_COLUMNS:
                if fc not in df_cached.columns:
                    df_cached[fc] = 0.0
                else:
                    df_cached[fc] = pd.to_numeric(df_cached[fc], errors="coerce").fillna(0.0)

            is_fc = df_cached["is_forecast"].astype("boolean").fillna(False).astype(bool)
            past_cached = df_cached[(~is_fc) & (df_cached["river_discharge"].notna())]
            last_obs = str(past_cached["date"].iloc[-1]) if not past_cached.empty else None
            return {
                "df": df_cached,
                "status": "FALLBACK_CACHE",
                "source_used": "live_cache",
                "fetched_at_utc": cache_mtime_utc,
                "last_observation_date": last_obs,
                "cache_file": str(latest_cache),
                "warnings": warnings,
            }

        if not df_history.empty:
            proc_mtime_utc = (
                datetime.fromtimestamp(proc_files[-1].stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if proc_files
                else now_utc_str
            )
            df_history["is_forecast"] = (df_history["date"] >= today_kolkata_str).astype("boolean").astype(bool)
            for fc in FLAG_COLUMNS:
                if fc not in df_history.columns:
                    df_history[fc] = 0.0
                else:
                    df_history[fc] = pd.to_numeric(df_history[fc], errors="coerce").fillna(0.0)

            is_fc_hist = df_history["is_forecast"].astype("boolean").fillna(False).astype(bool)
            past_hist = df_history[(~is_fc_hist) & (df_history["river_discharge"].notna())]
            last_obs = str(past_hist["date"].iloc[-1]) if not past_hist.empty else None
            return {
                "df": df_history,
                "status": "FALLBACK_CACHE",
                "source_used": "historical_processed",
                "fetched_at_utc": proc_mtime_utc,
                "last_observation_date": last_obs,
                "cache_file": str(proc_files[-1]) if proc_files else None,
                "warnings": warnings,
            }

        return {
            "df": pd.DataFrame(),
            "status": "UNAVAILABLE",
            "source_used": "none",
            "fetched_at_utc": now_utc_str,
            "last_observation_date": None,
            "cache_file": None,
            "warnings": warnings + ["No live data or cached historical data available"],
        }
    except Exception as exc:
        logger.exception("Unexpected error during live fetch for station %s: %s", station_id, exc)
        raise


def fetch_all_stations_live(
    stations: Optional[List[Dict[str, Any]]] = None,
    delay_s: float = 1.0,
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch live data for all stations with short delay, retries, and local caching.

    Returns mapping from station_id -> fetch report dict.
    """
    from src.utils.config import load_all_stations
    stns = stations or load_all_stations()
    reports = {}
    for idx, s in enumerate(stns):
        sid = s["id"]
        logger.info("Fetching live data for station %s (%d/%d)...", sid, idx + 1, len(stns))
        try:
            res = fetch_station_live_data(sid, s, force_refresh=force_refresh)
            status_val = "OK" if res.get("source_used") == "live_api" else (
                "FALLBACK_CACHE" if res.get("source_used") == "live_cache" else "FAILED"
            )
            reason_val = (
                "Fetched fresh daily data from Open-Meteo API"
                if status_val == "OK"
                else (
                    f"Used local cache: {res.get('last_observation_date')}"
                    if status_val == "FALLBACK_CACHE"
                    else (res.get("warnings", ["Failed"])[0] if res.get("warnings") else "Fetch failed")
                )
            )
            reports[sid] = {
                "station_id": sid,
                "station_name": s["name"],
                "fetch_status": status_val,
                "source_used": res.get("source_used"),
                "last_observation_date": res.get("last_observation_date"),
                "reason": reason_val,
                "df": res.get("df"),
                "warnings": res.get("warnings", []),
            }
        except Exception as exc:
            reports[sid] = {
                "station_id": sid,
                "station_name": s["name"],
                "fetch_status": "FAILED",
                "source_used": "none",
                "last_observation_date": None,
                "reason": f"Exception during fetch: {exc}",
                "df": pd.DataFrame(),
                "warnings": [str(exc)],
            }
        if delay_s > 0 and idx < len(stns) - 1:
            time.sleep(delay_s)

    return reports
