"""
src/train.py
────────────
Full reproducible training, evaluation, and serialization pipeline.

Honesty Rules Enforced:
  - All scores are strictly written via `python -m src.train` to reports/model_scores.json and .csv.
  - Scores are NEVER hardcoded, estimated, or copied from publications.
  - Identical test timestamps evaluated across every model per horizon.
  - Baseline comparison flag ('beats_baseline') verified objectively against Persistence.
  - Event threshold: 95th percentile of the training set (statistical rule, NOT official CWC level).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.ingestion import ingest_station
from src.data.validation import validate_dataframe
from src.data.cleaning import clean_rainfall, clean_discharge
from src.data.alignment import align_datasets, save_processed
from src.features.temporal_features import (
    build_feature_matrix,
    get_feature_columns,
    get_target_column,
    HORIZONS,
)
from src.data.split import chronological_split
from src.models.persistence import PersistenceBaseline
from src.models.linear import build_ridge_model
import hashlib
import time
from sklearn.model_selection import TimeSeriesSplit

from src.models.tree import build_decision_tree_model
from src.models.random_forest import build_random_forest_model
from src.models.gradient_boosting import build_gradient_boosting_model
from src.models.xgboost_model import build_xgboost_model
from src.models.advanced import (
    build_elastic_net_model,
    build_extra_trees_model,
    build_hist_gradient_boosting_model,
)
from src.models.baselines import SeasonalClimatology, SeasonalNaive
from src.evaluation.uncertainty import (
    ConformalIntervalCalibrator,
    evaluate_prediction_interval,
)
from src.evaluation.metrics import compute_standard_metrics
from src.evaluation.event_metrics import compute_event_metrics
from src.risk.thresholds import compute_station_thresholds, update_thresholds_config
from src.utils.config import load_data_config, load_all_stations, get_station

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_model_registry() -> dict[str, Any]:
    """Registry of models evaluated on the common benchmark."""
    return {
        "persistence": PersistenceBaseline(discharge_col="river_discharge"),
        "ridge": build_ridge_model(random_state=42),
        "decision_tree": build_decision_tree_model(random_state=42),
        "random_forest": build_random_forest_model(n_estimators=60, random_state=42),
        "gradient_boosting": build_gradient_boosting_model(n_estimators=60, random_state=42),
        "xgboost": build_xgboost_model(n_estimators=60, random_state=42),
        "hist_gradient_boosting": build_hist_gradient_boosting_model(max_iter=60, random_state=42),
        "elastic_net": build_elastic_net_model(random_state=42),
        "extra_trees": build_extra_trees_model(n_estimators=60, random_state=42),
    }


def _print_single_station_summary(
    station_id: str,
    rows: int | str,
    status: str,
    best_model: str,
    best_rmse: str,
    reason: str,
) -> None:
    """Print standard 6-column tabular summary for a single station run."""
    print("\n" + "=" * 115)
    print("CLIMATESHIELD AI — SINGLE-STATION TRAINING & SANITY RESULT")
    print("=" * 115)
    header = f"{'station':<20} | {'rows':<6} | {'status':<18} | {'best model':<14} | {'best RMSE (1-day)':<18} | {'reason if failed'}"
    print(header)
    print("-" * 115)
    print(f"{station_id:<20} | {str(rows):<6} | {status:<18} | {best_model:<14} | {best_rmse:<18} | {reason}")
    print("=" * 115 + "\n")


def check_station_data_sanity(
    df_disc: pd.DataFrame,
    df_rain: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
    station_id: str = "UNKNOWN",
) -> tuple[bool, str, dict[str, Any]]:
    """Flag stations where discharge is near-constant, mostly zero, has very few rows, or has gaps over a set limit.

    Parameters
    ----------
    df_disc: Discharge DataFrame with 'river_discharge' and 'date'.
    df_rain: Optional Rainfall DataFrame.
    cfg: Config dictionary.
    station_id: Identifier string for reporting.

    Returns
    -------
    tuple[bool, str, dict[str, Any]]
        (is_sane, reason, details)
    """
    if len(df_disc) == 0:
        return False, "Discharge dataset is empty (0 rows)", {
            "total_rows": 0, "valid_count": 0, "failure_type": "empty_dataframe"
        }

    col = "river_discharge"
    if col not in df_disc.columns:
        return False, "Column 'river_discharge' missing from discharge dataset", {"error": "missing_column"}

    series = df_disc[col]
    valid_series = series.dropna()
    total_rows = len(df_disc)
    valid_count = len(valid_series)

    # 1. Very few rows or all NaN
    if valid_count == 0:
        return False, "All discharge values are NaN (GloFAS resolution limit; unmodelled basin)", {
            "total_rows": total_rows, "valid_count": 0, "failure_type": "all_nan"
        }
    min_rows = int(cfg.get("validation", {}).get("min_rows", 60)) if cfg else 60
    if valid_count < min_rows:
        return False, f"Insufficient rows: only {valid_count} valid rows found (minimum required: {min_rows})", {
            "total_rows": total_rows, "valid_count": valid_count, "failure_type": "too_few_rows"
        }

    # 2. Mostly zero
    zero_count = int((valid_series == 0.0).sum())
    zero_pct = 100.0 * zero_count / valid_count
    if zero_pct >= 90.0:
        return False, f"Discharge is mostly zero ({zero_pct:.1f}% zeros >= 90.0% threshold)", {
            "total_rows": total_rows, "valid_count": valid_count, "zero_pct": zero_pct, "failure_type": "mostly_zero"
        }

    # 3. Near-constant / flat signal
    std_val = float(valid_series.std())
    rng_val = float(valid_series.max() - valid_series.min())
    if std_val < 0.005 or rng_val < 0.05:
        return False, f"Discharge is near-constant (std={std_val:.4f} m³/s, range={rng_val:.4f} m³/s)", {
            "total_rows": total_rows, "valid_count": valid_count, "std": std_val, "range": rng_val, "failure_type": "near_constant"
        }

    # 4. Gaps over a set limit (consecutive missing dates > 14 days)
    is_na = series.isna()
    consec_nans = is_na.groupby((~is_na).cumsum()).cumsum()
    max_gap = int(consec_nans.max()) if len(consec_nans) else 0
    if max_gap > 14:
        return False, f"Discharge has gap of {max_gap} consecutive days exceeding limit of 14 days", {
            "total_rows": total_rows, "valid_count": valid_count, "max_gap_days": max_gap, "failure_type": "gap_exceeded"
        }

    if "date" in df_disc.columns:
        try:
            dates = pd.to_datetime(df_disc["date"]).sort_values()
            diffs = (dates.diff().dt.days).dropna()
            max_date_gap = int(diffs.max()) if len(diffs) else 1
            if max_date_gap > 14:
                return False, f"Discharge has missing date gap of {max_date_gap} days exceeding limit of 14 days", {
                    "total_rows": total_rows, "valid_count": valid_count, "max_date_gap": max_date_gap, "failure_type": "date_gap_exceeded"
                }
        except Exception:
            pass

    return True, "Data sanity checks passed", {
        "total_rows": total_rows,
        "valid_count": valid_count,
        "zero_pct": round(zero_pct, 2),
        "std": round(std_val, 4),
        "range": round(rng_val, 4),
        "max_gap_days": max_gap,
    }


def train_and_evaluate(
    station_id: str | None = None,
    horizons: list[int] | None = None,
    cfg: dict[str, Any] | None = None,
    df_aligned: pd.DataFrame | None = None,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Train all models across horizons and compute all continuous and event metrics.

    Parameters
    ----------
    station_id: Target Tamil Nadu river station ID.
    horizons: Forecast horizons in days (default: [1, 3, 7]).
    cfg: Loaded data config.
    df_aligned: Optional pre-aligned dataframe for testing / offline execution.

    Returns
    -------
    dict
        Evaluation results summary dictionary.
    """
    if cfg is None:
        cfg = load_data_config()

    if station_id is None:
        station_id = cfg["default_station_id"]

    if horizons is None:
        horizons = HORIZONS

    station = get_station(cfg, station_id)
    models_dir = _PROJECT_ROOT / cfg["paths"]["models_dir"]
    reports_dir = _PROJECT_ROOT / cfg["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Ingest / Prepare data if not supplied ──────────────────────────────
    if df_aligned is None:
        raw_dir = _PROJECT_ROOT / "data" / "raw"
        proc_dir = _PROJECT_ROOT / cfg["paths"]["processed_dir"]
        proc_files = sorted(proc_dir.glob(f"{station_id}_aligned_*.csv"))
        rain_files = sorted(raw_dir.glob(f"{station_id}_rainfall_raw_*.csv"))
        disc_files = sorted(raw_dir.glob(f"{station_id}_discharge_raw_*.csv"))

        if proc_files:
            logger.info("Using cached aligned file for station %s (%s)...", station_id, proc_files[-1].name)
            df_aligned = pd.read_csv(proc_files[-1])
        elif rain_files and disc_files:
            logger.info("Using cached raw files for station %s...", station_id)
            df_rain_raw = pd.read_csv(rain_files[-1])
            df_disc_raw = pd.read_csv(disc_files[-1])
            validate_dataframe(df_rain_raw, "rainfall", station_id, cfg)
            validate_dataframe(df_disc_raw, "discharge", station_id, cfg)
            df_rain_clean = clean_rainfall(df_rain_raw, cfg, station_id=station_id)
            df_disc_clean = clean_discharge(df_disc_raw, cfg, station_id=station_id)
            df_aligned, align_report = align_datasets(df_rain_clean, df_disc_clean)
            save_processed(df_aligned, station_id, cfg, alignment_report=align_report)
        else:
            logger.info("Ingesting fresh data for station %s...", station_id)
            raw_paths = ingest_station(station_id, cfg)
            df_rain_raw = pd.read_csv(raw_paths["rainfall"])
            df_disc_raw = pd.read_csv(raw_paths["discharge"])

            df_rain_clean = clean_rainfall(df_rain_raw, cfg, station_id=station_id)
            df_disc_clean = clean_discharge(df_disc_raw, cfg, station_id=station_id)

            df_aligned, align_report = align_datasets(df_rain_clean, df_disc_clean)
            save_processed(df_aligned, station_id, cfg, alignment_report=align_report)

    # ── 1.5. Data Sanity Check per station before training ────────────────────
    is_sane, sanity_reason, sanity_details = check_station_data_sanity(
        df_disc=df_aligned,
        cfg=cfg,
        station_id=station_id,
    )
    sanity_report_path = reports_dir / f"data_sanity_{station_id}.json"
    with open(sanity_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "station_id": station_id,
            "is_sane": is_sane,
            "reason": sanity_reason,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "details": sanity_details,
        }, f, indent=2)

    if not is_sane:
        logger.warning("Station %s data sanity check failed: %s", station_id, sanity_reason)
        stn_rows = len(df_aligned) if df_aligned is not None else 0
        if print_summary:
            _print_single_station_summary(
                station_id=station_id,
                rows=stn_rows,
                status="insufficient data",
                best_model="N/A",
                best_rmse="N/A",
                reason=sanity_reason,
            )
        return {
            "station_id": station_id,
            "total_rows": stn_rows,
            "status": "insufficient data",
            "reason": sanity_reason,
            "details": sanity_details,
            "scores": [],
            "models": {},
        }

    # ── 2. Feature & Target Matrix ───────────────────────────────────────────
    logger.info("Building temporal features matrix...")
    matrix = build_feature_matrix(df_aligned, horizons=horizons)
    feature_cols = get_feature_columns(matrix, horizons=horizons)

    # Chronological 70/15/15 split
    train_df, val_df, test_df = chronological_split(matrix, 0.70, 0.15, 0.15)
    logger.info(
        "Split sizes: Train=%d, Val=%d, Test=%d (Dates: %s -> %s)",
        len(train_df), len(val_df), len(test_df),
        train_df['date'].min(), test_df['date'].max(),
    )

    # Statistical 95th percentile event threshold derived strictly from training set
    p95_event_threshold = float(train_df["river_discharge"].quantile(0.95))
    logger.info("Statistical 95th pct event threshold: %.2f m³/s", p95_event_threshold)

    # Calibrate and update station thresholds in thresholds.yaml
    try:
        thresholds_data = compute_station_thresholds(train_df["river_discharge"].values)
        update_thresholds_config(station_id, thresholds_data)
        logger.info("Calibrated and updated thresholds for station %s", station_id)
    except Exception as e:
        logger.warning("Could not calibrate thresholds for %s: %s", station_id, e)

    results_table: list[dict[str, Any]] = []
    best_params_map: dict[str, Any] = {}
    target_transforms_map: dict[str, str] = {}
    ablation_records: list[dict[str, Any]] = []

    # Calculate training data hash for provenance and reproducibility (Part C.3)
    train_bytes = train_df[feature_cols].values.tobytes() if feature_cols else b""
    data_hash = hashlib.sha256(train_bytes).hexdigest()[:16]

    training_log_path = reports_dir / "training_log.jsonl"

    # ── 3. Train and evaluate across horizons ─────────────────────────────────
    for h in horizons:
        target_col = get_target_column(h)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        test_dates = test_df["date"].values

        # ── 3.1 Baselines (Persistence, Seasonal Climatology, Seasonal Naive) ──
        # 1. Persistence Baseline
        base_persist = PersistenceBaseline(discharge_col="river_discharge")
        base_persist.fit(train_df[["river_discharge"]], y_train)
        pred_persist = base_persist.predict(test_df[["river_discharge"]])
        persist_metrics = compute_standard_metrics(y_test, pred_persist)
        persistence_rmse = float(persist_metrics["rmse"])

        # 2. Seasonal Climatology Baseline
        base_clim = SeasonalClimatology(date_col="date", discharge_col="river_discharge")
        base_clim.fit(train_df, y_train)
        pred_clim = base_clim.predict(test_df)
        clim_metrics = compute_standard_metrics(y_test, pred_clim)
        climatology_rmse = float(clim_metrics["rmse"])

        # 3. Seasonal Naive Baseline (same calendar day last year)
        base_naive = SeasonalNaive(date_col="date", discharge_col="river_discharge")
        base_naive.fit(train_df, y_train)
        pred_naive = base_naive.predict(test_df)
        naive_metrics = compute_standard_metrics(y_test, pred_naive)
        seasonal_naive_rmse = float(naive_metrics["rmse"])

        logger.info(
            "Station %s h=%dd Baselines RMSE: Persistence=%.4f, Climatology=%.4f, SeasonalNaive=%.4f",
            station_id, h, persistence_rmse, climatology_rmse, seasonal_naive_rmse,
        )

        registry = get_model_registry()

        for model_name, model in registry.items():
            t0 = time.perf_counter()
            logger.info("Evaluating %s for horizon h=%dd on station %s...", model_name, h, station_id)

            if model_name == "persistence":
                X_tr = train_df[["river_discharge"]]
                X_v = val_df[["river_discharge"]]
                X_te = test_df[["river_discharge"]]
                use_log1p = False
            else:
                X_tr = train_df[feature_cols]
                X_v = val_df[feature_cols]
                X_te = test_df[feature_cols]

                # ── Part B.2: Target Transformation (log1p vs raw on validation) ──
                # Test raw target
                model.fit(X_tr, y_train)
                val_raw_preds = model.predict(X_v)
                rmse_val_raw = float(np.sqrt(np.mean((y_val - val_raw_preds) ** 2)))

                # Test log1p target
                y_tr_log = np.log1p(np.maximum(y_train, 0.0))
                model.fit(X_tr, y_tr_log)
                val_log_preds = np.expm1(model.predict(X_v))
                rmse_val_log = float(np.sqrt(np.mean((y_val - val_log_preds) ** 2)))

                if rmse_val_log < rmse_val_raw:
                    use_log1p = True
                    target_transform_name = "log1p"
                else:
                    use_log1p = False
                    target_transform_name = "raw"

                target_transforms_map[f"{station_id}_h{h}d_{model_name}"] = target_transform_name

            # ── Fit final model on train set ──
            if use_log1p:
                model.fit(X_tr, np.log1p(np.maximum(y_train, 0.0)))
                val_preds = np.clip(np.expm1(model.predict(X_v)), 0.0, None)
                y_pred = np.clip(np.expm1(model.predict(X_te)), 0.0, None)
            else:
                model.fit(X_tr, y_train)
                val_preds = np.clip(model.predict(X_v), 0.0, None)
                y_pred = np.clip(model.predict(X_te), 0.0, None)

            train_time_s = time.perf_counter() - t0

            # ── Part B.7: Split-Conformal Prediction Interval & Measured Coverage ──
            calibrator = ConformalIntervalCalibrator(confidence_level=0.90)
            calibrator.calibrate(y_val, val_preds)
            lower_test, upper_test = calibrator.predict_interval(y_pred)
            unc_eval = evaluate_prediction_interval(y_test, lower_test, upper_test, nominal_confidence=0.90)
            empirical_coverage = float(unc_eval["empirical_coverage_pct"])

            # ── Standard & Event Metrics ──
            std_metrics = compute_standard_metrics(y_test, y_pred)
            ev_metrics = compute_event_metrics(
                y_true=y_test,
                y_pred=y_pred,
                threshold=p95_event_threshold,
                dates=test_dates,
                horizon_days=h,
            )

            mod_rmse = float(std_metrics["rmse"])
            beats_persist = bool(mod_rmse < persistence_rmse)
            beats_all = bool(
                (mod_rmse < persistence_rmse)
                and (mod_rmse < seasonal_naive_rmse)
                and (mod_rmse < climatology_rmse)
            )

            entry = {
                "station_id": station_id,
                "horizon_days": h,
                "model": model_name,
                "mae": round(std_metrics["mae"], 4),
                "rmse": round(mod_rmse, 4),
                "r2": round(std_metrics["r2"], 4),
                "nse": round(std_metrics["nse"], 4),
                "peak_flow_error_pct": round(ev_metrics["peak_flow_error_pct"], 2),
                "peak_timing_error_days": ev_metrics["peak_timing_error_days"],
                "missed_event_rate": round(ev_metrics["missed_event_rate"], 4),
                "false_alarm_rate": round(ev_metrics["false_alarm_rate"], 4),
                "event_threshold_m3s": round(p95_event_threshold, 2),
                "interval_coverage": round(empirical_coverage, 2),
                "train_time_s": round(train_time_s, 4),
                "beats_baseline": beats_persist,
                "beats_persistence": beats_persist,
                "beats_all_baselines": beats_all,
                "target_transform": "log1p" if use_log1p else "raw",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "test_start_date": str(test_df["date"].min()),
                "test_end_date": str(test_df["date"].max()),
            }

            # ── Save model artifacts ──
            artifact_filename = f"{station_id}_{model_name}_h{h}d.joblib"
            artifact_path = models_dir / artifact_filename
            meta_path = models_dir / f"{station_id}_{model_name}_h{h}d.meta.json"

            # Compute drift reference statistics on training discharge
            train_dis_clean = train_df["river_discharge"].dropna().to_numpy(dtype=float)
            drift_ref_stats = {
                "feature_name": "river_discharge",
                "mean": round(float(np.mean(train_dis_clean)), 4),
                "std": round(float(np.std(train_dis_clean)), 4),
                "min": round(float(np.min(train_dis_clean)), 4),
                "max": round(float(np.max(train_dis_clean)), 4),
                "p10": round(float(np.percentile(train_dis_clean, 10)), 4),
                "p25": round(float(np.percentile(train_dis_clean, 25)), 4),
                "p50": round(float(np.percentile(train_dis_clean, 50)), 4),
                "p75": round(float(np.percentile(train_dis_clean, 75)), 4),
                "p90": round(float(np.percentile(train_dis_clean, 90)), 4),
                "quantiles_10bin": [round(float(q), 4) for q in np.percentile(train_dis_clean, np.linspace(0, 100, 11))],
                "n_samples": int(len(train_dis_clean)),
            }

            joblib.dump(model, artifact_path)
            meta_dict = {
                "model_name": model_name,
                "station_id": station_id,
                "horizon_days": h,
                "features": feature_cols if model_name != "persistence" else ["river_discharge"],
                "target_transform": "log1p" if use_log1p else "raw",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": entry,
                "conformal_q_hat": round(calibrator.q_hat, 4),
                "conformal_test_coverage_pct": round(empirical_coverage, 2),
                "drift_reference_stats": drift_ref_stats,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)

            stn_models_dir = models_dir / station_id
            stn_models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, stn_models_dir / f"{model_name}_h{h}d.joblib")
            joblib.dump(model, stn_models_dir / artifact_filename)
            with open(stn_models_dir / f"{model_name}_h{h}d.meta.json", "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)

            # Save station-level drift reference sidecar
            with open(stn_models_dir / "drift_reference.json", "w", encoding="utf-8") as f:
                json.dump(drift_ref_stats, f, indent=2)

            # Save conformal calibrator metadata sidecar
            conf_sidecar = stn_models_dir / f"{model_name}_h{h}d_conformal.json"
            with open(conf_sidecar, "w", encoding="utf-8") as f:
                json.dump({
                    "station_id": station_id,
                    "horizon_days": h,
                    "model_name": model_name,
                    "confidence_level": 0.90,
                    "q_hat": round(calibrator.q_hat, 4),
                    "empirical_coverage_pct": round(empirical_coverage, 2),
                    "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)

            results_table.append(entry)

            # Append to training log JSONL (Part C.3)
            with open(training_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "station_id": station_id,
                    "horizon_days": h,
                    "model": model_name,
                    "train_time_s": round(train_time_s, 4),
                    "data_hash": data_hash,
                    "val_rmse": round(float(np.sqrt(np.mean((y_val - val_preds) ** 2))), 4),
                    "test_rmse": round(mod_rmse, 4),
                    "interval_coverage": round(empirical_coverage, 2),
                }) + "\n")

        # ── Part B.3 Feature-Group Ablation (evaluated at h=1 on primary tree model) ──
        if h == 1 and len(feature_cols) > 5:
            ablation_groups = {
                "all_features": [],
                "without_rainfall_intensity": [c for c in feature_cols if "precip_max_" in c or "api_decay_" in c or "days_since_" in c],
                "without_discharge_dynamics": [c for c in feature_cols if "river_discharge_roll_std_" in c or "river_discharge_roc_" in c],
                "without_calendar": [c for c in feature_cols if "month_" in c or "monsoon_" in c],
            }
            baseline_ridge = build_ridge_model(random_state=42)
            baseline_ridge.fit(train_df[feature_cols], y_train)
            base_val_rmse = float(np.sqrt(np.mean((y_val - baseline_ridge.predict(val_df[feature_cols])) ** 2)))

            for grp_name, drop_cols in ablation_groups.items():
                active_feats = [c for c in feature_cols if c not in drop_cols]
                abl_model = build_ridge_model(random_state=42)
                abl_model.fit(train_df[active_feats], y_train)
                abl_val_rmse = float(np.sqrt(np.mean((y_val - abl_model.predict(val_df[active_feats])) ** 2)))
                delta_rmse = abl_val_rmse - base_val_rmse
                ablation_records.append({
                    "station": station_id,
                    "model": "ridge",
                    "horizon": 1,
                    "group_removed": grp_name,
                    "n_features": len(active_feats),
                    "val_rmse": round(abl_val_rmse, 4),
                    "delta_rmse": round(delta_rmse, 4),
                    "status": "baseline" if grp_name == "all_features" else ("hurts" if delta_rmse > 0 else "helps"),
                })

    # Save feature ablation table to reports/feature_ablation.csv
    if ablation_records:
        abl_df = pd.DataFrame(ablation_records)
        abl_file = reports_dir / "feature_ablation.csv"
        abl_df.to_csv(abl_file, index=False)
        logger.info("Saved feature ablation study to %s", abl_file)

    # Save target transforms and best params
    with open(reports_dir / "target_transforms.json", "w", encoding="utf-8") as f:
        json.dump(target_transforms_map, f, indent=2)

    # Also save features matrix in data/processed/
    processed_dir = _PROJECT_ROOT / cfg["paths"]["processed_dir"]
    try:
        matrix.to_csv(processed_dir / f"{station_id}_features.csv", index=False)
        for h in horizons:
            matrix.to_csv(processed_dir / f"{station_id}_features_h{h}.csv", index=False)
    except Exception as e:
        logger.warning("Could not write processed features CSV for %s: %s", station_id, e)

    # ── 4. Write verified scores to reports/ (incremental station merge) ────
    csv_out = reports_dir / "model_scores.csv"
    json_out = reports_dir / "model_scores.json"

    # Load existing scores if present to maintain multi-station records
    existing_scores = []
    if json_out.exists():
        try:
            with open(json_out, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                existing_scores = [
                    s for s in prev_data.get("scores", [])
                    if s.get("station_id") != station_id
                ]
        except Exception as e:
            logger.warning("Could not parse previous model_scores.json: %s", e)

    merged_scores = existing_scores + results_table
    scores_df = pd.DataFrame(merged_scores)
    scores_df.to_csv(csv_out, index=False)

    full_report = {
        "status": "SUCCESS",
        "total_rows": len(matrix),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "station_id": station_id,
        "station_name": station.get("name"),
        "caveat": (
            "Scores apply strictly to each specific station, evaluated time period, and Open-Meteo modelled discharge. "
            "Event metrics use training 95th percentile as statistical threshold, NOT official CWC flood levels. "
            "No pooled cross-station accuracy claim is made."
        ),
        "event_threshold_rule": "95th percentile of historical training set river discharge",
        "scores": merged_scores,
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Write per-station scores file: reports/model_scores_<station_id>.json
    stn_json_out = reports_dir / f"model_scores_{station_id}.json"
    stn_report = {
        "status": "SUCCESS",
        "total_rows": len(matrix),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "station_id": station_id,
        "station_name": station.get("name"),
        "caveat": full_report["caveat"],
        "event_threshold_rule": full_report["event_threshold_rule"],
        "scores": results_table,
    }
    with open(stn_json_out, "w", encoding="utf-8") as f:
        json.dump(stn_report, f, indent=2)

    # Write combined reports/model_scores_all.csv with EXACT required columns:
    # station, horizon, model, MAE, RMSE, R2, NSE, peak_error, peak_timing_error,
    # missed_events, false_alarms, interval_coverage, train_time_s, beats_persistence, beats_all_baselines
    formatted_rows = []
    for entry in merged_scores:
        formatted_rows.append({
            "station": entry["station_id"],
            "horizon": entry["horizon_days"],
            "model": entry["model"],
            "MAE": round(entry["mae"], 4),
            "RMSE": round(entry["rmse"], 4),
            "R2": round(entry["r2"], 4),
            "NSE": round(entry["nse"], 4),
            "peak_error": round(entry["peak_flow_error_pct"], 2) if entry.get("peak_flow_error_pct") is not None else None,
            "peak_timing_error": entry.get("peak_timing_error_days", 0),
            "missed_events": round(entry["missed_event_rate"], 4) if entry.get("missed_event_rate") is not None else None,
            "false_alarms": round(entry["false_alarm_rate"], 4) if entry.get("false_alarm_rate") is not None else None,
            "interval_coverage": round(entry.get("interval_coverage", 90.0), 2),
            "train_time_s": round(entry.get("train_time_s", 0.0), 4),
            "beats_persistence": bool(entry.get("beats_persistence", entry.get("beats_baseline", False))),
            "beats_all_baselines": bool(entry.get("beats_all_baselines", False)),
        })
    csv_all_out = reports_dir / "model_scores_all.csv"
    pd.DataFrame(formatted_rows).to_csv(csv_all_out, index=False)

    logger.info("Saved verified scores to %s, %s, and %s (%d total entries)", json_out, csv_out, stn_json_out, len(merged_scores))

    # Print single-station summary if requested
    if print_summary:
        stn_h1_scores = [
            s for s in results_table
            if s.get("station_id") == station_id and s.get("horizon_days") == 1
        ]
        if stn_h1_scores:
            best_h1 = min(stn_h1_scores, key=lambda x: x["rmse"])
            best_mod_name = best_h1["model"]
            best_rmse_val = f"{best_h1['rmse']:.4f}"
        else:
            best_mod_name = "N/A"
            best_rmse_val = "N/A"

        _print_single_station_summary(
            station_id=station_id,
            rows=len(matrix),
            status="SUCCESS",
            best_model=best_mod_name,
            best_rmse=best_rmse_val,
            reason="-",
        )

    return full_report


def train_all_stations(
    horizons: list[int] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and evaluate models for every station in data.yaml.

    Runs ingestion, validation, sanity checking, cleaning, alignment, features,
    and training for horizons 1/3/7. Continues past a failed/insufficient station
    and prints a final summary table of:
    station | rows | status | best model | best RMSE (1-day) | reason if failed.
    """
    if cfg is None:
        cfg = load_data_config()

    stations = cfg.get("stations", [])
    logger.info("Starting train_all_stations for %d stations...", len(stations))

    reports_dir = _PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for stn in stations:
        sid = stn["id"]
        sname = stn.get("name", sid)
        logger.info("Processing station %s (%s)...", sid, sname)

        try:
            # 1. Ingestion / Check cached raw files
            raw_dir = _PROJECT_ROOT / "data" / "raw"
            rain_files = sorted(raw_dir.glob(f"{sid}_rainfall_raw_*.csv"))
            disc_files = sorted(raw_dir.glob(f"{sid}_discharge_raw_*.csv"))

            if not rain_files or not disc_files:
                logger.info("Fetching raw data from API for %s...", sid)
                raw_paths = ingest_station(sid, cfg)
                df_rain_raw = pd.read_csv(raw_paths["rainfall"])
                df_disc_raw = pd.read_csv(raw_paths["discharge"])
            else:
                logger.info("Using cached raw files for %s...", sid)
                df_rain_raw = pd.read_csv(rain_files[-1])
                df_disc_raw = pd.read_csv(disc_files[-1])

            # 2. Validation
            validate_dataframe(df_rain_raw, "rainfall", sid, cfg)
            validate_dataframe(df_disc_raw, "discharge", sid, cfg)

            # 3. Data Sanity Check per station before training
            is_sane, sanity_reason, sanity_details = check_station_data_sanity(
                df_disc=df_disc_raw,
                df_rain=df_rain_raw,
                cfg=cfg,
                station_id=sid,
            )

            # Save sanity check report sidecar
            sanity_report_path = reports_dir / f"data_sanity_{sid}.json"
            with open(sanity_report_path, "w", encoding="utf-8") as f:
                json.dump({
                    "station_id": sid,
                    "is_sane": is_sane,
                    "reason": sanity_reason,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "details": sanity_details,
                }, f, indent=2)

            if not is_sane:
                logger.warning("Station %s marked insufficient data: %s", sid, sanity_reason)
                summary_records.append({
                    "station": sid,
                    "rows": len(df_disc_raw),
                    "status": "insufficient data",
                    "best model": "N/A",
                    "best RMSE (1-day)": "N/A",
                    "reason if failed": sanity_reason,
                })
                continue

            # 4. Cleaning
            df_rain_clean = clean_rainfall(df_rain_raw, cfg, station_id=sid)
            df_disc_clean = clean_discharge(df_disc_raw, cfg, station_id=sid)

            # 5. Alignment
            df_aligned, align_report = align_datasets(df_rain_clean, df_disc_clean)
            if df_aligned.empty:
                raise ValueError("Aligned dataset is empty (no overlapping dates).")

            save_processed(df_aligned, sid, cfg, alignment_report=align_report)

            # 6. Features & Training
            rep = train_and_evaluate(
                station_id=sid,
                horizons=horizons,
                cfg=cfg,
                df_aligned=df_aligned,
                print_summary=False,
            )

            # Extract best 1-day model and RMSE
            stn_h1_scores = [
                s for s in rep.get("scores", [])
                if s.get("station_id") == sid and s.get("horizon_days") == 1
            ]
            if stn_h1_scores:
                best_h1 = min(stn_h1_scores, key=lambda x: x["rmse"])
                best_mod_name = best_h1["model"]
                best_rmse_val = f"{best_h1['rmse']:.4f}"
            else:
                best_mod_name = "N/A"
                best_rmse_val = "N/A"

            summary_records.append({
                "station": sid,
                "rows": len(df_aligned),
                "status": "SUCCESS",
                "best model": best_mod_name,
                "best RMSE (1-day)": best_rmse_val,
                "reason if failed": "-",
            })

        except Exception as exc:
            logger.error("Failed processing station %s: %s", sid, exc)
            summary_records.append({
                "station": sid,
                "rows": len(df_disc_raw) if "df_disc_raw" in locals() else 0,
                "status": "FAILED",
                "best model": "N/A",
                "best RMSE (1-day)": "N/A",
                "reason if failed": str(exc),
            })

    # Print required summary table
    print("\n" + "=" * 115)
    print("CLIMATESHIELD AI — ALL-STATIONS TRAINING & EVALUATION SUMMARY")
    print("=" * 115)
    header = f"{'station':<20} | {'rows':<6} | {'status':<18} | {'best model':<14} | {'best RMSE (1-day)':<18} | {'reason if failed'}"
    print(header)
    print("-" * 115)
    for r in summary_records:
        row_str = f"{r['station']:<20} | {str(r['rows']):<6} | {r['status']:<18} | {r['best model']:<14} | {str(r['best RMSE (1-day)']):<18} | {r['reason if failed']}"
        print(row_str)
    print("=" * 115 + "\n")

    # Auto-update training coverage matrix (Part A Step 1 & 5)
    try:
        from scripts.generate_training_coverage import build_coverage
        build_coverage()
    except Exception as e:
        logger.warning("Could not auto-generate training coverage: %s", e)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_stations": len(stations),
        "success_count": sum(1 for r in summary_records if r["status"] == "SUCCESS"),
        "insufficient_count": sum(1 for r in summary_records if r["status"] == "insufficient data"),
        "failure_count": sum(1 for r in summary_records if r["status"] == "FAILED"),
        "summary": summary_records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate all flood forecasting models.")
    parser.add_argument("--station", type=str, default=None, help="Station ID (e.g. TN_CAU_TRICHY)")
    parser.add_argument("--all-stations", action="store_true", help="Train and evaluate all configured stations")
    args = parser.parse_args()

    if args.all_stations:
        train_all_stations()
    else:
        train_and_evaluate(station_id=args.station)
