"""
src/retrain.py
──────────────
Part C Retraining Script:
Command: `python -m src.retrain --station <ID>`
Refreshes data, retrains models, and replaces saved model artifacts
ONLY if the new validation RMSE is not worse than the currently deployed model.
Logs decisions to reports/retrain_log.jsonl.
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
from src.features.temporal_features import build_feature_matrix, get_feature_columns, get_target_column, HORIZONS
from src.data.split import chronological_split
from src.train import get_model_registry, check_station_data_sanity
from src.utils.config import load_data_config, get_station

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def retrain_station(station_id: str | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retrain and conditionally replace models for station_id."""
    if cfg is None:
        cfg = load_data_config()

    if station_id is None:
        station_id = cfg["default_station_id"]

    reports_dir = _PROJECT_ROOT / cfg["paths"]["reports_dir"]
    models_dir = _PROJECT_ROOT / cfg["paths"]["models_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load current scores to get existing validation RMSE
    current_val_scores: dict[tuple[int, str], float] = {}
    stn_scores_file = reports_dir / f"model_scores_{station_id}.json"
    if stn_scores_file.exists():
        try:
            with open(stn_scores_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for sc in data.get("scores", []):
                    # Use val_rmse if present, or fallback to test rmse
                    h = sc.get("horizon_days", 1)
                    m = sc.get("model", "")
                    rmse = sc.get("val_rmse", sc.get("rmse", float("inf")))
                    current_val_scores[(h, m)] = float(rmse)
        except Exception as e:
            logger.warning("Could not read current scores for %s: %s", station_id, e)

    # 2. Ingest / Load fresh data
    logger.info("Retraining: checking data for station %s...", station_id)
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    rain_files = sorted(raw_dir.glob(f"{station_id}_rainfall_raw_*.csv"))
    disc_files = sorted(raw_dir.glob(f"{station_id}_discharge_raw_*.csv"))

    df_rain_raw = None
    df_disc_raw = None
    try:
        logger.info("Attempting fresh network ingestion for station %s...", station_id)
        raw_paths = ingest_station(station_id, cfg)
        df_rain_raw = pd.read_csv(raw_paths["rainfall"])
        df_disc_raw = pd.read_csv(raw_paths["discharge"])
    except Exception as e:
        logger.warning("Fresh network ingestion failed (%s). Falling back to cached raw files.", e)
        if rain_files and disc_files:
            df_rain_raw = pd.read_csv(rain_files[-1])
            df_disc_raw = pd.read_csv(disc_files[-1])
        else:
            raise RuntimeError(f"No cached raw files available for station {station_id}: {e}")

    # 3. Sanity check
    is_sane, sanity_reason, sanity_details = check_station_data_sanity(
        df_disc=df_disc_raw,
        df_rain=df_rain_raw,
        cfg=cfg,
        station_id=station_id,
    )
    if not is_sane:
        logger.error("Station %s data sanity check failed: %s. Retraining aborted.", station_id, sanity_reason)
        return {
            "station_id": station_id,
            "status": "ABORTED_INSUFFICIENT_DATA",
            "reason": sanity_reason,
            "replaced_count": 0,
            "kept_count": 0,
        }

    # 4. Clean & Align
    df_rain_clean = clean_rainfall(df_rain_raw, cfg, station_id=station_id)
    df_disc_clean = clean_discharge(df_disc_raw, cfg, station_id=station_id)
    df_aligned, align_report = align_datasets(df_rain_clean, df_disc_clean)
    save_processed(df_aligned, station_id, cfg, alignment_report=align_report)

    # 5. Build features and chronological split
    matrix = build_feature_matrix(df_aligned, horizons=HORIZONS)
    feature_cols = get_feature_columns(matrix, horizons=HORIZONS)
    train_df, val_df, test_df = chronological_split(matrix, 0.70, 0.15, 0.15)

    registry = get_model_registry()
    stn_models_dir = models_dir / station_id
    stn_models_dir.mkdir(parents=True, exist_ok=True)

    retrain_log_file = reports_dir / "retrain_log.jsonl"
    decisions = []

    print("\n" + "=" * 115)
    print(f"CLIMATESHIELD AI — RETRAINING AUDIT: {station_id}")
    print("=" * 115)
    print(f"{'horizon':<8} | {'model':<22} | {'current val RMSE':<18} | {'new val RMSE':<16} | {'decision':<14} | {'reason'}")
    print("-" * 115)

    for h in HORIZONS:
        target_col = get_target_column(h)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]

        for mod_name, model in registry.items():
            if mod_name == "persistence":
                X_tr = train_df[["river_discharge"]]
                X_v = val_df[["river_discharge"]]
            else:
                X_tr = X_train
                X_v = X_val

            model.fit(X_tr, y_train)
            val_preds = model.predict(X_v)
            new_val_rmse = float(np.sqrt(np.mean((y_val - val_preds) ** 2)))

            current_rmse = current_val_scores.get((h, mod_name))
            # Decision rule: replace ONLY if new validation RMSE is not worse than current
            if current_rmse is None or new_val_rmse <= (current_rmse * 1.001):
                decision = "REPLACED"
                reason = "New model improved or matched validation RMSE"
                # Save replaced artifacts
                joblib.dump(model, stn_models_dir / f"{mod_name}_h{h}d.joblib")
                joblib.dump(model, stn_models_dir / f"{station_id}_{mod_name}_h{h}d.joblib")
                joblib.dump(model, models_dir / f"{station_id}_{mod_name}_h{h}d.joblib")
            else:
                decision = "KEPT_CURRENT"
                reason = f"New val RMSE ({new_val_rmse:.4f}) > current ({current_rmse:.4f})"

            decisions.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "station_id": station_id,
                "horizon_days": h,
                "model": mod_name,
                "current_val_rmse": current_rmse,
                "new_val_rmse": round(new_val_rmse, 4),
                "decision": decision,
                "reason": reason,
            })

            curr_str = f"{current_rmse:.4f}" if current_rmse is not None else "None (New)"
            print(f"{h:<8} | {mod_name:<22} | {curr_str:<18} | {new_val_rmse:<16.4f} | {decision:<14} | {reason}")

    print("=" * 115 + "\n")

    # Append to retrain log JSONL
    with open(retrain_log_file, "a", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")

    logger.info("Logged %d retraining decisions to %s", len(decisions), retrain_log_file)
    return {
        "station_id": station_id,
        "status": "SUCCESS",
        "decisions": decisions,
        "replaced_count": sum(1 for d in decisions if d["decision"] == "REPLACED"),
        "kept_count": sum(1 for d in decisions if d["decision"] == "KEPT_CURRENT"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain station models with validation replacement gate.")
    parser.add_argument("--station", type=str, default=None, help="Station ID (e.g. TN_CAU_TRICHY)")
    args = parser.parse_args()

    retrain_station(station_id=args.station)
