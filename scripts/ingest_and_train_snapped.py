"""scripts/ingest_and_train_snapped.py
───────────────────────────────────
Pipeline execution for stations with updated snapped coordinates:
1. Ingests raw rainfall & GloFAS river discharge (2014-01-01 -> 2026-10-28).
2. Cleans rainfall and discharge datasets.
3. Aligns datasets to create data/processed/<station_id>_aligned_2026-10-29.csv.
4. Recalibrates statistical thresholds on the 70% chronological training split.
5. Ingests and refreshes live cache for these stations.
6. Trains all models (persistence, ridge, tree models, boosting) across horizons 1, 3, 7.
7. Evaluates test set performance (RMSE, MAE, R2, KGE, beats_persistence).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_data_config, load_thresholds_config
from src.data.ingestion import fetch_rainfall, fetch_discharge, save_raw
from src.data.validation import validate_dataframe
from src.data.cleaning import clean_rainfall, clean_discharge
from src.data.alignment import align_datasets, save_processed
from src.data.live import fetch_station_live_data
from src.risk.thresholds import compute_station_thresholds
from src.forecast.engine import evaluate_station_signal
from src.train import train_and_evaluate

SNAPPED_STATION_IDS = ["TN_CAU_METTUR", "TN_CAU_MUSIRI", "TN_BHA_BHAVANI"]


def main():
    cfg = load_data_config()
    thresh_cfg = load_thresholds_config()
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    proc_dir = _PROJECT_ROOT / "data" / "processed"
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    retrieval_date = "2026-10-29"

    stations = [s for s in cfg.get("stations", []) if s["id"] in SNAPPED_STATION_IDS]
    period = cfg["period"]

    print("=" * 80)
    print(f"PIPELINE RUN FOR SNAPPED STATIONS: {SNAPPED_STATION_IDS}")
    print(f"Period: {period['start']} -> {period['end']}")
    print("=" * 80)

    # 1. Ingestion
    print("\n--- STEP 1: Ingesting Raw Data ---")
    for stn in stations:
        sid = stn["id"]
        sname = stn["name"]
        print(f"Fetching raw data for {sid} ({sname}) at ({stn['latitude']}, {stn['longitude']})...")
        try:
            # Remove old raw files to allow write with snapped coordinates
            for old_f in raw_dir.glob(f"{sid}_*_raw_{retrieval_date}.*"):
                try:
                    old_f.unlink()
                except Exception:
                    pass

            df_rain = fetch_rainfall(stn, cfg)
            p_rain = save_raw(df_rain, sid, "rainfall", cfg, retrieval_date=retrieval_date)
            time.sleep(1.0)

            df_disc = fetch_discharge(stn, cfg)
            p_disc = save_raw(df_disc, sid, "discharge", cfg, retrieval_date=retrieval_date)
            time.sleep(1.0)
            print(f"  Saved raw files: rain={len(df_rain)} rows, disc={len(df_disc)} rows")
        except Exception as exc:
            print(f"  Error fetching raw data for {sid}: {exc}")

    # 2. Cleaning and Alignment
    print("\n--- STEP 2: Validation, Cleaning, and Alignment ---")
    aligned_dfs = {}
    for stn in stations:
        sid = stn["id"]
        print(f"Processing {sid}...")
        rain_files = sorted(raw_dir.glob(f"{sid}_rainfall_raw_{retrieval_date}.csv"))
        disc_files = sorted(raw_dir.glob(f"{sid}_discharge_raw_{retrieval_date}.csv"))
        if not rain_files or not disc_files:
            print(f"  Missing raw files for {sid}. Skipping.")
            continue

        df_rain_raw = pd.read_csv(rain_files[-1])
        df_disc_raw = pd.read_csv(disc_files[-1])

        val_rain = validate_dataframe(df_rain_raw, "rainfall", sid, cfg)
        val_disc = validate_dataframe(df_disc_raw, "discharge", sid, cfg)

        df_rain_clean = clean_rainfall(df_rain_raw, cfg, station_id=sid)
        df_disc_clean = clean_discharge(df_disc_raw, cfg, station_id=sid)

        df_aligned, align_rep = align_datasets(df_rain_clean, df_disc_clean)
        saved_path = save_processed(df_aligned, sid, cfg, alignment_report=align_rep)
        aligned_dfs[sid] = df_aligned
        print(f"  Aligned {len(df_aligned)} rows -> {saved_path}")

    # 3. Recalibrate Statistical Thresholds from 70% Training Split
    print("\n--- STEP 3: Recalibrating Thresholds (Training Split Only) ---")
    thresh_file = _PROJECT_ROOT / "configs" / "thresholds.yaml"
    with open(thresh_file, "r", encoding="utf-8") as tf:
        thresh_data = yaml.safe_load(tf)

    for sid, df_aligned in aligned_dfs.items():
        n_train = int(len(df_aligned) * 0.70)
        df_train = df_aligned.iloc[:n_train].copy()
        q_train = df_train["river_discharge"].dropna().values

        stn_thresh = compute_station_thresholds(q_train)
        stn_thresh["metadata"] = {
            "training_rows": n_train,
            "training_period_start": str(df_train["date"].iloc[0]),
            "training_period_end": str(df_train["date"].iloc[-1]),
            "calibrated_from": "70% chronological training split only",
        }
        thresh_data["stations"][sid] = stn_thresh
        p_basis = stn_thresh.get("percentile_basis", {})
        print(f"  {sid} thresholds: p50={p_basis.get('p50')}, p75={p_basis.get('p75')}, p90={p_basis.get('p90')}, p95={p_basis.get('p95')}")

    thresh_data["metadata"]["last_recalibration_utc"] = datetime.now(timezone.utc).isoformat()
    with open(thresh_file, "w", encoding="utf-8") as tf:
        yaml.dump(thresh_data, tf, default_flow_style=False, sort_keys=False)
    print(f"Updated {thresh_file}")

    # 4. Refresh Live Cache with Snapped Coordinates
    print("\n--- STEP 4: Refreshing Live Cache ---")
    for stn in stations:
        sid = stn["id"]
        try:
            print(f"Refreshing live fetch for {sid}...")
            res_live = fetch_station_live_data(sid, stn, force_refresh=True)
            print(f"  Status: {res_live['status']}, last_obs: {res_live.get('last_observation_date')}")
        except Exception as e:
            print(f"  Live fetch note for {sid}: {e}")

    # 5. Train Models for Snapped Stations
    print("\n--- STEP 5: Training All Models Across Horizons 1, 3, 7 ---")
    for sid in SNAPPED_STATION_IDS:
        print(f"\n=======================================================")
        print(f"Training models for {sid}...")
        print(f"=======================================================")
        try:
            train_and_evaluate(station_id=sid)
            print(f"Training completed successfully for {sid}!")
        except Exception as exc:
            print(f"Training error for {sid}: {exc}")

    # 6. Audit Signal and Evaluation Metrics
    print("\n--- STEP 6: Hydrologic Signal & Model Evaluation Audit ---")
    scores_file = _PROJECT_ROOT / "reports" / "model_scores_all.csv"
    if scores_file.exists():
        df_scores = pd.read_csv(scores_file)
        sub = df_scores[df_scores["station"].isin(SNAPPED_STATION_IDS)]
        print(sub[["station", "horizon", "model", "test_rmse", "test_mae", "test_kge", "beats_persistence"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
