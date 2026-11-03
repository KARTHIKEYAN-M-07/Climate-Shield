"""scripts/prepare_processed_and_calibrate.py
─────────────────────────────────────────────
1. Validates raw files for all 9 stations (dated 2026-10-29).
2. Cleans rainfall and discharge datasets.
3. Aligns datasets to create data/processed/<station_id>_aligned_2026-10-29.csv.
4. Calibrates statistical prototype thresholds strictly on the TRAINING split (70% chronological).
5. Updates configs/thresholds.yaml with training date range, sample count, and academic notice.
6. Runs station signal audit across all stations and saves reports/station_signal_audit.csv.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

from src.utils.config import load_data_config, load_thresholds_config
from src.data.validation import validate_dataframe
from src.data.cleaning import clean_rainfall, clean_discharge
from src.data.alignment import align_datasets, save_processed
from src.risk.thresholds import compute_station_thresholds

def main():
    cfg = load_data_config()
    thresh_cfg = load_thresholds_config()
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    stations = cfg.get("stations", [])
    print(f"Processing and aligning {len(stations)} stations up to 2026-10-28...")
    
    aligned_info = {}
    
    for stn in stations:
        sid = stn["id"]
        sname = stn.get("name", sid)
        print(f"\nProcessing {sid} ({sname})...")
        
        rain_files = sorted(raw_dir.glob(f"{sid}_rainfall_raw_2026-10-29.csv"))
        disc_files = sorted(raw_dir.glob(f"{sid}_discharge_raw_2026-10-29.csv"))
        
        if not rain_files or not disc_files:
            print(f"Missing 2026-10-29 raw files for {sid}. Skipping alignment.")
            continue
            
        df_rain_raw = pd.read_csv(rain_files[-1])
        df_disc_raw = pd.read_csv(disc_files[-1])
        
        # Validation
        val_rain = validate_dataframe(df_rain_raw, "rainfall", sid, cfg)
        val_disc = validate_dataframe(df_disc_raw, "discharge", sid, cfg)
        
        # Cleaning
        df_rain_clean = clean_rainfall(df_rain_raw, cfg, station_id=sid)
        df_disc_clean = clean_discharge(df_disc_raw, cfg, station_id=sid)
        
        # Alignment
        df_aligned, align_rep = align_datasets(df_rain_clean, df_disc_clean)
        saved_path = save_processed(df_aligned, sid, cfg, alignment_report=align_rep)
        print(f"Aligned {len(df_aligned)} rows -> {saved_path}")
        aligned_info[sid] = df_aligned

    # 2. Recalibrate thresholds on TRAINING split only (70% chronological)
    print("\n" + "=" * 80)
    print("Recalibrating prototype thresholds on 70% chronological TRAINING split...")
    print("=" * 80)
    
    calib_records = []
    
    for sid, df in aligned_info.items():
        if "river_discharge" not in df.columns:
            continue
            
        total_rows = len(df)
        n_train = int(total_rows * 0.70)
        train_df = df.iloc[:n_train].copy().reset_index(drop=True)
        train_discharge = pd.to_numeric(train_df["river_discharge"], errors="coerce").dropna()
        train_discharge = train_discharge[train_discharge >= 0]
        
        if train_discharge.empty or len(train_discharge) < 50:
            print(f"Station {sid}: Insufficient valid training discharge observations ({len(train_discharge)}).")
            continue
            
        start_date = str(train_df["date"].iloc[0])
        end_date = str(train_df["date"].iloc[-1])
        
        th_data = compute_station_thresholds(train_discharge)
        th_data["metadata"] = {
            "source": "prototype statistical thresholds, not official CWC levels",
            "training_split": "first 70% chronological",
            "training_period": f"{start_date} to {end_date}",
            "training_rows": len(train_discharge),
            "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        
        if "stations" not in thresh_cfg:
            thresh_cfg["stations"] = {}
        thresh_cfg["stations"][sid] = th_data
        
        p = th_data["percentile_basis"]
        calib_records.append({
            "station_id": sid,
            "train_rows": len(train_discharge),
            "period": f"{start_date} -> {end_date}",
            "p50": p["p50"],
            "p75": p["p75"],
            "p90": p["p90"],
            "p95": p["p95"],
        })
        print(f"{sid:<20} | Train: {len(train_discharge)} rows ({start_date} to {end_date}) | p50={p['p50']:.2f}, p75={p['p75']:.2f}, p90={p['p90']:.2f}, p95={p['p95']:.2f}")

    thresh_cfg["metadata"]["last_recalibration_utc"] = datetime.now(timezone.utc).isoformat()
    thresh_cfg["metadata"]["source_notice"] = "prototype statistical thresholds, not official CWC levels"
    
    with open("configs/thresholds.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(thresh_cfg, f, sort_keys=False)
    print("\nSaved updated configs/thresholds.yaml")
    
    df_calib = pd.DataFrame(calib_records)
    df_calib.to_csv("reports/threshold_calibration_training_split.csv", index=False)

    # 3. Re-run Station Signal Audit
    print("\n" + "=" * 80)
    print("Auditing Station Signal Quality across all stations...")
    print("=" * 80)
    
    policy = thresh_cfg.get("station_signal_policy", {
        "min_rows": 500,
        "min_cv": 0.80,
        "min_p90_m3s": 0.20,
        "max_near_zero_share": 0.20,
    })
    min_rows = int(policy["min_rows"])
    min_cv = float(policy["min_cv"])
    min_p90 = float(policy["min_p90_m3s"])
    max_zero_share = float(policy["max_near_zero_share"])
    
    signal_rows = []
    for stn in stations:
        sid = stn["id"]
        sname = stn.get("name", sid)
        if sid not in aligned_info:
            signal_rows.append({
                "station": sid,
                "name": sname,
                "rows": 0,
                "train_mean": 0.0,
                "train_std": 0.0,
                "cv": 0.0,
                "p90": 0.0,
                "near_zero_pct": 100.0,
                "status": "Insufficient signal",
                "decision_reason": f"rows (0) < {min_rows}, unmodeled coastal grid",
            })
            continue
            
        df = aligned_info[sid]
        n_rows = len(df)
        n_train = int(n_rows * 0.70)
        train_df = df.iloc[:n_train]
        q = train_df["river_discharge"].dropna()
        
        if q.empty or len(q) < 10:
            signal_rows.append({
                "station": sid,
                "name": sname,
                "rows": n_rows,
                "train_mean": 0.0,
                "train_std": 0.0,
                "cv": 0.0,
                "p90": 0.0,
                "near_zero_pct": 100.0,
                "status": "Insufficient signal",
                "decision_reason": "No valid discharge measurements in training set",
            })
            continue
            
        mean_val = float(q.mean())
        std_val = float(q.std(ddof=1))
        cv_val = float(std_val / mean_val) if mean_val > 1e-4 else 0.0
        p90_val = float(np.percentile(q, 90))
        near_zero_share = float((q <= 0.01).mean())
        
        reasons = []
        if n_rows < min_rows:
            reasons.append(f"rows ({n_rows}) < {min_rows}")
        if p90_val < min_p90:
            reasons.append(f"p90 ({p90_val:.2f}) < {min_p90:.2f} m3/s")
        if cv_val < min_cv:
            reasons.append(f"CV ({cv_val:.2f}) < {min_cv:.2f}")
        if near_zero_share > max_zero_share:
            reasons.append(f"near-zero ({near_zero_share*100:.1f}%) > {max_zero_share*100:.0f}%")
            
        status = "Insufficient signal" if reasons else "Sufficient signal"
        dec_reason = "; ".join(reasons) if reasons else "Passes all criteria"
        
        signal_rows.append({
            "station": sid,
            "name": sname,
            "rows": n_rows,
            "train_mean": round(mean_val, 3),
            "train_std": round(std_val, 3),
            "cv": round(cv_val, 2),
            "p90": round(p90_val, 3),
            "near_zero_pct": round(near_zero_share * 100, 1),
            "status": status,
            "decision_reason": dec_reason,
        })
        
    df_sig = pd.DataFrame(signal_rows)
    df_sig.to_csv("reports/station_signal_audit.csv", index=False)
    print(df_sig[["station", "rows", "cv", "p90", "near_zero_pct", "status", "decision_reason"]].to_string(index=False))

if __name__ == "__main__":
    main()
