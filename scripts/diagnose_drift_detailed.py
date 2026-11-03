import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.features.temporal_features import build_feature_matrix, get_feature_columns
from src.utils.config import load_all_stations, load_thresholds_config
from src.data.live import fetch_station_live_data
from src.monitoring.drift import compute_psi

kolkata_tz = timezone(timedelta(hours=5, minutes=30))
today_kolk = datetime.now(kolkata_tz).strftime("%Y-%m-%d")

def diagnose_station(station_id: str):
    stations = load_all_stations()
    stn = next(s for s in stations if s["id"] == station_id)
    
    # 1. Fetch data (uses cache)
    live_info = fetch_station_live_data(station_id, stn)
    df_live = live_info["df"]
    
    # 2. Slice to past observations
    df_live["is_forecast"] = df_live["date"] > today_kolk
    df_past = df_live[~df_live["is_forecast"] & df_live["river_discharge"].notna()].copy()
    
    # Fill flags
    flag_cols = [c for c in df_past.columns if "flag" in c]
    for fc in flag_cols:
        df_past[fc] = df_past[fc].fillna(0.0)
        
    # Training split
    # From historical processed or the first 70% of historical series
    proc_files = sorted(Path("data/processed").glob(f"{station_id}_aligned_*.csv"))
    if proc_files:
        df_hist = pd.read_csv(proc_files[-1])
        n_tr = int(len(df_hist) * 0.70)
        df_tr = df_hist.iloc[:n_tr].copy()
    else:
        n_tr = int(len(df_past) * 0.70)
        df_tr = df_past.iloc[:n_tr].copy()
        
    X_tr = build_feature_matrix(df_tr, horizons=[1]).dropna(subset=["target_discharge_h1d"])
    X_live = build_feature_matrix(df_past, horizons=[1])
    live_row = X_live.iloc[[-1]]
    live_date = df_past["date"].iloc[-1]
    
    feat_cols = [c for c in get_feature_columns(X_tr, horizons=[1]) if c not in ("source", "source_used", "fetched_at_utc", "is_forecast", "date", "date_time")]
    
    results = []
    for col in feat_cols:
        if col in X_tr.columns and col in live_row.columns:
            tr_series = pd.to_numeric(X_tr[col], errors="coerce").dropna().astype(float)
            lv_series = pd.to_numeric(live_row[col], errors="coerce").dropna()
            if tr_series.empty or lv_series.empty:
                continue
            
            lv_val = float(lv_series.iloc[0])
            tr_m = float(tr_series.mean())
            tr_s = float(tr_series.std(ddof=1))
            tr_min = float(tr_series.min())
            tr_max = float(tr_series.max())
            
            # z-score with no floor
            z_raw = (lv_val - tr_m) / tr_s if tr_s > 0 else np.nan
            
            # 1-row PSI vs 10-bin training quantiles
            try:
                q_edges = np.percentile(tr_series, np.linspace(0, 100, 11))
                if len(np.unique(q_edges)) == 11:
                    hist_live, _ = np.histogram([lv_val], bins=q_edges)
                    p = (hist_live + 1e-4) / (1.0 + 10 * 1e-4)
                    q = np.full(10, 0.1)
                    psi_1row = float(np.sum((p - q) * np.log(p / q)))
                else:
                    psi_1row = np.nan
            except Exception:
                psi_1row = np.nan
                
            results.append({
                "feature": col,
                "live_value": lv_val,
                "train_mean": tr_m,
                "train_std": tr_s,
                "train_min": tr_min,
                "train_max": tr_max,
                "z_score": z_raw,
                "abs_z": abs(z_raw) if pd.notna(z_raw) else 0.0,
                "psi": psi_1row,
            })
            
    df_res = pd.DataFrame(results).sort_values(by="abs_z", ascending=False)
    return df_res, live_date

print("================ Trichy (TN_CAU_TRICHY) Detailed Drift ================")
df_trichy, date_trichy = diagnose_station("TN_CAU_TRICHY")
print(f"Origin Date: {date_trichy}")
print(df_trichy.to_string(index=False))

print("\n================ Tirunelveli (TN_THA_TIRUNELVELI) Detailed Drift ================")
df_tiru, date_tiru = diagnose_station("TN_THA_TIRUNELVELI")
print(f"Origin Date: {date_tiru}")
print(df_tiru.to_string(index=False))
