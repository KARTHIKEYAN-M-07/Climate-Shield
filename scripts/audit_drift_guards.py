import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.live import fetch_station_live_data
from src.features.temporal_features import build_feature_matrix, get_feature_columns
from src.utils.config import load_all_stations, load_thresholds_config

thresh_cfg = load_thresholds_config()
drift_policy = thresh_cfg.get("drift_risk_policy", {})
std_floor = float(drift_policy.get("min_feature_std_floor", 0.05))
excluded_patterns = drift_policy.get("excluded_feature_patterns", ["_flag"])

for sid in ["TN_CAU_TRICHY", "TN_THA_TIRUNELVELI"]:
    stn = next(s for s in load_all_stations() if s["id"] == sid)
    live_res = fetch_station_live_data(sid, stn)
    df_live = live_res["df"]
    
    # Processed training split
    proc_files = sorted(Path("data/processed").glob(f"{sid}_aligned_*.csv"))
    df_hist = pd.read_csv(proc_files[-1])
    n_tr = int(len(df_hist) * 0.70)
    df_tr = df_hist.iloc[:n_tr]
    
    X_tr = build_feature_matrix(df_tr, horizons=[1]).dropna(subset=["target_discharge_h1d"])
    X_live = build_feature_matrix(df_live, horizons=[1])
    last_row = X_live.iloc[[-1]]
    
    feat_cols = [
        c for c in get_feature_columns(X_tr, horizons=[1])
        if not any(pat in c for pat in excluded_patterns)
        and c not in ("source", "source_used", "fetched_at_utc", "is_forecast", "date", "date_time")
    ]
    
    scores = []
    for col in feat_cols:
        if col in X_tr.columns and col in last_row.columns:
            tr_series = pd.to_numeric(X_tr[col], errors="coerce").dropna().astype(float)
            lv_series = pd.to_numeric(last_row[col], errors="coerce").dropna()
            if tr_series.empty or lv_series.empty:
                continue
            lv_val = float(lv_series.iloc[0])
            tr_m = float(tr_series.mean())
            tr_s = float(tr_series.std(ddof=1))
            
            # Guard against zero or near-zero variance
            if tr_s < 1e-4:
                # Excluded from z-score
                continue
                
            tr_s_guarded = max(tr_s, std_floor)
            z = (lv_val - tr_m) / tr_s_guarded
            scores.append({
                "feature": col,
                "live_value": round(lv_val, 2),
                "train_mean": round(tr_m, 2),
                "train_std": round(tr_s, 2),
                "guarded_std": round(tr_s_guarded, 2),
                "z_score": round(z, 2),
                "abs_z": abs(z),
            })
            
    df_scores = pd.DataFrame(scores).sort_values(by="abs_z", ascending=False)
    print(f"\n=== Top 5 Guarded Drifting Features for {sid} ===")
    print(df_scores.head(5).to_string(index=False))
