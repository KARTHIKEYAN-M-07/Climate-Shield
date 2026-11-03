import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path("flood-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.temporal_features import build_feature_matrix, get_feature_columns

stations = [
    ("TN_CAU_TRICHY", "Cauvery at Trichy"),
    ("TN_CAU_METTUR", "Cauvery at Mettur Dam"),
    ("TN_CAU_MUSIRI", "Cauvery at Musiri"),
    ("TN_VAI_MADURAI", "Vaigai at Madurai"),
    ("TN_NOY_TIRUPPUR", "Noyyal at Tiruppur"),
    ("TN_THA_TIRUNELVELI", "Thamirabarani at Tirunelveli"),
    ("TN_PAL_VELLORE", "Palar at Vellore"),
    ("TN_BHA_BHAVANI", "Bhavani at Bhavani"),
]

origin_date = "2026-10-29"
origin_dt = datetime.strptime(origin_date, "%Y-%m-%d")
origin_doy = origin_dt.timetuple().tm_yday
season_window_days = 30
min_outside_thresh = 2
p_low, p_high = 1.0, 99.0

print(f"Origin Date: {origin_date} (DOY: {origin_doy})")
print(f"Seasonal window: DOY {origin_doy - season_window_days} to {origin_doy + season_window_days}")
print(f"Rule: OUT_OF_DISTRIBUTION if >= {min_outside_thresh} features outside [{p_low}th, {p_high}th] seasonal percentiles\n")

for sid, sname in stations:
    proc_file = PROJECT_ROOT / "data" / "processed" / f"{sid}_aligned_2026-10-29.csv"
    if not proc_file.exists():
        continue
    df = pd.read_csv(proc_file)
    n_train = int(len(df) * 0.70)
    df_train = df.iloc[:n_train].copy()
    
    # Feature matrix on training split
    X_train = build_feature_matrix(df_train, horizons=[1])
    X_train["doy"] = pd.to_datetime(X_train["date"]).dt.dayofyear
    
    # Same season rows in training split
    season_mask = (X_train["doy"] >= origin_doy - season_window_days) & (X_train["doy"] <= origin_doy + season_window_days)
    X_train_season = X_train[season_mask].copy()
    
    # Live feature row
    cache_file = sorted((PROJECT_ROOT / "data" / "live_cache").glob(f"{sid}_*.csv"))[-1]
    df_live = pd.read_csv(cache_file)
    is_fc = df_live["is_forecast"].astype("boolean").fillna(False).astype(bool)
    df_live_past = df_live[(~is_fc) & (df_live["date"] <= origin_date)].copy()
    X_live = build_feature_matrix(df_live_past, horizons=[1])
    live_row = X_live.iloc[-1]
    
    feat_cols = [c for c in get_feature_columns(X_train, horizons=[1]) if not c.endswith("_flag") and not c in ("date", "date_time", "source", "source_used", "is_forecast")]
    
    violations = []
    scores = []
    
    for c in feat_cols:
        if c not in X_train_season.columns or c not in live_row:
            continue
        lv = float(live_row[c])
        s_vals = pd.to_numeric(X_train_season[c], errors="coerce").dropna().values
        if len(s_vals) < 10 or not np.isfinite(lv):
            continue
        std = float(np.std(s_vals))
        if std < 1e-4:
            continue
        mean = float(np.mean(s_vals))
        p1 = float(np.percentile(s_vals, p_low))
        p99 = float(np.percentile(s_vals, p_high))
        
        is_outside = (lv < p1) or (lv > p99)
        z_sea = (lv - mean) / max(std, 0.05)
        
        dist = 0.0
        if lv < p1:
            dist = p1 - lv
        elif lv > p99:
            dist = lv - p99
            
        scores.append({
            "feature": c,
            "live": round(lv, 2),
            "range": f"[{p1:.2f}, {p99:.2f}]",
            "z_sea": round(z_sea, 2),
            "abs_z": abs(z_sea),
            "is_outside": is_outside,
            "dist": dist,
        })
        if is_outside:
            violations.append(scores[-1])
            
    scores.sort(key=lambda x: (x["dist"] > 0, x["abs_z"]), reverse=True)
    n_outside = len(violations)
    status = "OUT_OF_DISTRIBUTION" if n_outside >= min_outside_thresh else "IN_DISTRIBUTION"
    
    top = scores[0] if scores else {"feature": "None", "live": 0, "range": "[]", "z_sea": 0}
    print(f"{sid:20s} | {n_outside:2d} outside | Top: {top['feature']} (live={top['live']} vs {top['range']}, z={top['z_sea']:+.2f}) | Status: {status}")
