import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path("flood-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.temporal_features import build_feature_matrix, get_feature_columns
from src.forecast.engine import generate_forecast

stations = [
    ("TN_CAU_METTUR", "Cauvery at Mettur Dam"),
    ("TN_CAU_TRICHY", "Cauvery at Trichy"),
    ("TN_VAI_MADURAI", "Vaigai at Madurai"),
    ("TN_NOY_TIRUPPUR", "Noyyal at Tiruppur"),
    ("TN_THA_TIRUNELVELI", "Thamirabarani at Tirunelveli"),
    ("TN_PAL_VELLORE", "Palar at Vellore"),
    ("TN_BHA_BHAVANI", "Bhavani at Bhavani"),
    ("TN_CAU_MUSIRI", "Cauvery at Musiri"),
]

origin_date = "2026-10-29"
origin_dt = datetime.strptime(origin_date, "%Y-%m-%d")
origin_doy = origin_dt.timetuple().tm_yday

print(f"Origin Date: {origin_date} (Day of Year: {origin_doy})")
print(f"Season window: DOY {origin_doy - 30} to DOY {origin_doy + 30}\n")

results = []

for sid, sname in stations:
    proc_file = PROJECT_ROOT / "data" / "processed" / f"{sid}_aligned_2026-10-29.csv"
    if not proc_file.exists():
        continue
    df = pd.read_csv(proc_file)
    
    # Training split is 70% chronological
    n_train = int(len(df) * 0.70)
    df_train = df.iloc[:n_train].copy()
    
    # Build feature matrix on training split
    X_train = build_feature_matrix(df_train, horizons=[1])
    X_train["doy"] = pd.to_datetime(X_train["date"]).dt.dayofyear
    
    # Filter same season (doy +/- 30) across training years
    train_season = X_train[(X_train["doy"] >= origin_doy - 30) & (X_train["doy"] <= origin_doy + 30)].copy()
    
    # Get live feature row using generate_forecast
    res = generate_forecast(sid, horizon=1, model_name="persistence")
    # Also build features from the live dataset sliced to origin
    cache_file = sorted((PROJECT_ROOT / "data" / "live_cache").glob(f"{sid}_*.csv"))[-1]
    df_live = pd.read_csv(cache_file)
    is_fc = df_live["is_forecast"].astype("boolean").fillna(False).astype(bool)
    df_live_past = df_live[(~is_fc) & (df_live["date"] <= origin_date)].copy()
    X_live = build_feature_matrix(df_live_past, horizons=[1])
    live_row = X_live.iloc[-1]
    
    features_to_check = [
        "precipitation_sum_roll_mean_14d",
        "precipitation_sum_cumul_14d",
        "precipitation_sum_lag_3",
        "precipitation_sum_lag_7",
        "river_discharge_roc_1d",
    ]
    
    for feat in features_to_check:
        if feat not in X_train.columns or feat not in live_row:
            continue
        
        lv = float(live_row[feat])
        tr_all = pd.to_numeric(X_train[feat], errors="coerce").dropna().values
        tr_sea = pd.to_numeric(train_season[feat], errors="coerce").dropna().values
        
        if len(tr_all) == 0 or len(tr_sea) == 0:
            continue
            
        all_min, all_max = float(np.min(tr_all)), float(np.max(tr_all))
        all_p1, all_p99 = float(np.percentile(tr_all, 1)), float(np.percentile(tr_all, 99))
        
        sea_min, sea_max = float(np.min(tr_sea)), float(np.max(tr_sea))
        sea_p1, sea_p99 = float(np.percentile(tr_sea, 1)), float(np.percentile(tr_sea, 99))
        
        inside_all_p99 = (lv >= all_p1) and (lv <= all_p99)
        inside_sea_p99 = (lv >= sea_p1) and (lv <= sea_p99)
        inside_sea_minmax = (lv >= sea_min) and (lv <= sea_max)
        
        results.append({
            "station": sid,
            "feature": feat,
            "live_val": round(lv, 2),
            "all_min": round(all_min, 2),
            "all_max": round(all_max, 2),
            "all_p1": round(all_p1, 2),
            "all_p99": round(all_p99, 2),
            "sea_min": round(sea_min, 2),
            "sea_max": round(sea_max, 2),
            "sea_p1": round(sea_p1, 2),
            "sea_p99": round(sea_p99, 2),
            "inside_sea_p1_p99": inside_sea_p99,
            "inside_sea_min_max": inside_sea_minmax,
        })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv(PROJECT_ROOT / "reports" / "seasonal_drift_audit.csv", index=False)
