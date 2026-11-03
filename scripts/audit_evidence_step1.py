import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("flood-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

stations = [
    ("TN_CAU_TRICHY", "Cauvery at Trichy"),
    ("TN_VAI_MADURAI", "Vaigai at Madurai"),
    ("TN_NOY_TIRUPPUR", "Noyyal at Tiruppur"),
    ("TN_THA_TIRUNELVELI", "Thamirabarani at Tirunelveli"),
]

for sid, sname in stations:
    print("=" * 80)
    print(f"STATION: {sname} ({sid})")
    print("=" * 80)
    
    # 1. Historical archive (training source)
    hist_file = PROJECT_ROOT / "data" / "processed" / f"{sid}_aligned_2026-10-29.csv"
    if not hist_file.exists():
        print(f"Error: {hist_file} does not exist")
        continue
    df_hist = pd.read_csv(hist_file)
    
    # 2. Live cache file
    cache_dir = PROJECT_ROOT / "data" / "live_cache"
    cache_files = sorted(cache_dir.glob(f"{sid}_*.csv"))
    if not cache_files:
        print(f"Error: No live cache file found for {sid}")
        continue
    df_live = pd.read_csv(cache_files[-1])
    
    # Separate past vs forecast
    df_live["is_forecast"] = df_live["is_forecast"].astype("boolean").fillna(False).astype(bool)
    past_live = df_live[~df_live["is_forecast"]].copy()
    
    print(f"Total live rows: {len(df_live)}")
    print(f"Past/observed rows in live: {len(past_live)}")
    print(f"Future/forecast rows in live: {len(df_live) - len(past_live)}")
    
    # Check NaN and zero counts in past_live
    nan_count = past_live["precipitation_sum"].isna().sum()
    zero_count = (past_live["precipitation_sum"] == 0.0).sum()
    is_fc_in_past = past_live["is_forecast"].any()
    print(f"Past live precipitation NaNs: {nan_count}")
    print(f"Past live precipitation Zeroes: {zero_count} / {len(past_live)}")
    print(f"Any is_forecast in past_live: {is_fc_in_past}")
    
    # Last 30 past days
    past_live_30 = past_live.tail(30).copy()
    print(f"Last 30 past days date range: {past_live_30['date'].iloc[0]} to {past_live_30['date'].iloc[-1]}")
    
    # 14-day window check
    past_live_14 = past_live.tail(14).copy()
    is_14_strictly_past = (~past_live_14["is_forecast"]).all()
    print(f"14-day window strictly past/observed: {is_14_strictly_past} ({past_live_14['date'].iloc[0]} to {past_live_14['date'].iloc[-1]})")
    
    # Merge on date for side-by-side comparison
    merged = pd.merge(
        past_live_30[["date", "precipitation_sum", "is_forecast"]].rename(columns={"precipitation_sum": "live_rainfall"}),
        df_hist[["date", "precipitation_sum"]].rename(columns={"precipitation_sum": "hist_rainfall"}),
        on="date",
        how="left"
    )
    merged["diff"] = (merged["live_rainfall"] - merged["hist_rainfall"]).round(4)
    
    print("\nLast 30 days side-by-side comparison (Live vs Historical Archive):")
    print(merged[["date", "live_rainfall", "hist_rainfall", "diff", "is_forecast"]].to_string(index=False))
    print()
