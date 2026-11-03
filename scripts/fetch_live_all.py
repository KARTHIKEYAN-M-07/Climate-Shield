import json
from pathlib import Path
import pandas as pd
from src.data.live import fetch_all_stations_live
from src.utils.config import load_all_stations

print("Fetching live data for all stations with short delay, retries, and caching...")
reports = fetch_all_stations_live(delay_s=1.5, force_refresh=True)

rows = []
for sid, rep in reports.items():
    rows.append({
        "station_id": sid,
        "station_name": rep["station_name"],
        "fetch_status": rep["fetch_status"],
        "source_used": rep["source_used"],
        "last_observation_date": rep["last_observation_date"],
        "reason": rep["reason"],
    })

df_rep = pd.DataFrame(rows)
Path("reports").mkdir(exist_ok=True)
df_rep.to_csv("reports/live_fetch_status.csv", index=False)

print("\n" + "=" * 95)
print(f"{'Station':<25} | {'Fetch Status':<12} | {'Source Used':<16} | {'Last Obs Date':<13} | Reason")
print("-" * 95)
for _, r in df_rep.iterrows():
    print(f"{r['station_name'][:25]:<25} | {str(r['fetch_status']):<12} | {str(r['source_used']):<16} | {str(r['last_observation_date']):<13} | {str(r['reason'])[:35]}")
print("=" * 95)
