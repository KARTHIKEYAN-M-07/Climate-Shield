"""scripts/check_live.py
Check live data fetching status across all stations with caching and fallback.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from src.data.live import fetch_all_stations_live

def main():
    print("Fetching live data for all stations with short delay, retries, and caching...")
    reports = fetch_all_stations_live(delay_s=0.5, force_refresh=False)

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
    print(f"{'Station':<25} | {'Fetch Status':<14} | {'Source Used':<18} | {'Last Obs Date':<13} | Reason")
    print("-" * 95)
    for _, r in df_rep.iterrows():
        print(f"{r['station_name'][:25]:<25} | {str(r['fetch_status']):<14} | {str(r['source_used']):<18} | {str(r['last_observation_date']):<13} | {str(r['reason'])[:35]}")
    print("=" * 95)

if __name__ == "__main__":
    main()
