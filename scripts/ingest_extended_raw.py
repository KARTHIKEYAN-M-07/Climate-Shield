"""scripts/ingest_extended_raw.py
──────────────────────────────
Fetches historical raw rainfall and discharge series for all configured stations
up to the latest date (2026-10-28) from Open-Meteo archive and flood APIs.
Saves new raw files timestamped with retrieval date 2026-10-29.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.utils.config import load_data_config
from src.data.ingestion import fetch_rainfall, fetch_discharge, save_raw

def main():
    cfg = load_data_config()
    stations = cfg.get("stations", [])
    retrieval_date = "2026-10-29"
    period = cfg["period"]
    
    print(f"Ingesting extended raw data ({period['start']} -> {period['end']}) for {len(stations)} stations...")
    results = []
    
    for stn in stations:
        sid = stn["id"]
        sname = stn.get("name", sid)
        print(f"\nProcessing {sid} ({sname})...")
        
        try:
            print(f" - Fetching rainfall ({period['start']} -> {period['end']})...")
            df_rain = fetch_rainfall(stn, cfg)
            p_rain = save_raw(df_rain, sid, "rainfall", cfg, retrieval_date=retrieval_date)
            
            time.sleep(1.2)  # Polite rate limiting between API endpoints
            
            print(f" - Fetching modelled river discharge ({period['start']} -> {period['end']})...")
            df_disc = fetch_discharge(stn, cfg)
            p_disc = save_raw(df_disc, sid, "discharge", cfg, retrieval_date=retrieval_date)
            
            time.sleep(1.2)  # Polite rate limiting between stations
            
            results.append({
                "station_id": sid,
                "station_name": sname,
                "status": "OK",
                "rain_rows": len(df_rain),
                "disc_rows": len(df_disc),
                "rain_file": str(p_rain),
                "disc_file": str(p_disc),
            })
            print(f"   => Success: {len(df_rain)} rain rows, {len(df_disc)} discharge rows")
            
        except Exception as exc:
            print(f"   => Failed for {sid}: {exc}")
            results.append({
                "station_id": sid,
                "station_name": sname,
                "status": "FAILED",
                "rain_rows": 0,
                "disc_rows": 0,
                "reason": str(exc),
            })
            time.sleep(2.0)
            
    df_summary = pd.DataFrame(results)
    Path("reports").mkdir(exist_ok=True)
    df_summary.to_csv("reports/raw_ingestion_summary.csv", index=False)
    print("\n" + "=" * 80)
    print("Extended Raw Ingestion Completed:")
    print(df_summary[["station_id", "status", "rain_rows", "disc_rows"]].to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()
