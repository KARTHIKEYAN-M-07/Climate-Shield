from datetime import datetime, timezone, timedelta
import pandas as pd
from src.forecast.engine import generate_forecast
from src.data.live import fetch_station_live_data
from src.utils.config import load_all_stations

stations = load_all_stations()
kolkata_tz = timezone(timedelta(hours=5, minutes=30))
today_kolk = datetime.now(kolkata_tz).strftime("%Y-%m-%d")

print(f"{'Station':<25} | {'Origin Date':<12} | {'Is Past?':<9} | {'PSI / z':<10} | {'Drift Status':<20} | {'Risk':<12} | {'Status':<10}")
print("-" * 105)

for s in stations:
    sid = s["id"]
    sname = s["name"]
    try:
        live_res = fetch_station_live_data(sid, s)
        df_live = live_res["df"] if not live_res["df"].empty else None
    except Exception:
        df_live = None

    res = generate_forecast(sid, horizon=1, model_name="xgboost", df_data=df_live)
    val_str = f"{res.drift_value:.3f}" if res.drift_value is not None else "N/A"
    is_past = (res.last_observation_time <= today_kolk) if res.last_observation_time else True

    print(f"{sname[:25]:<25} | {str(res.last_observation_time):<12} | {'Yes' if is_past else 'No':<9} | {val_str:<10} | {str(res.drift_status):<20} | {str(res.risk.get('category')):<12} | {str(res.status):<10}")
print("-" * 105)
