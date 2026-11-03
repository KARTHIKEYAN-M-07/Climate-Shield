"""scripts/diagnose_why_blocked.py
Generates reports/why_blocked.csv diagnosing why forecasts or risks are blocked or flagged
across all stations in both Live and Historical Replay modes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_all_stations, load_thresholds_config
from src.forecast.engine import generate_forecast, get_highest_flow_test_date, evaluate_station_signal
from src.data.live import fetch_station_live_data

def main():
    stations = load_all_stations()
    thresh_cfg = load_thresholds_config()
    models_dir = PROJECT_ROOT / "models"

    results = []
    print("Running diagnosis across all stations in LIVE and HISTORICAL REPLAY modes...\n")

    for mode in ["live", "historical_replay"]:
        for stn in stations:
            sid = stn["id"]
            sname = stn["name"]

            # Model exists
            cand_models = list((models_dir / sid).glob("*.joblib"))
            model_exists = len(cand_models) > 0

            # Live fetch status
            live_fetch = fetch_station_live_data(sid, stn, force_refresh=False)
            fetch_status = live_fetch.get("status", "UNKNOWN")

            # Signal rule
            proc_files = sorted((PROJECT_ROOT / "data" / "processed").glob(f"{sid}_aligned_*.csv"))
            signal_res = "PASS"
            if proc_files:
                df_proc = pd.read_csv(proc_files[-1])
                sig_ok, sig_reason = evaluate_station_signal(df_proc, sid, thresh_cfg)
                signal_res = "PASS" if sig_ok else f"FAIL: {sig_reason}"
            else:
                signal_res = "FAIL: No processed data"

            # Forecast execution
            replay_origin = get_highest_flow_test_date(sid) if mode == "historical_replay" else None
            res = generate_forecast(
                station_id=sid,
                horizon=1,
                model_name="xgboost",
                mode=mode,
                replay_origin_date=replay_origin,
            )

            # Features outside same-season range
            outside_names = []
            if res.top_drifting_features:
                outside_names = [f["feature"] for f in res.top_drifting_features if f.get("is_outside")]
            outside_count = getattr(res, "features_outside_range_count", len(outside_names))
            outside_str = f"{outside_count} ({', '.join(outside_names)})" if outside_count > 0 else "0 (None)"

            data_age_str = f"{res.data_age_hours}h" if res.data_age_hours is not None else "N/A"

            results.append({
                "mode": mode,
                "station_id": sid,
                "station_name": sname,
                "model_exists": model_exists,
                "fetch_status": fetch_status,
                "last_observation": res.last_observation_time or "N/A",
                "data_age": data_age_str,
                "signal_rule": signal_res,
                "features_outside_same_season_range": outside_str,
                "drift_status": res.drift_status,
                "risk": res.risk.get("category", "UNAVAILABLE"),
                "final_status": res.status,
                "exact_reason": res.reason or "OK",
            })

    df_diag = pd.DataFrame(results)
    out_csv = PROJECT_ROOT / "reports" / "why_blocked.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_diag.to_csv(out_csv, index=False)
    print(f"Saved diagnosis to {out_csv}\n")

    cols_to_print = [
        "station_id", "mode", "model_exists", "fetch_status", "last_observation",
        "data_age", "signal_rule", "drift_status", "risk", "final_status", "exact_reason"
    ]
    print(df_diag[cols_to_print].to_string(index=False))

if __name__ == "__main__":
    main()
