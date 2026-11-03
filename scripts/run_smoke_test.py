"""scripts/run_smoke_test.py
─────────────────────────
Smoke test runner across all stations in Live and Historical Replay modes:
1. Runs generate_forecast for Live mode (all stations × horizons × models).
2. Runs generate_forecast for Historical Replay mode (origin in test set).
3. Records: station | features outside range | top drifting feature (live value vs range) | drift status | risk | status | reason.
4. Saves to reports/forecast_smoke_test.csv.
5. Prints comprehensive tables for Live mode and Historical Replay mode.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from src.forecast.engine import generate_forecast, get_highest_flow_test_date
from src.utils.config import load_all_stations

MODELS = [
    "persistence",
    "ridge",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "elastic_net",
    "extra_trees",
]

HORIZONS = [1, 3, 7]


def main():
    stations = load_all_stations()
    results = []

    print(f"Running smoke test across {len(stations)} stations × {len(HORIZONS)} horizons × {len(MODELS)} models in LIVE and REPLAY modes...")
    t0 = time.time()

    for mode in ["live", "historical_replay"]:
        for stn in stations:
            sid = stn["id"]
            sname = stn["name"]
            
            # Select replay origin date from test period
            replay_date = get_highest_flow_test_date(sid) if mode == "historical_replay" else None

            for h in HORIZONS:
                for m in MODELS:
                    try:
                        res = generate_forecast(
                            station_id=sid,
                            horizon=h,
                            model_name=m,
                            mode=mode,
                            replay_origin_date=replay_date,
                        )
                        err_type = "None" if res.status in ("OK", "STALE") else (res.risk.get("status") or res.status)
                        msg = res.reason or (res.warnings[0] if res.warnings else "Forecast OK")
                        
                        top = res.top_drifting_features[0] if (res.top_drifting_features and len(res.top_drifting_features) > 0) else None
                        if top:
                            top_driver_str = f"{top['feature']} (live={top.get('live_value', 'N/A')} vs {top.get('range_str', '[]')})"
                        else:
                            top_driver_str = "None"

                        results.append({
                            "mode": mode,
                            "station_id": sid,
                            "station_name": sname,
                            "horizon": h,
                            "model": m,
                            "origin_date": res.last_observation_time or "N/A",
                            "features_outside_range": getattr(res, "features_outside_range_count", 0),
                            "top_drifting_feature": top_driver_str,
                            "drift_status": res.drift_status,
                            "risk": res.risk.get("category", "UNAVAILABLE"),
                            "status": res.status,
                            "forecast": res.forecast,
                            "interval_lower": res.interval[0] if res.interval else None,
                            "interval_upper": res.interval[1] if res.interval else None,
                            "error_type": err_type,
                            "reason": msg,
                            "message": msg,
                        })
                    except Exception as exc:
                        results.append({
                            "mode": mode,
                            "station_id": sid,
                            "station_name": sname,
                            "horizon": h,
                            "model": m,
                            "origin_date": "N/A",
                            "features_outside_range": 0,
                            "top_drifting_feature": "None",
                            "drift_status": "UNKNOWN",
                            "risk": "UNAVAILABLE",
                            "status": "ERROR",
                            "forecast": None,
                            "interval_lower": None,
                            "interval_upper": None,
                            "error_type": type(exc).__name__,
                            "reason": str(exc),
                            "message": str(exc),
                        })

    df = pd.DataFrame(results)
    out_csv = _PROJECT_ROOT / "reports" / "forecast_smoke_test.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    print(f"Smoke test complete in {elapsed:.2f}s. Saved {len(df)} records to {out_csv}.\n")

    # 1. LIVE MODE SUMMARY TABLE (Horizon 1, XGBoost / Default best model)
    print("=" * 175)
    print("LIVE MODE SMOKE TEST SUMMARY (Horizon 1, Default Model)")
    print("=" * 175)
    print(f"{'Station':<24} | {'Outside':<7} | {'Top Drifting Feature (Live vs Range)':<60} | {'Drift Status':<19} | {'Risk':<12} | {'Status':<16} | Reason")
    print("-" * 175)
    df_live_h1 = df[(df["mode"] == "live") & (df["horizon"] == 1) & (df["model"] == "xgboost")]
    for _, r in df_live_h1.iterrows():
        driver = str(r["top_drifting_feature"])[:60]
        reason_str = str(r["reason"])[:45]
        print(f"{r['station_name'][:24]:<24} | {r['features_outside_range']:<7} | {driver:<60} | {str(r['drift_status']):<19} | {str(r['risk']):<12} | {str(r['status']):<16} | {reason_str}")
    print("=" * 175)

    # 2. HISTORICAL REPLAY MODE SUMMARY TABLE (Horizon 1, XGBoost / Default best model)
    print("\n" + "=" * 175)
    print("HISTORICAL REPLAY MODE SMOKE TEST SUMMARY (Test Set Origin, Horizon 1, Default Model)")
    print("=" * 175)
    print(f"{'Station':<24} | {'Outside':<7} | {'Top Drifting Feature (Live vs Range)':<60} | {'Drift Status':<19} | {'Risk':<12} | {'Status':<16} | Reason")
    print("-" * 175)
    df_rep_h1 = df[(df["mode"] == "historical_replay") & (df["horizon"] == 1) & (df["model"] == "xgboost")]
    for _, r in df_rep_h1.iterrows():
        driver = str(r["top_drifting_feature"])[:60]
        reason_str = str(r["reason"])[:45]
        print(f"{r['station_name'][:24]:<24} | {r['features_outside_range']:<7} | {driver:<60} | {str(r['drift_status']):<19} | {str(r['risk']):<12} | {str(r['status']):<16} | {reason_str}")
    print("=" * 175)


if __name__ == "__main__":
    main()
