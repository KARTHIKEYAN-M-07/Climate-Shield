import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import load_all_stations, load_thresholds_config

thresh_cfg = load_thresholds_config()
policy = thresh_cfg.get("station_signal_policy", {
    "min_rows": 500,
    "min_cv": 0.80,
    "min_p90_m3s": 0.20,
    "max_near_zero_share": 0.20,
})

min_rows = int(policy["min_rows"])
min_cv = float(policy["min_cv"])
min_p90 = float(policy["min_p90_m3s"])
max_zero_share = float(policy["max_near_zero_share"])

all_stns = load_all_stations()
rows = []

for s in all_stns:
    sid = s["id"]
    sname = s["name"]
    proc_files = sorted(Path("data/processed").glob(f"{sid}_aligned_*.csv"))
    if not proc_files:
        rows.append({
            "station": sid,
            "name": sname,
            "rows": 0,
            "train_mean": 0.0,
            "train_std": 0.0,
            "cv": 0.0,
            "p90": 0.0,
            "near_zero_pct": 100.0,
            "status": "Insufficient signal",
            "decision_reason": f"rows (0) < {min_rows}, unmodeled coastal grid",
        })
        continue

    df = pd.read_csv(proc_files[-1])
    n_rows = len(df)
    split_idx = int(n_rows * 0.70)
    train_df = df.iloc[:split_idx]

    if "river_discharge" not in train_df.columns or train_df["river_discharge"].dropna().empty:
        rows.append({
            "station": sid,
            "name": sname,
            "rows": n_rows,
            "train_mean": 0.0,
            "train_std": 0.0,
            "cv": 0.0,
            "p90": 0.0,
            "near_zero_pct": 100.0,
            "status": "Insufficient signal",
            "decision_reason": "No valid discharge measurements in training set",
        })
        continue

    q = train_df["river_discharge"].dropna()
    mean_val = float(q.mean())
    std_val = float(q.std(ddof=1))
    cv_val = float(std_val / mean_val) if mean_val > 1e-4 else 0.0
    p90_val = float(np.percentile(q, 90))
    near_zero_share = float((q <= 0.01).mean())

    reasons = []
    if n_rows < min_rows:
        reasons.append(f"rows ({n_rows}) < {min_rows}")
    if p90_val < min_p90:
        reasons.append(f"p90 ({p90_val:.2f}) < {min_p90:.2f} m3/s")
    if cv_val < min_cv:
        reasons.append(f"CV ({cv_val:.2f}) < {min_cv:.2f}")
    if near_zero_share > max_zero_share:
        reasons.append(f"near-zero ({near_zero_share*100:.1f}%) > {max_zero_share*100:.0f}%")

    if reasons:
        status = "Insufficient signal"
        dec_reason = "; ".join(reasons)
    else:
        status = "Sufficient signal"
        dec_reason = "Passes all criteria"

    rows.append({
        "station": sid,
        "name": sname,
        "rows": n_rows,
        "train_mean": round(mean_val, 3),
        "train_std": round(std_val, 3),
        "cv": round(cv_val, 2),
        "p90": round(p90_val, 3),
        "near_zero_pct": round(near_zero_share * 100, 1),
        "status": status,
        "decision_reason": dec_reason,
    })

df_res = pd.DataFrame(rows)
df_res.to_csv("reports/station_signal_audit.csv", index=False)
print(df_res.to_string(index=False))
