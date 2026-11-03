import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.live import fetch_station_live_data
from src.features.temporal_features import build_feature_matrix, get_feature_columns
from src.utils.config import load_all_stations

proc_files = sorted(Path("data/processed").glob("TN_CAU_TRICHY_aligned_*.csv"))
if not proc_files:
    raise FileNotFoundError("No processed files found for Trichy.")

df = pd.read_csv(proc_files[-1])
split_idx = int(len(df) * 0.70)
train_df = df.iloc[:split_idx]

train_feat = build_feature_matrix(train_df, horizons=[1]).dropna(subset=["target_discharge_h1d"])
feature_cols = get_feature_columns(train_feat)

train_stats = {}
for col in feature_cols:
    if col in train_feat.columns:
        vals = train_feat[col].dropna().astype(float)
        q = np.percentile(vals, np.linspace(0, 100, 11))
        train_stats[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "bins": [round(float(x), 4) for x in q.tolist()],
        }

stn = next(s for s in load_all_stations() if s["id"] == "TN_CAU_TRICHY")
live_res = fetch_station_live_data("TN_CAU_TRICHY", stn)
live_matrix = build_feature_matrix(live_res["df"], horizons=[1])
live_row = live_matrix.iloc[[-1]]

rows = []
for col in feature_cols:
    if col in live_row.columns and col in train_stats:
        l_val = float(live_row[col].iloc[0]) if pd.notna(live_row[col].iloc[0]) else np.nan
        t_m = train_stats[col]["mean"]
        t_s = train_stats[col]["std"]
        z = (l_val - t_m) / (t_s + 1e-6) if pd.notna(l_val) else np.nan

        bins = train_stats[col]["bins"]
        q_edges = np.array(bins)
        if len(np.unique(q_edges)) == len(q_edges):
            # 1-row histogram
            serving_cnt, _ = np.histogram([l_val], bins=q_edges)
            p = (serving_cnt + 1e-4) / (1 + 10 * 1e-4)
            q_ref = np.full(10, 0.1)
            psi_1row = np.sum((p - q_ref) * np.log(p / q_ref))
        else:
            psi_1row = np.nan

        rows.append({
            "feature": col,
            "live_value": round(l_val, 4),
            "train_mean": round(t_m, 4),
            "train_std": round(t_s, 4),
            "train_min": round(train_stats[col]["min"], 4),
            "train_max": round(train_stats[col]["max"], 4),
            "z_score": round(z, 2) if pd.notna(z) else None,
            "psi_1row": round(psi_1row, 3) if pd.notna(psi_1row) else None,
            "bins": bins,
        })

df_res = pd.DataFrame(rows)
df_res["abs_z"] = df_res["z_score"].abs()
df_res = df_res.sort_values(by="abs_z", ascending=False)
Path("reports").mkdir(exist_ok=True)
df_res.to_csv("reports/trichy_drift_audit.csv", index=False)

print("Top 15 drifting features by |z|:")
print(df_res[["feature", "live_value", "train_mean", "train_std", "train_min", "train_max", "z_score", "psi_1row"]].head(15).to_string())
print("\nTop 5 feature bins:")
for _, r in df_res.head(5).iterrows():
    print(f"{r['feature']}: bins={r['bins']}")
