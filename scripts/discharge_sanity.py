"""
scripts/discharge_sanity.py
───────────────────────────
Compute and print discharge sanity statistics:
mean, median, 95th percentile, and max of modelled discharge (m³/s),
plus nominal and snapped grid coordinates.
"""

from __future__ import annotations

import glob
from pathlib import Path
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def print_discharge_sanity() -> list[dict]:
    with open(_ROOT / "configs" / "data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stations = cfg.get("stations", [])
    records = []

    print("\n" + "=" * 115)
    print("CLIMATESHIELD AI — DISCHARGE SANITY & GRID COORDINATES AUDIT")
    print("=" * 115)
    print(f"{'Station ID':<20} | {'Lat':<7} | {'Lon':<7} | {'Mean (m³/s)':<12} | {'Median':<10} | {'95th Pct':<10} | {'Max (m³/s)':<12} | {'Status'}")
    print("-" * 115)

    for s in stations:
        sid = s["id"]
        lat = s["latitude"]
        lon = s["longitude"]

        raw_files = sorted((_ROOT / "data" / "raw").glob(f"{sid}_discharge_raw_*.csv"))
        if not raw_files:
            rec = {
                "station_id": sid, "latitude": lat, "longitude": lon,
                "mean": None, "median": None, "p95": None, "max": None,
                "status": "No raw data file",
            }
            records.append(rec)
            print(f"{sid:<20} | {lat:<7.4f} | {lon:<7.4f} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10} | {'N/A':<12} | No raw data file")
            continue

        df = pd.read_csv(raw_files[-1])
        s_disc = df["river_discharge"].dropna()

        if len(s_disc) == 0:
            rec = {
                "station_id": sid, "latitude": lat, "longitude": lon,
                "mean": None, "median": None, "p95": None, "max": None,
                "status": "All NaN (unmodelled channel)",
            }
            records.append(rec)
            print(f"{sid:<20} | {lat:<7.4f} | {lon:<7.4f} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10} | {'N/A':<12} | All NaN (unmodelled channel)")
        else:
            mean_q = float(s_disc.mean())
            med_q = float(s_disc.median())
            p95_q = float(s_disc.quantile(0.95))
            max_q = float(s_disc.max())
            rec = {
                "station_id": sid, "latitude": lat, "longitude": lon,
                "mean": round(mean_q, 3), "median": round(med_q, 3),
                "p95": round(p95_q, 3), "max": round(max_q, 3),
                "status": "Active channel",
            }
            records.append(rec)
            print(f"{sid:<20} | {lat:<7.4f} | {lon:<7.4f} | {mean_q:<12.3f} | {med_q:<10.3f} | {p95_q:<10.3f} | {max_q:<12.3f} | Active channel")

    print("=" * 115 + "\n")
    return records


if __name__ == "__main__":
    print_discharge_sanity()
