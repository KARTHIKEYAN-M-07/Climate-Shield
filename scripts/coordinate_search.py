"""scripts/coordinate_search.py
─────────────────────────────
Coordinate Grid Search for Stations Failing Hydrologic Signal Rule.

Evaluates Flood API discharge on a 7x7 grid (0.05° spacing) around
nominal coordinates for the TRAINING PERIOD ONLY (2014-01-01 to 2022-12-22).

Rules (defined in configs/thresholds.yaml before running):
  - Must pass existing signal rule: rows >= 500, CV >= 0.80, p90 >= 0.20 m³/s, near_zero <= 20%
  - Must lie within max_distance_km (25.0 km)
  - Ranked strictly by training-period signal score (CV * p90)
  - Saves to reports/coordinate_search_<station_id>.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests

from src.utils.config import load_all_stations, load_thresholds_config


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)


def evaluate_signal_metrics(
    values: List[float | None],
    thresh_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate hydrologic signal rule metrics strictly on training series."""
    sig_policy = thresh_cfg.get("station_signal_policy", {
        "min_rows": 500,
        "min_cv": 0.80,
        "min_p90_m3s": 0.20,
        "max_near_zero_share": 0.20,
    })

    clean = [v for v in values if v is not None and np.isfinite(v)]
    n_rows = len(clean)

    if n_rows < sig_policy.get("min_rows", 500):
        return {
            "n_rows": n_rows,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "cv": 0.0,
            "zero_share": 1.0,
            "signal_pass": False,
            "signal_reason": f"Insufficient records ({n_rows} < {sig_policy.get('min_rows', 500)})",
            "score": 0.0,
        }

    arr = np.array(clean, dtype=float)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    p90_val = float(np.percentile(arr, 90))
    max_val = float(np.max(arr))
    std_val = float(np.std(arr))
    cv_val = float(std_val / mean_val) if mean_val > 1e-6 else 0.0
    zero_share = float(np.mean(arr <= 0.01))

    reasons = []
    if cv_val < sig_policy.get("min_cv", 0.80):
        reasons.append(f"CV={cv_val:.2f} < {sig_policy.get('min_cv', 0.80):.2f}")
    if p90_val < sig_policy.get("min_p90_m3s", 0.20):
        reasons.append(f"p90={p90_val:.2f} < {sig_policy.get('min_p90_m3s', 0.20):.2f} m³/s")
    if zero_share > sig_policy.get("max_near_zero_share", 0.20):
        reasons.append(f"dry_bed={zero_share*100:.1f}% > {sig_policy.get('max_near_zero_share', 0.20)*100:.0f}%")

    is_pass = len(reasons) == 0
    reason_str = "PASS" if is_pass else f"FAIL: {'; '.join(reasons)}"
    score = float(cv_val * p90_val) if is_pass else 0.0

    return {
        "n_rows": n_rows,
        "mean": round(mean_val, 3),
        "median": round(median_val, 3),
        "p90": round(p90_val, 3),
        "max": round(max_val, 3),
        "cv": round(cv_val, 3),
        "zero_share": round(zero_share, 4),
        "signal_pass": is_pass,
        "signal_reason": reason_str,
        "score": round(score, 3),
    }


def fetch_grid_batch(
    coords: List[Tuple[float, float]],
    start_date: str,
    end_date: str,
    batch_size: int = 15,
) -> Dict[Tuple[float, float], List[float | None]]:
    """Fetch Flood API discharge series for a list of (lat, lon) pairs."""
    url = "https://flood-api.open-meteo.com/v1/flood"
    results: Dict[Tuple[float, float], List[float | None]] = {}

    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        lats_str = ",".join(f"{c[0]:.4f}" for c in chunk)
        lons_str = ",".join(f"{c[1]:.4f}" for c in chunk)

        params = {
            "latitude": lats_str,
            "longitude": lons_str,
            "daily": "river_discharge",
            "start_date": start_date,
            "end_date": end_date,
        }

        retries = 3
        backoff = 2
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=25)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        for c, item in zip(chunk, data):
                            results[c] = item.get("daily", {}).get("river_discharge", [])
                    elif isinstance(data, dict):
                        # Single location returned
                        results[chunk[0]] = data.get("daily", {}).get("river_discharge", [])
                    break
                else:
                    print(f"  Warning: HTTP {r.status_code} for chunk {i//batch_size + 1}")
                    time.sleep(backoff)
                    backoff *= 2
            except Exception as exc:
                print(f"  Attempt {attempt + 1} failed: {exc}")
                time.sleep(backoff)
                backoff *= 2
        time.sleep(1.0)  # Gentle rate limiting

    return results


def run_coordinate_search_for_station(
    station: Dict[str, Any],
    thresh_cfg: Dict[str, Any],
    reports_dir: Path,
) -> pd.DataFrame:
    """Run 7x7 grid coordinate search for a single station."""
    sid = station["id"]
    sname = station["name"]
    nom_lat = station["latitude"]
    nom_lon = station["longitude"]

    coord_policy = thresh_cfg.get("coordinate_search_policy", {})
    grid_size = int(coord_policy.get("grid_size", 7))
    step_deg = float(coord_policy.get("step_degrees", 0.05))
    max_dist_km = float(coord_policy.get("max_distance_km", 25.0))
    start_date = str(coord_policy.get("training_start_date", "2014-01-01"))
    end_date = str(coord_policy.get("training_end_date", "2022-12-22"))

    print(f"\n==========================================================================")
    print(f"Searching {grid_size}x{grid_size} grid around {sid} ({sname})")
    print(f"Nominal: ({nom_lat:.4f}, {nom_lon:.4f}) | Training split: {start_date} to {end_date}")
    print(f"==========================================================================")

    half = grid_size // 2
    offsets = [i * step_deg for i in range(-half, half + 1)]

    candidates: List[Tuple[float, float]] = []
    for d_lat in offsets:
        for d_lon in offsets:
            c_lat = round(nom_lat + d_lat, 4)
            c_lon = round(nom_lon + d_lon, 4)
            candidates.append((c_lat, c_lon))

    print(f"Fetching {len(candidates)} candidate grid points from Open-Meteo Flood API...")
    fetched_data = fetch_grid_batch(candidates, start_date=start_date, end_date=end_date)

    rows = []
    for c_lat, c_lon in candidates:
        dist_km = haversine_km(nom_lat, nom_lon, c_lat, c_lon)
        within_max = dist_km <= max_dist_km
        q_vals = fetched_data.get((c_lat, c_lon), [])

        metrics = evaluate_signal_metrics(q_vals, thresh_cfg)

        # Candidate viability
        passes_dist = within_max
        passes_sig = metrics["signal_pass"]
        is_viable = passes_dist and passes_sig

        row = {
            "station_id": sid,
            "station_name": sname,
            "lat": c_lat,
            "lon": c_lon,
            "distance_km": round(dist_km, 2),
            "is_nominal": (c_lat == round(nom_lat, 4) and c_lon == round(nom_lon, 4)),
            "within_max_dist": passes_dist,
            "n_training_days": metrics["n_rows"],
            "mean_m3s": metrics["mean"],
            "median_m3s": metrics["median"],
            "p90_m3s": metrics["p90"],
            "max_m3s": metrics["max"],
            "cv": metrics["cv"],
            "zero_share": metrics["zero_share"],
            "signal_rule": metrics["signal_reason"],
            "signal_pass": metrics["signal_pass"],
            "raw_score": metrics["score"],
            "score": round(metrics["score"] / (1.0 + 0.05 * dist_km), 3) if is_viable else 0.0,
            "is_viable": is_viable,
        }
        rows.append(row)

    df_res = pd.DataFrame(rows)

    # Sort: viable candidates first by score descending, then by distance ascending
    df_res = df_res.sort_values(
        by=["is_viable", "score", "distance_km"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    out_file = reports_dir / f"coordinate_search_{sid}.csv"
    df_res.to_csv(out_file, index=False)
    print(f"Saved {len(df_res)} candidates to {out_file}")

    # Print top 5 candidates
    display_cols = [
        "lat", "lon", "distance_km", "is_nominal", "mean_m3s", "p90_m3s", "cv", "zero_share", "signal_rule", "score"
    ]
    print("\nTop Candidates:")
    print(df_res[display_cols].head(8).to_string(index=False))

    return df_res


def main():
    stations = load_all_stations()
    thresh_cfg = load_thresholds_config()
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    target_sids = ["TN_CAU_METTUR", "TN_ADY_CHENNAI", "TN_BHA_BHAVANI", "TN_CAU_MUSIRI"]
    target_stations = [s for s in stations if s["id"] in target_sids]

    summary_rows = []

    for stn in target_stations:
        df_stn = run_coordinate_search_for_station(stn, thresh_cfg, reports_dir)
        nom_row = df_stn[df_stn["is_nominal"]].iloc[0] if not df_stn[df_stn["is_nominal"]].empty else None
        viable = df_stn[df_stn["is_viable"]]

        if not viable.empty:
            chosen = viable.iloc[0]
            status = "FOUND_VIABLE_CHANNEL"
            chosen_lat = chosen["lat"]
            chosen_lon = chosen["lon"]
            dist_moved = chosen["distance_km"]
            chosen_p90 = chosen["p90_m3s"]
            chosen_cv = chosen["cv"]
            chosen_mean = chosen["mean_m3s"]
        else:
            status = "NO_VIABLE_CHANNEL_WITHIN_RANGE"
            chosen_lat = stn["latitude"]
            chosen_lon = stn["longitude"]
            dist_moved = 0.0
            chosen_p90 = nom_row["p90_m3s"] if nom_row is not None else 0.0
            chosen_cv = nom_row["cv"] if nom_row is not None else 0.0
            chosen_mean = nom_row["mean_m3s"] if nom_row is not None else 0.0

        summary_rows.append({
            "station_id": stn["id"],
            "station_name": stn["name"],
            "status": status,
            "nominal_lat": stn["latitude"],
            "nominal_lon": stn["longitude"],
            "chosen_lat": chosen_lat,
            "chosen_lon": chosen_lon,
            "distance_moved_km": dist_moved,
            "training_mean_m3s": chosen_mean,
            "training_p90_m3s": chosen_p90,
            "training_cv": chosen_cv,
            "viable_candidate_count": len(viable),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = reports_dir / "coordinate_search_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 110)
    print("COORDINATE SEARCH SUMMARY TABLE")
    print("=" * 110)
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
