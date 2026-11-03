"""
scripts/generate_training_coverage.py
─────────────────────────────────────
Builds the complete coverage matrix of:
station x horizon (1, 3, 7) x model (persistence, ridge, decision_tree, random_forest, gradient_boosting, xgboost, lstm)
Reports status (trained / failed / skipped / insufficient data) and artifact existence.
Saves to reports/training_coverage.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

def build_coverage():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stations = [s["id"] for s in cfg.get("stations", [])]
    horizons = [1, 3, 7]
    models = [
        "persistence",
        "ridge",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "hist_gradient_boosting",
        "elastic_net",
        "extra_trees",
        "lstm",
    ]

    # Load existing scores if available
    scores_file = REPORTS_DIR / "model_scores_all.csv"
    scores_set = set()
    if scores_file.exists():
        df_scores = pd.read_csv(scores_file)
        stn_col = "station" if "station" in df_scores.columns else "station_id"
        h_col = "horizon" if "horizon" in df_scores.columns else "horizon_days"
        for _, row in df_scores.iterrows():
            scores_set.add((str(row[stn_col]), int(row[h_col]), str(row["model"])))

    # Check LSTM availability
    try:
        import tensorflow as tf
        has_lstm = True
    except ImportError:
        has_lstm = False

    rows = []
    for stn in stations:
        stn_dir = MODELS_DIR / stn
        for h in horizons:
            for m in models:
                ext = ".keras" if m == "lstm" else ".joblib"
                expected_artifact_rel = f"models/{stn}/{m}_h{h}d{ext}"
                p1 = stn_dir / f"{m}_h{h}d{ext}"
                p2 = stn_dir / f"{stn}_{m}_h{h}d{ext}"
                p3 = MODELS_DIR / f"{stn}_{m}_h{h}d{ext}"

                artifact_exists = p1.exists() or p2.exists() or p3.exists()

                # Determine status
                if stn == "TN_ADY_CHENNAI":
                    status = "insufficient data"
                    notes = "Discharge has 0 valid readings (100% missing in raw GloFAS data)"
                elif m == "lstm":
                    if not has_lstm:
                        status = "skipped"
                        notes = "TensorFlow/Keras not installed (ModuleNotFoundError)"
                    else:
                        status = "skipped"
                        notes = "LSTM optional, requires approval / insufficient rows"
                else:
                    if artifact_exists and (stn, h, m) in scores_set:
                        status = "trained"
                        notes = "Trained and evaluated successfully"
                    elif artifact_exists:
                        status = "trained"
                        notes = "Artifact exists but score missing from summary CSV"
                    else:
                        status = "failed"
                        notes = "Artifact file missing"

                rows.append({
                    "station": stn,
                    "horizon": h,
                    "model": m,
                    "status": status,
                    "artifact_file": expected_artifact_rel,
                    "artifact_exists": artifact_exists,
                    "notes": notes,
                })

    coverage_df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "training_coverage.csv"
    coverage_df.to_csv(out_csv, index=False)
    print(f"Coverage matrix written to {out_csv} ({len(coverage_df)} rows)")

    # Print summary breakdown
    print("\nCoverage Status Summary:")
    print(coverage_df["status"].value_counts().to_string())

    print("\nArtifact Exists Summary:")
    print(coverage_df["artifact_exists"].value_counts().to_string())

    return coverage_df

if __name__ == "__main__":
    build_coverage()
