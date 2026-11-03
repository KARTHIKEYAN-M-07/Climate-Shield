"""Benchmark ingest-to-display latency across pipeline stages.

Stages measured:
1. fetch: Data ingest / mock API fetch time
2. features: Feature engineering (API, lags, rolling stats)
3. predict: Model inference time
4. render: Visualization / JSON payload formatting time

Writes reports/latency.json with median, p95, and per-stage breakdowns.
Only when this file exists can the term 'real-time' be used alongside its measured numbers.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.features.temporal_features import (
    add_antecedent_precipitation_index,
    add_days_since_heavy_rain,
    add_wet_spell_length,
    add_monsoon_flags,
)
from src.models.baselines import PersistenceBaseline


def benchmark_pipeline_latency(
    n_runs: int = 50,
    output_path: str = "reports/latency.json",
) -> Dict[str, Any]:
    """Run N benchmark cycles measuring each stage's execution time."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stage_times: Dict[str, List[float]] = {
        "fetch_ms": [],
        "features_ms": [],
        "predict_ms": [],
        "render_ms": [],
        "total_ms": [],
    }

    # Prepare benchmark dummy data
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    sample_df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "precipitation_sum": np.random.exponential(scale=10.0, size=180),
        "river_discharge": np.random.uniform(50.0, 500.0, size=180),
    })

    model = PersistenceBaseline(discharge_col="river_discharge")
    model.fit(sample_df)

    for _ in range(n_runs):
        t_start = time.perf_counter()

        # 1. Fetch simulation (reading data / memory payload)
        t0 = time.perf_counter()
        _ = sample_df.copy()
        time.sleep(0.001)  # tiny simulated network/disk slice
        t_fetch = (time.perf_counter() - t0) * 1000.0

        # 2. Features stage
        t0 = time.perf_counter()
        df_feat = add_antecedent_precipitation_index(sample_df)
        df_feat = add_days_since_heavy_rain(df_feat)
        df_feat = add_wet_spell_length(df_feat)
        df_feat = add_monsoon_flags(df_feat)
        t_features = (time.perf_counter() - t0) * 1000.0

        # 3. Predict stage
        t0 = time.perf_counter()
        preds = model.predict(df_feat)
        t_predict = (time.perf_counter() - t0) * 1000.0

        # 4. Render / serialization stage
        t0 = time.perf_counter()
        _ = {
            "dates": [str(d) for d in dates[-7:]],
            "predictions": preds[-7:].tolist(),
            "status": "ready",
        }
        t_render = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_start) * 1000.0

        stage_times["fetch_ms"].append(t_fetch)
        stage_times["features_ms"].append(t_features)
        stage_times["predict_ms"].append(t_predict)
        stage_times["render_ms"].append(t_render)
        stage_times["total_ms"].append(t_total)

    report = {
        "benchmark_metadata": {
            "n_runs": n_runs,
            "unit": "milliseconds (ms)",
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": {
            "median_total_ms": round(float(np.median(stage_times["total_ms"])), 2),
            "p95_total_ms": round(float(np.percentile(stage_times["total_ms"], 95)), 2),
            "min_total_ms": round(float(np.min(stage_times["total_ms"])), 2),
            "max_total_ms": round(float(np.max(stage_times["total_ms"])), 2),
        },
        "stage_breakdown": {
            "fetch": {
                "median_ms": round(float(np.median(stage_times["fetch_ms"])), 2),
                "p95_ms": round(float(np.percentile(stage_times["fetch_ms"], 95)), 2),
            },
            "features": {
                "median_ms": round(float(np.median(stage_times["features_ms"])), 2),
                "p95_ms": round(float(np.percentile(stage_times["features_ms"], 95)), 2),
            },
            "predict": {
                "median_ms": round(float(np.median(stage_times["predict_ms"])), 2),
                "p95_ms": round(float(np.percentile(stage_times["predict_ms"], 95)), 2),
            },
            "render": {
                "median_ms": round(float(np.median(stage_times["render_ms"])), 2),
                "p95_ms": round(float(np.percentile(stage_times["render_ms"], 95)), 2),
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = benchmark_pipeline_latency()
    print(f"Latency benchmark written. Median total: {rep['summary']['median_total_ms']} ms, p95: {rep['summary']['p95_total_ms']} ms")
