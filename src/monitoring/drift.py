"""Drift monitoring using Population Stability Index (PSI) and Kolmogorov-Smirnov test.

Detects feature drift between reference (training) and target (serving/current) data distributions.
Also flags stale data and model age warnings.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy import stats


def compute_psi(
    expected: Union[List[float], np.ndarray],
    actual: Union[List[float], np.ndarray],
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) between expected and actual distributions.

    Rules of thumb:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate drift
    - PSI >= 0.2: Significant drift
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Bucket breakpoints based on quantiles of expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    breakpoints = np.percentile(expected, percentiles)
    breakpoints[0] -= 1e-5
    breakpoints[-1] += 1e-5

    # Count occurrences in each bucket
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to fractions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid zero division with epsilon smoothing
    expected_pct = np.clip(expected_pct, epsilon, 1.0)
    actual_pct = np.clip(actual_pct, epsilon, 1.0)

    # Re-normalize
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    # PSI calculation
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def compute_ks_test(
    expected: Union[List[float], np.ndarray],
    actual: Union[List[float], np.ndarray],
) -> Dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test comparing two distributions."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    res = stats.ks_2samp(expected, actual)
    p_val = float(res.pvalue)
    return {
        "statistic": round(float(res.statistic), 4),
        "p_value": round(p_val, 6),
        "drift_detected": bool(p_val < 0.05),
    }


def check_data_and_model_freshness(
    latest_data_timestamp: str,
    model_trained_timestamp: str,
    max_data_stale_hours: float = 48.0,
    max_model_age_days: float = 30.0,
) -> Dict[str, Any]:
    """Check whether incoming data is stale and whether the model is overdue for retraining."""
    now = datetime.now(timezone.utc)

    # Parse timestamps
    def _parse(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    data_dt = _parse(latest_data_timestamp)
    model_dt = _parse(model_trained_timestamp)

    data_age_hours = (now - data_dt).total_seconds() / 3600.0
    model_age_days = (now - model_dt).total_seconds() / 86400.0

    is_data_stale = data_age_hours > max_data_stale_hours
    is_model_stale = model_age_days > max_model_age_days

    return {
        "data_age_hours": round(data_age_hours, 2),
        "model_age_days": round(model_age_days, 2),
        "is_data_stale": is_data_stale,
        "is_model_stale": is_model_stale,
        "warnings": [
            *(["Data is stale (> 48h old)"] if is_data_stale else []),
            *(["Model is stale (> 30 days since retrain)"] if is_model_stale else []),
        ],
    }


def compute_drift_status(
    train_values: Optional[Union[List[float], np.ndarray]] = None,
    serving_values: Optional[Union[List[float], np.ndarray]] = None,
    feature_name: str = "river_discharge",
    reference_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare recent serving feature distribution with training distribution via PSI and z-score.

    Parameters
    ----------
    train_values:     Optional array of historical training feature observations.
    serving_values:   Array of recent serving/operational feature observations.
    feature_name:     Name of monitored feature (default: 'river_discharge').
    reference_stats:  Pre-computed reference summary stats (mean, std, quantiles) from model metadata.

    Returns
    -------
    dict with keys guaranteed to include:
      - drift_status: 'IN_DISTRIBUTION' | 'OUT_OF_DISTRIBUTION' | 'UNKNOWN'
      - status: 'In-distribution' | 'Out-of-distribution' | 'Moderate shift' | 'Unknown'
      - is_drifted: bool
      - psi: float or None
      - z_score: float or None
      - drift_value: float or None
      - rule: str
      - warnings: list[str]
    """
    warnings: List[str] = []

    if serving_values is None:
        return {
            "feature_name": feature_name,
            "drift_status": "UNKNOWN",
            "status": "Unknown",
            "is_drifted": False,
            "psi": None,
            "z_score": None,
            "drift_value": None,
            "rule": "No serving values provided",
            "warnings": ["Serving values missing"],
            "train_mean": None,
            "serving_mean": None,
        }

    sv = np.asarray(serving_values, dtype=float)
    sv = sv[~np.isnan(sv)]
    if len(sv) == 0:
        return {
            "feature_name": feature_name,
            "drift_status": "UNKNOWN",
            "status": "Unknown",
            "is_drifted": False,
            "psi": None,
            "z_score": None,
            "drift_value": None,
            "rule": "Empty serving values",
            "warnings": ["Serving values are empty or all NaN"],
            "train_mean": None,
            "serving_mean": None,
        }

    mean_sv = float(np.mean(sv))

    # Case 1: reference_stats supplied
    if reference_stats and "mean" in reference_stats and "std" in reference_stats:
        mean_tr = float(reference_stats["mean"])
        std_tr = float(reference_stats["std"]) if float(reference_stats.get("std", 1.0)) > 1e-6 else 1.0
        z_score = float((mean_sv - mean_tr) / std_tr)

        # PSI computation if quantiles/bins are stored and serving sample is sufficient (n >= 10)
        if len(sv) >= 10 and "quantiles_10bin" in reference_stats and len(reference_stats["quantiles_10bin"]) == 11:
            breakpoints = np.array(reference_stats["quantiles_10bin"], dtype=float)
            breakpoints[0] -= 1e-5
            breakpoints[-1] += 1e-5
            counts, _ = np.histogram(sv, bins=breakpoints)
            actual_pct = counts / len(sv)
            expected_pct = np.full(10, 0.10)
            actual_pct = np.clip(actual_pct, 1e-4, 1.0)
            actual_pct /= actual_pct.sum()
            psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            psi = float(np.sum(psi_values))
        elif train_values is not None and len(train_values) >= 10 and len(sv) >= 10:
            tr = np.asarray(train_values, dtype=float)
            tr = tr[~np.isnan(tr)]
            psi = compute_psi(tr, sv) if len(tr) >= 10 else 0.0
        else:
            psi = 0.0

    # Case 2: raw train_values supplied
    elif train_values is not None:
        tr = np.asarray(train_values, dtype=float)
        tr = tr[~np.isnan(tr)]
        if len(tr) == 0:
            return {
                "feature_name": feature_name,
                "drift_status": "UNKNOWN",
                "status": "Unknown",
                "is_drifted": False,
                "psi": None,
                "z_score": None,
                "drift_value": None,
                "rule": "Insufficient reference data to compute drift",
                "warnings": ["Drift reference statistics and training data missing"],
                "train_mean": None,
                "serving_mean": round(mean_sv, 2),
            }
        psi = compute_psi(tr, sv)
        mean_tr = float(np.mean(tr))
        std_tr = float(np.std(tr)) if np.std(tr) > 1e-6 else 1.0
        z_score = float((mean_sv - mean_tr) / std_tr)

    # Case 3: neither supplied
    else:
        return {
            "feature_name": feature_name,
            "drift_status": "UNKNOWN",
            "status": "Unknown",
            "is_drifted": False,
            "psi": None,
            "z_score": None,
            "drift_value": None,
            "rule": "Drift reference missing",
            "warnings": ["Drift reference statistics not available for this model artifact"],
            "train_mean": None,
            "serving_mean": round(mean_sv, 2),
        }

    # Evaluate decision rule
    if psi < 0.1 and abs(z_score) < 2.5:
        status_label = "In-distribution"
        drift_code = "IN_DISTRIBUTION"
        is_drifted = False
    elif psi >= 0.2 or abs(z_score) >= 3.5:
        status_label = "Out-of-distribution"
        drift_code = "OUT_OF_DISTRIBUTION"
        is_drifted = True
    else:
        status_label = "Moderate shift"
        drift_code = "IN_DISTRIBUTION"
        is_drifted = False

    return {
        "feature_name": feature_name,
        "drift_status": drift_code,
        "status": status_label,
        "is_drifted": is_drifted,
        "psi": round(psi, 4),
        "z_score": round(z_score, 2),
        "drift_value": round(psi, 4),
        "rule": "In-distribution: PSI < 0.1 & |z| < 2.5 | Out-of-distribution: PSI >= 0.2 or |z| >= 3.5",
        "train_mean": round(mean_tr, 2),
        "serving_mean": round(mean_sv, 2),
        "warnings": warnings,
    }
