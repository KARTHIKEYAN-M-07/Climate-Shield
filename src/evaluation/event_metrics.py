"""
src/evaluation/event_metrics.py
───────────────────────────────
Multi-threshold hydrologic event metrics:
  - Event thresholds: 90th, 95th, and 99th percentiles of the TRAINING set only.
  - Contingency metrics: Hits (TP), False Alarms (FP), Misses (FN), Correct Negatives (TN)
  - POD (Probability of Detection / Hit Rate): TP / (TP + FN)
  - FAR (False Alarm Ratio): FP / (TP + FP)
  - CSI (Critical Success Index / Threat Score): TP / (TP + FP + FN)
  - Missed Event Rate: FN / (TP + FN)
  - False Alarm Rate (False Positive Rate): FP / (FP + TN)
  - Peak Flow Error (%) and Peak Timing Error (days)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def peak_flow_error(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Calculate relative percentage error on peak flow: (pred_peak - true_peak) / true_peak * 100."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")
    true_peak = float(np.max(yt))
    pred_peak = float(np.max(yp))
    if true_peak == 0.0:
        return 0.0 if pred_peak == 0.0 else float("inf")
    return float(((pred_peak - true_peak) / true_peak) * 100.0)


def peak_timing_error(
    dates: list[str] | pd.Series | np.ndarray,
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> int:
    """Difference in days between predicted peak date and observed peak date."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    dt = pd.to_datetime(pd.Series(dates))
    if len(yt) == 0 or len(dt) == 0:
        return 0
    true_peak_idx = int(np.argmax(yt))
    pred_peak_idx = int(np.argmax(yp))
    date_true = dt.iloc[true_peak_idx]
    date_pred = dt.iloc[pred_peak_idx]
    diff_days = (date_pred - date_true).days
    return int(diff_days)


def contingency_table(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float,
) -> dict[str, int]:
    """Calculate contingency table counts (TP, FP, FN, TN) for y >= threshold."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    actual_event = yt >= threshold
    pred_event = yp >= threshold

    tp = int(np.sum(actual_event & pred_event))
    fp = int(np.sum((~actual_event) & pred_event))
    fn = int(np.sum(actual_event & (~pred_event)))
    tn = int(np.sum((~actual_event) & (~pred_event)))

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def missed_event_rate(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float,
) -> float:
    """Missed-event rate: FN / (TP + FN)."""
    tbl = contingency_table(y_true, y_pred, threshold)
    actual_positives = tbl["TP"] + tbl["FN"]
    if actual_positives == 0:
        return 0.0
    return float(tbl["FN"] / actual_positives)


def false_alarm_rate(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float,
) -> float:
    """False-alarm rate: FP / (FP + TN)."""
    tbl = contingency_table(y_true, y_pred, threshold)
    actual_negatives = tbl["FP"] + tbl["TN"]
    if actual_negatives == 0:
        return 0.0
    return float(tbl["FP"] / actual_negatives)


def compute_multi_threshold_event_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    thresholds_dict: dict[str, float],
    dates: list[str] | pd.Series | np.ndarray | None = None,
    horizon_days: int = 1,
) -> dict[str, Any]:
    """Compute event metrics across multiple training percentiles (e.g. P90, P95, P99).

    Parameters
    ----------
    thresholds_dict: e.g. {"p90": 120.5, "p95": 210.0, "p99": 350.2} (from train only).
    """
    res: dict[str, Any] = {
        "peak_flow_error_pct": peak_flow_error(y_true, y_pred),
        "peak_timing_error_days": peak_timing_error(dates, y_true, y_pred) if dates is not None else 0,
        "warning_lead_time_days": int(horizon_days),
        "threshold_metrics": {},
    }

    for label, thresh in thresholds_dict.items():
        tbl = contingency_table(y_true, y_pred, thresh)
        tp, fp, fn, tn = tbl["TP"], tbl["FP"], tbl["FN"], tbl["TN"]

        pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
        csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        missed_rate = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
        false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        res["threshold_metrics"][label] = {
            "threshold_m3s": round(float(thresh), 2),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "POD": round(pod, 4),
            "FAR": round(far, 4),
            "CSI": round(csi, 4),
            "missed_event_rate": round(missed_rate, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
        }

    # Backward compatibility with v1 single-threshold keys (using p95)
    p95_info = res["threshold_metrics"].get("p95", list(res["threshold_metrics"].values())[0])
    res["event_threshold"] = p95_info["threshold_m3s"]
    res["missed_event_rate"] = p95_info["missed_event_rate"]
    res["false_alarm_rate"] = p95_info["false_alarm_rate"]
    res["POD"] = p95_info["POD"]
    res["FAR"] = p95_info["FAR"]
    res["CSI"] = p95_info["CSI"]
    res["true_positives"] = p95_info["TP"]
    res["false_positives"] = p95_info["FP"]
    res["false_negatives"] = p95_info["FN"]
    res["true_negatives"] = p95_info["TN"]

    return res


def compute_event_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float,
    dates: list[str] | pd.Series | np.ndarray | None = None,
    horizon_days: int = 1,
) -> dict[str, Any]:
    """Single threshold wrapper maintaining backwards compatibility."""
    return compute_multi_threshold_event_metrics(
        y_true=y_true,
        y_pred=y_pred,
        thresholds_dict={"p95": threshold},
        dates=dates,
        horizon_days=horizon_days,
    )
