"""
src/evaluation/metrics.py
─────────────────────────
Standard and advanced hydrologic evaluation metrics:
  - MAE, RMSE, R²
  - NSE (Nash-Sutcliffe Efficiency) — NEVER clipped
  - KGE (Kling-Gupta Efficiency): KGE = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2)
  - PBIAS (Percent Bias): 100 * sum(y_pred - y_true) / sum(y_true)
  - log-NSE: NSE evaluated on ln(y + 1e-4) to measure low-flow recession accuracy
  - MSE Skill Score (SS_ref): 1 - (MSE_model / MSE_persistence)
  - Top 5% Relative Flow Error: Percentage error evaluated over top 5th percentile flows
  - Paired Diebold-Mariano test vs Persistence baseline with Harvey-Leybourne-Newbold correction
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_absolute_error(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def root_mean_squared_error(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2_score(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Calculate Coefficient of Determination (R²)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) < 2:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0.0:
        return 0.0 if ss_res == 0.0 else -float("inf")
    return float(1.0 - (ss_res / ss_tot))


def nash_sutcliffe_efficiency(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Calculate Nash-Sutcliffe Efficiency (NSE).

    Formula:
        NSE = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))

    Rule: Never clip negative values.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) < 2:
        return float("nan")
    numerator = np.sum((yt - yp) ** 2)
    denominator = np.sum((yt - np.mean(yt)) ** 2)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else -float("inf")
    return float(1.0 - (numerator / denominator))


def kling_gupta_efficiency(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Calculate Kling-Gupta Efficiency (KGE-2009).

    Decomposes hydrologic performance into correlation (r), variability ratio (gamma),
    and bias ratio (beta):
        KGE = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2)
    where:
        r = Pearson correlation coefficient
        beta = mean(y_pred) / mean(y_true)
        gamma = (std(y_pred) / mean(y_pred)) / (std(y_true) / mean(y_true))
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) < 3:
        return float("nan")

    std_t = np.std(yt)
    std_p = np.std(yp)
    mean_t = np.mean(yt)
    mean_p = np.mean(yp)

    if std_t == 0.0 or std_p == 0.0 or mean_t == 0.0 or mean_p == 0.0:
        return float("nan")

    r = float(np.corrcoef(yt, yp)[0, 1])
    beta = float(mean_p / mean_t)
    gamma = float((std_p / mean_p) / (std_t / mean_t))

    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)
    return float(kge)


def percent_bias(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Calculate Percent Bias (PBIAS).

    PBIAS = 100 * [ sum(y_pred - y_true) / sum(y_true) ]
    Positive values indicate overestimation; negative indicate underestimation.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sum_t = np.sum(yt)
    if sum_t == 0.0:
        return 0.0 if np.sum(yp) == 0.0 else float("inf")
    return float(100.0 * np.sum(yp - yt) / sum_t)


def log_nash_sutcliffe_efficiency(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    epsilon: float = 1e-4,
) -> float:
    """Calculate log-NSE for low-flow and recession hydrograph accuracy.

    Evaluates NSE on ln(y + epsilon).
    """
    yt = np.log(np.maximum(np.asarray(y_true, dtype=float), 0.0) + epsilon)
    yp = np.log(np.maximum(np.asarray(y_pred, dtype=float), 0.0) + epsilon)
    return nash_sutcliffe_efficiency(yt, yp)


def mse_skill_score(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    y_baseline: np.ndarray | list[float],
) -> float:
    """Calculate Mean Squared Error Skill Score (SS_ref) relative to baseline:

    SS = 1 - (MSE_model / MSE_baseline)
    Positive indicates improvement over baseline.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yb = np.asarray(y_baseline, dtype=float)

    mse_m = np.mean((yt - yp) ** 2)
    mse_b = np.mean((yt - yb) ** 2)
    if mse_b == 0.0:
        return 0.0 if mse_m == 0.0 else -float("inf")
    return float(1.0 - (mse_m / mse_b))


def top_flow_relative_error(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    top_quantile: float = 0.95,
) -> float:
    """Calculate mean relative percentage error evaluated on the top flows (>= 95th percentile)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")

    threshold = np.percentile(yt, top_quantile * 100)
    mask = yt >= threshold
    if not np.any(mask):
        return 0.0

    top_true = yt[mask]
    top_pred = yp[mask]
    rel_errors = (top_pred - top_true) / np.maximum(top_true, 1e-4) * 100.0
    return float(np.mean(rel_errors))


def diebold_mariano_test(
    y_true: np.ndarray | list[float],
    y_pred_model: np.ndarray | list[float],
    y_pred_baseline: np.ndarray | list[float],
    h: int = 1,
) -> dict[str, Any]:
    """Perform Paired Diebold-Mariano test of predictive accuracy vs Persistence baseline.

    Uses squared-error loss differential:
        d_t = (y_t - y_pred_model_t)^2 - (y_t - y_pred_baseline_t)^2
    With Harvey, Leybourne, Newbold (1997) finite-sample correction.

    Null hypothesis H0: Expected loss differential is zero (forecasts have equal accuracy).
    Alternative H1: Model has significantly lower loss (one-tailed or two-tailed).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_model, dtype=float)
    yb = np.asarray(y_pred_baseline, dtype=float)
    n = len(yt)

    if n < 5:
        return {
            "dm_stat": float("nan"),
            "p_value": 1.0,
            "statistically_significant": False,
            "interpretation": "Insufficient samples for significance testing (n < 5)",
        }

    # Loss differential (negative means model has lower error than baseline)
    d = (yt - yp) ** 2 - (yt - yb) ** 2
    mean_d = float(np.mean(d))

    # Autocovariance up to lag h-1 (standard Diebold-Mariano bandwidth)
    autocov_sum = 0.0
    for lag in range(1, max(h, 1)):
        gamma = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        autocov_sum += 2.0 * gamma

    var_d = np.var(d, ddof=1) + autocov_sum
    if var_d <= 0.0:
        se_d = 1e-6
    else:
        se_d = np.sqrt(var_d / n)

    dm_stat = mean_d / se_d

    # Harvey-Leybourne-Newbold small sample adjustment factor
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_hln = dm_stat * hln_factor

    # Two-sided p-value against Student-t with n-1 df
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat_hln), df=n - 1)))
    # Model beats baseline significantly if mean_d < 0 and p_val < 0.05
    is_sig = bool(mean_d < 0 and p_val < 0.05)

    interpretation = (
        "Statistically significant improvement over Persistence (p < 0.05)"
        if is_sig
        else "No statistically significant difference vs Persistence (p >= 0.05)"
    )

    return {
        "dm_stat": round(float(dm_stat_hln), 4),
        "mean_loss_diff": round(mean_d, 4),
        "p_value": round(p_val, 4),
        "statistically_significant": is_sig,
        "interpretation": interpretation,
    }


def compute_standard_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    y_baseline: np.ndarray | list[float] | None = None,
    horizon_days: int = 1,
) -> dict[str, Any]:
    """Compute comprehensive hydrologic continuous metrics including KGE, PBIAS, log-NSE, and DM test."""
    res = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "nse": nash_sutcliffe_efficiency(y_true, y_pred),
        "kge": kling_gupta_efficiency(y_true, y_pred),
        "pbias": percent_bias(y_true, y_pred),
        "log_nse": log_nash_sutcliffe_efficiency(y_true, y_pred),
        "top_5pct_relative_error": top_flow_relative_error(y_true, y_pred, top_quantile=0.95),
    }

    if y_baseline is not None:
        res["skill_score_vs_persistence"] = mse_skill_score(y_true, y_pred, y_baseline)
        dm_res = diebold_mariano_test(y_true, y_pred, y_baseline, h=horizon_days)
        res["dm_test"] = dm_res
        res["statistically_significant_beats_baseline"] = dm_res["statistically_significant"]
        res["significance_p_value"] = dm_res["p_value"]
    else:
        res["statistically_significant_beats_baseline"] = False
        res["significance_p_value"] = 1.0

    return res
