"""
src/features/temporal_features.py
──────────────────────────────────
Reusable, pure feature-engineering functions for the flood-prediction pipeline.

Leakage contract
────────────────
Every function here MUST satisfy the causality constraint:
    feature(t) uses only data[0 … t] (no data[t+1] or later)

For forecast-covariate mode:
    forecast_covariate(t, h) uses ONLY precipitation forecasts issued at or prior to origin t
    covering [t+1 … t+h]. It NEVER touches observed future rainfall at t+1.

API
───
All public functions accept a pd.DataFrame with at minimum:
    date               – str or datetime (YYYY-MM-DD), monotonically increasing
    precipitation_sum  – float, mm/day
    river_discharge    – float, m³/s (modelled)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── default constants ─────────────────────────────────────────────────────────
LAG_DAYS: list[int] = [1, 2, 3, 7, 14]
CUMUL_WINDOWS: list[int] = [3, 7, 14]
ROLL_WINDOWS: list[int] = [7, 14, 30]
HORIZONS: list[int] = [1, 3, 7]


# =============================================================================
# Pure Feature Functions (Time t Causality Preserved)
# =============================================================================


def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: Sequence[int] = LAG_DAYS,
) -> pd.DataFrame:
    """Add lagged copies of *col* as new columns.

    For lag k: feature[t] = df[col][t - k]
    """
    df = df.copy()
    for k in lags:
        df[f"{col}_lag_{k}"] = df[col].shift(k)
    return df


def add_cumulative_rainfall(
    df: pd.DataFrame,
    col: str = "precipitation_sum",
    windows: Sequence[int] = CUMUL_WINDOWS,
) -> pd.DataFrame:
    """Add cumulative rainfall over the past *w* days including today.

    feature[t] = sum(df[col][t-w+1 … t])
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_cumul_{w}d"] = df[col].rolling(window=w, min_periods=1).sum()
    return df


def add_rolling_mean(
    df: pd.DataFrame,
    col: str,
    windows: Sequence[int] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """Add rolling mean of *col* over past *w* days including today.

    feature[t] = mean(df[col][t-w+1 … t])
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_mean_{w}d"] = df[col].rolling(window=w, min_periods=1).mean()
    return df


def add_rolling_max(
    df: pd.DataFrame,
    col: str,
    windows: Sequence[int] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """Add rolling maximum of *col* over past *w* days including today.

    feature[t] = max(df[col][t-w+1 … t])
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_max_{w}d"] = df[col].rolling(window=w, min_periods=1).max()
    return df


def add_rolling_std(
    df: pd.DataFrame,
    col: str = "river_discharge",
    windows: Sequence[int] = (7, 14),
) -> pd.DataFrame:
    """Add rolling standard deviation of *col* over past *w* days including today.

    feature[t] = std(df[col][t-w+1 … t])
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_std_{w}d"] = df[col].rolling(window=w, min_periods=2).std().fillna(0.0)
    return df


def add_rainfall_intensity(
    df: pd.DataFrame,
    col: str = "precipitation_sum",
    windows: Sequence[int] = (3, 7),
) -> pd.DataFrame:
    """Add rainfall intensity: max daily rainfall in the last 3 and 7 days.

    feature[t] = max(df[col][t-w+1 … t])
    """
    df = df.copy()
    for w in windows:
        df[f"precip_max_{w}d"] = df[col].rolling(window=w, min_periods=1).max()
    return df


def add_days_since_significant_rain(
    df: pd.DataFrame,
    col: str = "precipitation_sum",
    threshold_mm: float = 5.0,
) -> pd.DataFrame:
    """Calculate elapsed days since the last significant rainfall event (>= threshold_mm).

    Uses only past observations up to day t.
    """
    df = df.copy()
    rain = df[col].fillna(0.0).values
    n = len(rain)
    days_since = np.zeros(n, dtype=float)
    current_count = 999.0

    for t in range(n):
        if rain[t] >= threshold_mm:
            current_count = 0.0
        else:
            current_count = min(current_count + 1.0, 999.0)
        days_since[t] = current_count

    df["days_since_significant_rain"] = days_since
    return df



def add_rate_of_change(
    df: pd.DataFrame,
    col: str,
    lags: Sequence[int] = (1, 3, 7),
) -> pd.DataFrame:
    """Add absolute numeric differences relative to previous time steps."""
    df = df.copy()

    # Convert Boolean values to numeric values before subtraction.
    # Preserve existing numeric values.
    if pd.api.types.is_bool_dtype(df[col].dtype):
        numeric_col = df[col].astype(float)
    else:
        numeric_col = pd.to_numeric(df[col], errors="coerce")

    for k in lags:
        df[f"{col}_roc_{k}d"] = numeric_col - numeric_col.shift(k)

    return df


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin and cos encoding of calendar month on the unit circle."""
    df = df.copy()
    month = pd.to_datetime(df["date"]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


# ── Phase 13 Richer Features ──────────────────────────────────────────────────


def add_antecedent_precipitation_index(
    df: pd.DataFrame,
    col: str = "precipitation_sum",
    decay_factors: Sequence[float] = (0.85, 0.90),
) -> pd.DataFrame:
    """Compute Antecedent Precipitation Index (API) for soil wetness memory.

    Recursive formula:
        API[0] = rain[0]
        API[t] = decay * API[t-1] + rain[t]

    Strictly causal (uses only rain[0..t]).
    """
    df = df.copy()
    rain_vals = df[col].fillna(0.0).values
    n = len(rain_vals)

    for k in decay_factors:
        api = np.zeros(n, dtype=float)
        if n > 0:
            api[0] = rain_vals[0]
            for t in range(1, n):
                api[t] = k * api[t - 1] + rain_vals[t]
        k_str = str(int(k * 100))
        df[f"api_decay_{k_str}"] = api
    return df


def add_monsoon_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add Tamil Nadu regional monsoon season flags.

    - Southwest Monsoon (SW): June to September (months 6 to 9)
    - Northeast Monsoon (NE): October to December (months 10 to 12 - primary TN rainfall period)
    - Dry Period: January to May (months 1 to 5)
    """
    df = df.copy()
    month = pd.to_datetime(df["date"]).dt.month
    df["monsoon_sw"] = month.isin([6, 7, 8, 9]).astype(float)
    df["monsoon_ne"] = month.isin([10, 11, 12]).astype(float)
    return df


def add_days_since_heavy_rain(
    df: pd.DataFrame,
    col: str = "precipitation_sum",
    threshold_mm: float = 35.5,
) -> pd.DataFrame:
    """Calculate elapsed days since the last heavy rainfall event (>= threshold_mm).

    Uses only past observations up to day t.
    """
    df = df.copy()
    rain = df[col].fillna(0.0).values
    n = len(rain)
    days_since = np.zeros(n, dtype=float)
    current_count = 999.0  # Large baseline if no prior event

    for t in range(n):
        if rain[t] >= threshold_mm:
            current_count = 0.0
        else:
            current_count += 1.0
        days_since[t] = current_count

    df["days_since_heavy_rain"] = days_since
    return df


def add_wet_spell_length(
    df: pd.DataFrame,
    col: str = "precipitation_sum",
    threshold_mm: float = 1.0,
) -> pd.DataFrame:
    """Calculate consecutive days of measurable precipitation up to time t."""
    df = df.copy()
    rain = df[col].fillna(0.0).values
    n = len(rain)
    spell_len = np.zeros(n, dtype=float)
    curr_len = 0.0

    for t in range(n):
        if rain[t] >= threshold_mm:
            curr_len += 1.0
        else:
            curr_len = 0.0
        spell_len[t] = curr_len

    df["wet_spell_length_days"] = spell_len
    return df


def add_catchment_areal_rainfall(
    df: pd.DataFrame,
    upstream_cols: Sequence[str],
    weights: Sequence[float] | None = None,
    output_col: str = "catchment_areal_rain_mm",
) -> pd.DataFrame:
    """Compute weighted spatial average of catchment rainfall.

    areal_rain[t] = sum(w_i * rain_i[t]) / sum(w_i)
    """
    df = df.copy()
    valid_cols = [c for c in upstream_cols if c in df.columns]
    if not valid_cols:
        return df

    if weights is None or len(weights) != len(valid_cols):
        w = np.ones(len(valid_cols)) / len(valid_cols)
    else:
        w = np.array(weights) / np.sum(weights)

    areal_val = np.zeros(len(df))
    for col, weight in zip(valid_cols, w):
        areal_val += df[col].fillna(0.0).values * weight

    df[output_col] = areal_val
    return df


# =============================================================================
# Target Creation
# =============================================================================


def add_targets(
    df: pd.DataFrame,
    col: str = "river_discharge",
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Add forward-shifted target columns for each forecast horizon.

    target[t] = df[col][t + h]
    """
    df = df.copy()
    for h in horizons:
        df[f"target_discharge_h{h}d"] = df[col].shift(-h)
    return df


# =============================================================================
# Full Feature Matrix Builder
# =============================================================================


def build_feature_matrix(
    df: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS,
    lag_days: Sequence[int] = LAG_DAYS,
    cumul_windows: Sequence[int] = CUMUL_WINDOWS,
    roll_windows: Sequence[int] = ROLL_WINDOWS,
    roc_lags: Sequence[int] = (1, 3, 7),
    include_rich_features: bool = True,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Build the complete feature + target matrix from an aligned dataset."""
    out = df.copy()

    # ── 1. Base Lags ──────────────────────────────────────────────────────────
    out = add_lag_features(out, "precipitation_sum", lags=lag_days)
    out = add_lag_features(out, "river_discharge", lags=lag_days)

    # ── 2. Cumulative rainfall ────────────────────────────────────────────────
    out = add_cumulative_rainfall(out, "precipitation_sum", windows=cumul_windows)

    # ── 3. Rolling mean ───────────────────────────────────────────────────────
    out = add_rolling_mean(out, "precipitation_sum", windows=roll_windows)
    out = add_rolling_mean(out, "river_discharge", windows=roll_windows)

    # ── 4. Rolling max ────────────────────────────────────────────────────────
    out = add_rolling_max(out, "precipitation_sum", windows=roll_windows)
    out = add_rolling_max(out, "river_discharge", windows=roll_windows)

    # ── 5. Rate of change ─────────────────────────────────────────────────────
    out = add_rate_of_change(out, "precipitation_sum", lags=roc_lags)
    out = add_rate_of_change(out, "river_discharge", lags=roc_lags)

    # ── 6. Cyclical month ─────────────────────────────────────────────────────
    out = add_cyclical_month(out)

    # ── 7. Richer hydrological & meteorological features (Phase 13 & Part B) ──
    out = add_rolling_std(out, "river_discharge", windows=(7, 14))
    out = add_rainfall_intensity(out, "precipitation_sum", windows=(3, 7))
    out = add_days_since_significant_rain(out, "precipitation_sum", threshold_mm=5.0)

    if include_rich_features:
        out = add_antecedent_precipitation_index(out, "precipitation_sum")
        out = add_monsoon_flags(out)
        out = add_days_since_heavy_rain(out, "precipitation_sum")
        out = add_wet_spell_length(out, "precipitation_sum")

    if include_targets:
        # ── 8. Multi-horizon targets (Forward shift) ──────────────────────────
        out = add_targets(out, "river_discharge", horizons=horizons)

        # ── 9. Drop tail rows with NaN in any target column ───────────────────
        target_cols = [f"target_discharge_h{h}d" for h in horizons]
        out = out.dropna(subset=target_cols).reset_index(drop=True)
    else:
        target_cols = []

    feature_cols = get_feature_columns(out, horizons=horizons)
    logger.info(
        "Feature matrix built: %d rows × %d features + %d targets.",
        len(out), len(feature_cols), len(target_cols),
    )
    return out


def get_feature_columns(
    df: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS,
    exclude: Sequence[str] = ("date",),
) -> list[str]:
    """Return the list of model input feature column names (excludes targets, date, raw discharge)."""
    target_cols = {f"target_discharge_h{h}d" for h in horizons}
    raw_cols = {"precipitation_sum", "river_discharge"}
    excluded = set(exclude) | target_cols | raw_cols
    return [c for c in df.columns if c not in excluded]


def get_target_column(horizon: int) -> str:
    """Return target column name for a given forecast horizon."""
    return f"target_discharge_h{horizon}d"
