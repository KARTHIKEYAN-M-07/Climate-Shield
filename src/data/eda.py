"""
src/data/eda.py
───────────────
Exploratory Data Analysis: generate and save plots to reports/.

All plots are saved as PNG files — they are NOT displayed interactively.
Each function returns the path(s) to the saved file(s).

Plots produced
──────────────
1. time_series      – rainfall + discharge dual-axis time series
2. monthly_box      – monthly distribution box-plots (rainfall, discharge)
3. rolling_stats    – 30-day rolling mean + std for discharge
4. scatter_lag      – discharge(t) vs rainfall(t-k) for k=1,3,7
5. missing_heatmap  – calendar heatmap of missing values per year/month
6. correlation      – Pearson correlation matrix heat-map

Credit: Weather and flood data by Open-Meteo.com (CC BY 4.0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in server/test environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── consistent style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

_BLUE = "#1f77b4"
_ORANGE = "#ff7f0e"
_RED = "#d62728"


# =============================================================================
# Public helpers
# =============================================================================


def _reports_dir(cfg: dict[str, Any]) -> Path:
    d = _PROJECT_ROOT / cfg["paths"]["reports_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _savefig(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("EDA plot saved: %s", path)
    return path


# =============================================================================
# 1. Time series
# =============================================================================


def plot_time_series(
    df: pd.DataFrame,
    station_id: str,
    cfg: dict[str, Any],
) -> Path:
    """Dual-axis daily time series: precipitation (bar) + discharge (line)."""
    dates = pd.to_datetime(df["date"])
    rain = df["precipitation_sum"].fillna(0)
    disc = df["river_discharge"]

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.bar(dates, rain, color=_BLUE, alpha=0.5, width=1.0, label="Rainfall (mm/day)")
    ax1.set_ylabel("Precipitation (mm/day)", color=_BLUE)
    ax1.tick_params(axis="y", labelcolor=_BLUE)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    ax2 = ax1.twinx()
    ax2.plot(dates, disc, color=_ORANGE, linewidth=0.8,
             label="Discharge (m³/s, modelled)")
    ax2.set_ylabel("River Discharge (m³/s)", color=_ORANGE)
    ax2.tick_params(axis="y", labelcolor=_ORANGE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax1.set_title(
        f"{station_id} — Daily Rainfall & Modelled River Discharge\n"
        f"(Open-Meteo modelled discharge, not gauge data)",
        fontsize=10,
    )

    out = _reports_dir(cfg) / f"{station_id}_timeseries.png"
    return _savefig(fig, out)


# =============================================================================
# 2. Monthly box-plots
# =============================================================================


def plot_monthly_box(
    df: pd.DataFrame,
    station_id: str,
    cfg: dict[str, Any],
) -> list[Path]:
    """Box-plots of monthly distributions for rainfall and discharge."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    paths = []

    for col, label, colour, unit in [
        ("precipitation_sum", "Rainfall", _BLUE, "mm/day"),
        ("river_discharge", "Modelled Discharge", _ORANGE, "m³/s"),
    ]:
        if col not in df.columns:
            continue
        groups = [df.loc[df["month"] == m, col].dropna().values
                  for m in range(1, 13)]

        fig, ax = plt.subplots(figsize=(12, 4))
        bp = ax.boxplot(groups, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels)
        ax.set_xlabel("Month")
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_title(f"{station_id} — Monthly {label} Distribution")

        out = _reports_dir(cfg) / f"{station_id}_monthly_box_{col}.png"
        paths.append(_savefig(fig, out))

    return paths


# =============================================================================
# 3. Rolling statistics
# =============================================================================


def plot_rolling_stats(
    df: pd.DataFrame,
    station_id: str,
    cfg: dict[str, Any],
    window: int = 30,
) -> Path:
    """30-day rolling mean ± 1 std of modelled discharge."""
    dates = pd.to_datetime(df["date"])
    disc = df["river_discharge"]
    roll_mean = disc.rolling(window, min_periods=1).mean()
    roll_std = disc.rolling(window, min_periods=1).std().fillna(0)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, disc, color=_BLUE, linewidth=0.5, alpha=0.4, label="Daily")
    ax.plot(dates, roll_mean, color=_ORANGE, linewidth=1.5,
            label=f"{window}-day rolling mean")
    ax.fill_between(dates, roll_mean - roll_std, roll_mean + roll_std,
                    color=_ORANGE, alpha=0.15, label="±1 std")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title(f"{station_id} — {window}-Day Rolling Stats (Modelled Discharge)")
    ax.legend(fontsize=8)

    out = _reports_dir(cfg) / f"{station_id}_rolling_stats.png"
    return _savefig(fig, out)


# =============================================================================
# 4. Scatter: discharge(t) vs rainfall(t-k)
# =============================================================================


def plot_scatter_lag(
    df: pd.DataFrame,
    station_id: str,
    cfg: dict[str, Any],
    lags: list[int] | None = None,
) -> Path:
    """Scatter of discharge(t) vs rainfall(t-k) for k in lags."""
    if lags is None:
        lags = [1, 3, 7]

    disc = df["river_discharge"]
    rain = df["precipitation_sum"]

    fig, axes = plt.subplots(1, len(lags), figsize=(5 * len(lags), 4), sharey=True)
    if len(lags) == 1:
        axes = [axes]

    for ax, lag in zip(axes, lags):
        x = rain.shift(lag)
        valid = disc.notna() & x.notna()
        ax.scatter(x[valid], disc[valid], s=3, alpha=0.3, color=_BLUE)
        corr = x[valid].corr(disc[valid])
        ax.set_xlabel(f"Rainfall(t-{lag}) (mm/day)")
        ax.set_title(f"Lag {lag}  r={corr:.2f}")
        if ax is axes[0]:
            ax.set_ylabel("Discharge(t) (m³/s)")

    fig.suptitle(f"{station_id} — Discharge vs Lagged Rainfall", y=1.01)
    fig.tight_layout()

    out = _reports_dir(cfg) / f"{station_id}_scatter_lag.png"
    return _savefig(fig, out)


# =============================================================================
# 5. Missing-value heatmap  (year × month)
# =============================================================================


def plot_missing_heatmap(
    df: pd.DataFrame,
    station_id: str,
    cfg: dict[str, Any],
) -> Path:
    """Calendar heatmap: fraction of missing values per (year, month)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    for col in ["precipitation_sum", "river_discharge"]:
        if col not in df.columns:
            df[col] = np.nan

    df["any_missing"] = df["precipitation_sum"].isna() | df["river_discharge"].isna()

    pivot = df.groupby(["year", "month"])["any_missing"].mean().unstack(fill_value=0)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(12))
    ax.set_xticklabels([month_labels[m - 1] for m in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=8)
    ax.set_title(f"{station_id} — Fraction of Missing Values (year × month)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="Fraction missing")
    fig.tight_layout()

    out = _reports_dir(cfg) / f"{station_id}_missing_heatmap.png"
    return _savefig(fig, out)


# =============================================================================
# 6. Correlation matrix
# =============================================================================


def plot_correlation(
    df: pd.DataFrame,
    station_id: str,
    cfg: dict[str, Any],
) -> Path:
    """Pearson correlation matrix of numeric columns."""
    numeric_cols = [c for c in ["precipitation_sum", "river_discharge"]
                    if c in df.columns]
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation plot.")
        return None

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    labels = ["Rainfall\n(mm/day)", "Discharge\n(m³/s, modelled)"][:len(numeric_cols)]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{station_id} — Pearson Correlation")
    fig.tight_layout()

    out = _reports_dir(cfg) / f"{station_id}_correlation.png"
    return _savefig(fig, out)


# =============================================================================
# Convenience: run all EDA plots at once
# =============================================================================


def run_all_eda(
    df: pd.DataFrame,
    station_id: str,
    cfg: dict[str, Any],
) -> list[Path]:
    """Run all six EDA plots and return list of saved paths."""
    paths: list[Path] = []
    for fn in [
        lambda: plot_time_series(df, station_id, cfg),
        lambda: plot_monthly_box(df, station_id, cfg),
        lambda: plot_rolling_stats(df, station_id, cfg),
        lambda: plot_scatter_lag(df, station_id, cfg),
        lambda: plot_missing_heatmap(df, station_id, cfg),
        lambda: plot_correlation(df, station_id, cfg),
    ]:
        result = fn()
        if result is None:
            continue
        if isinstance(result, list):
            paths.extend(result)
        else:
            paths.append(result)

    logger.info("EDA complete: %d plots saved for %s.", len(paths), station_id)
    return paths
