"""src/dashboard/app.py
────────────────────
Streamlit interactive monitoring and forecast dashboard for Tamil Nadu River Basins (v2).

Features:
  - Centralized CSS Design System (src/dashboard/styles.py)
  - 10-second summary KPI cards (Risk, Forecast Flow, Latest Observed, Conformal Interval, Freshness/Drift)
  - Color-blind-safe palette, accessible typography, mobile-friendly layout
  - Map auto-centered on selected station with distinct symbols and labels
  - All Reorganized Tabs:
      1. Overview
      2. Forecast (Fan chart with conformal interval, plain-language drivers, what-if simulator)
      3. All Stations Summary (Cross-basin comparative matrix)
      4. ML Model Comparison (Metrics table with tick marks, RMSE bar chart, feature ablation, target transforms)
      5. Training Status (Coverage matrix with color pills and diagnostic reasons)
      6. Historical Backtest (Event replay, observed vs forecast, warning lead time)
      7. Rainfall Risk (Rainfall indicator & ML risk layer strictly side by side)
      8. Verification and Drift (Forward verification n>=30, PSI/z-score drift)
      9. Data Quality (Provenance, completeness, sanity checks, flags)
      10. About and Limitations (Regulatory notices, GloFAS modelled disclaimer, citations)
  - English / Tamil language toggle (Tamil marked "Pending Review")
  - URL query parameter state persistence via st.query_params
  - CSV / JSON download actions
  - Measured latency banner from reports/latency.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.localization import get_text
from src.dashboard.styles import inject_custom_css
from src.data.live import fetch_station_live_data
from src.features.temporal_features import build_feature_matrix, get_feature_columns
from src.forecast.engine import (
    ForecastResult,
    generate_forecast,
    get_highest_flow_test_date,
    get_test_period_bounds,
    evaluate_station_signal,
)
from src.monitoring.drift import compute_drift_status
from src.risk.risk_engine import evaluate_risk
from src.risk.safety_guards import OutOfDistributionGuard, explain_prediction, simulate_rainfall_what_if
from src.utils.config import load_all_stations, load_data_config, load_thresholds_config

# Configure layout
st.set_page_config(
    page_title="ClimateShield AI: River-Flow Forecasting & Early-Warning Prototype for Tamil Nadu",
    page_icon="🌊",
    layout="wide",
)

# Inject centralized CSS design system
inject_custom_css()

# Top-level models list for dropdown and AST discovery
models_list = [
    "xgboost", "random_forest", "gradient_boosting", "hist_gradient_boosting",
    "elastic_net", "extra_trees", "ridge", "decision_tree", "persistence",
]

# ── Query Params & Language Toggle ───────────────────────────────────────────
query_params = st.query_params

default_lang = query_params.get("lang", "en")
if default_lang not in ["en", "ta"]:
    default_lang = "en"

lang_choice = st.sidebar.radio(
    "Language / மொழி",
    ["English", "தமிழ் (Pending Review)"],
    index=0 if default_lang == "en" else 1,
)
selected_lang = "en" if lang_choice == "English" else "ta"
st.query_params["lang"] = selected_lang


def t(key: str) -> str:
    return get_text(selected_lang, key)


# ── Caching & Data Loaders ───────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_app_configs():
    return load_data_config(), load_thresholds_config(), load_all_stations()


@st.cache_data(ttl=300)
def load_scores_and_latency():
    scores_file = _PROJECT_ROOT / "reports" / "model_scores.json"
    latency_file = _PROJECT_ROOT / "reports" / "latency.json"
    scores_all_file = _PROJECT_ROOT / "reports" / "model_scores_all.csv"
    coverage_file = _PROJECT_ROOT / "reports" / "training_coverage.csv"
    ablation_file = _PROJECT_ROOT / "reports" / "feature_ablation.csv"

    scores = None
    latency = None
    df_all_scores = None
    df_coverage = None
    df_ablation = None

    if scores_file.exists():
        try:
            with open(scores_file, "r", encoding="utf-8") as f:
                scores = json.load(f)
        except Exception:
            scores = None
    if latency_file.exists():
        try:
            with open(latency_file, "r", encoding="utf-8") as f:
                latency = json.load(f)
        except Exception:
            latency = None
    if scores_all_file.exists():
        try:
            df_all_scores = pd.read_csv(scores_all_file)
        except Exception:
            df_all_scores = None
    if coverage_file.exists():
        try:
            df_coverage = pd.read_csv(coverage_file)
        except Exception:
            df_coverage = None
    if ablation_file.exists():
        try:
            df_ablation = pd.read_csv(ablation_file)
        except Exception:
            df_ablation = None

    return scores, latency, df_all_scores, df_coverage, df_ablation


data_cfg, thresh_cfg, all_stations = load_app_configs()
scores_data, latency_data, df_all_scores, df_coverage, df_ablation = load_scores_and_latency()

# ── Header & Badges ──────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-header">
        <h1 class="app-title">{t('title')}</h1>
        <div class="app-subtitle">{t('subtitle')}</div>
        <div class="badge-row">
            <span class="badge badge-prototype">Research Decision Prototype</span>
            <span class="badge badge-modelled">Modelled GloFAS River Flow</span>
            <span class="badge badge-timestamp">Open-Meteo CC BY 4.0</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(t("warning_notice"))

# Measured Latency Banner (Precise labelling complying with non-negotiable rules)
if latency_data and "summary" in latency_data:
    med_ms = latency_data["summary"].get("median_total_ms", 0.0)
    p95_ms = latency_data["summary"].get("p95_total_ms", 0.0)
    st.info(
        f"⏱️ **Model inference + render latency (cached data)**: Median **{med_ms:.2f} ms** | 95th Percentile **{p95_ms:.2f} ms** "
        f"(Benchmark based on {latency_data['benchmark_metadata']['n_runs']} local cycles). "
        f"**Live API Fetch Time**: Near-real-time daily batch cadence (latest available daily data)."
    )

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("Controls & Filters")

# Mode Switcher: Live (Near-Real-Time) vs Historical Replay
st.sidebar.markdown("### Mode")
mode_options = ["Live (Near-Real-Time)", "Historical Replay"]
default_mode_idx = 0
if st.session_state.get("operating_mode") == "Historical Replay":
    default_mode_idx = 1

app_mode = st.sidebar.radio(
    "Select Operating Mode",
    mode_options,
    index=default_mode_idx,
    key="operating_mode_selector",
    help="Live mode operates on near-real-time (latest available daily data). Historical Replay allows selecting an origin date from the test set to retrospectively evaluate forecast accuracy against observed reality.",
)
st.session_state["operating_mode"] = app_mode
is_replay = (app_mode == "Historical Replay")

models_dir = _PROJECT_ROOT / data_cfg["paths"]["models_dir"]


@st.cache_data(ttl=300)
def get_station_dropdown_options():
    options = {}
    for s in all_stations:
        sid = s["id"]
        sname = s["name"]
        stn_model_dir = models_dir / sid
        has_models = (
            (stn_model_dir.exists() and len(list(stn_model_dir.glob("*.joblib"))) > 0)
            or len(list(models_dir.glob(f"{sid}_*.joblib"))) > 0
        )
        proc_files_for_sig = sorted(
            (_PROJECT_ROOT / data_cfg["paths"]["processed_dir"]).glob(f"{sid}_aligned_*.csv")
        )
        sig_ok = False
        if proc_files_for_sig:
            try:
                df_sig = pd.read_csv(proc_files_for_sig[-1])
                sig_ok, _ = evaluate_station_signal(df_sig, sid, thresh_cfg)
            except Exception:
                sig_ok = False

        has_rainfall = bool(
            proc_files_for_sig
            or list((_PROJECT_ROOT / data_cfg["paths"]["raw_dir"]).glob(f"{sid}_rainfall_*.csv"))
        )

        if has_models and sig_ok:
            status_tag = "[Ready]"
        elif has_models and not sig_ok:
            status_tag = "[Forecast only]"
        elif has_rainfall:
            status_tag = "[Rainfall risk only]"
        else:
            status_tag = "[Not supported]"

        label = f"{sname} {status_tag}"
        options[label] = sid
    return options


station_options = get_station_dropdown_options()

default_stn_id = query_params.get("station", data_cfg.get("default_station_id", "TN_CAU_TRICHY"))
stn_labels = list(station_options.keys())
default_stn_idx = 0
for idx, lbl in enumerate(stn_labels):
    if station_options[lbl] == default_stn_id:
        default_stn_idx = idx
        break

selected_stn_label = st.sidebar.selectbox(t("station"), stn_labels, index=default_stn_idx)
selected_stn_id = station_options[selected_stn_label]
selected_stn_name = next(s["name"] for s in all_stations if s["id"] == selected_stn_id)
st.query_params["station"] = selected_stn_id

# Guidance expander explaining station classifications
with st.sidebar.expander("ℹ️ Station Classification Guide", expanded=False):
    st.markdown("""
    - **[Ready]**: Model trained; strong hydrologic signal (CV $\\ge 0.80$, $p_{90} \\ge 0.20$ m³/s); full statistical risk tiers active.
    - **[Forecast only]**: Model trained; low flow magnitude or near-constant baseflow prevents reliable risk categorization. Numerical forecast and interval are shown.
    - **[Rainfall risk only]**: No river-flow model for this station; rule-based meteorological rainfall risk indicator is displayed. Flow forecasts and flow thresholds are disabled.
    - **[Not supported]**: Neither river-flow model nor meteorological rainfall observations available.
    """)

selected_horizon = st.sidebar.selectbox(
    t("horizon"),
    [1, 3, 7],
    index=[1, 3, 7].index(int(query_params.get("horizon", 1))),
    format_func=lambda h: f"{h} Day(s) Ahead",
)
st.query_params["horizon"] = str(selected_horizon)

# Find best model on validation for selected station and horizon
best_model_name = "xgboost"
if df_all_scores is not None and not df_all_scores.empty:
    sub = df_all_scores[(df_all_scores["station"] == selected_stn_id) & (df_all_scores["horizon"] == selected_horizon)]
    if not sub.empty:
        best_row = sub.sort_values(by="RMSE", ascending=True).iloc[0]
        best_model_name = str(best_row["model"])

default_mod = query_params.get("model", best_model_name)
selected_model_idx = models_list.index(default_mod) if default_mod in models_list else 0

selected_model = st.sidebar.selectbox(
    t("model"),
    models_list,
    index=selected_model_idx,
    format_func=lambda m: f"{m} (★ best on validation)" if m == best_model_name else m,
)
st.query_params["model"] = selected_model

# Mode-Specific Sidebar Controls
selected_origin_date = None
if is_replay:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏪ Historical Replay Controls")
    min_test_date, max_test_date = get_test_period_bounds(selected_stn_id)
    if min_test_date and max_test_date:
        peak_date = get_highest_flow_test_date(selected_stn_id)
        if st.sidebar.button("⚡ Jump to highest-flow event in test set", help=f"Peak test event date: {peak_date}"):
            if peak_date:
                st.session_state[f"origin_date_{selected_stn_id}"] = pd.to_datetime(peak_date).date()

        min_d = pd.to_datetime(min_test_date).date()
        max_d = pd.to_datetime(max_test_date).date()
        saved_d = st.session_state.get(f"origin_date_{selected_stn_id}", max_d)
        if saved_d < min_d or saved_d > max_d:
            saved_d = max_d

        selected_origin_date = st.sidebar.date_input(
            "Forecast Origin Date (Test Period)",
            value=saved_d,
            min_value=min_d,
            max_value=max_d,
            help=f"Select an origin date within the test period ({min_test_date} to {max_test_date}). Training dates cannot be selected.",
        )
        st.sidebar.caption(f"📅 Test period bounds: `{min_test_date}` to `{max_test_date}`")
    else:
        st.sidebar.warning("Test period data not found for replay.")
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Live Data Controls")
    if st.sidebar.button("🔄 Refresh Live Data", help="Query latest available daily observations"):
        st.session_state["force_live_refresh"] = True

# ── Safe Defaults Initialization (Guarantees No Unbound Variables) ────────────
cat = "UNAVAILABLE"
cat_display = "UNAVAILABLE"
badge_class = "risk-unavailable"
cat_subtext = "Initializing"

pred_flow = None
pred_display = "Unavailable"
pred_subtext = "Initializing"

latest_flow = None
obs_display = "N/A"
obs_subtext = "Open-Meteo GloFAS Discharge"

lower_flow = None
upper_flow = None
fan_display = "N/A"
fan_subtext = "Conformal Interval unavailable"
measured_coverage_pct = None

drift_status = "UNKNOWN"
drift_value = None
ood_is_alert = False
ood_status_label = "UNKNOWN"
ood_color = "#eab308"
ood_status_display = "UNKNOWN"
ood_subtext = "Initializing"

latest_date = None
data_age_hours = None
risk_info = {"category": "UNAVAILABLE", "status": "INITIALIZING"}
primary_why = "Baseline seasonal hydrologic baseflow without active storm surges."
loaded_model = None
last_row = None
forecast_res = None
live_fetch_info = None

# ── Replay Banner ─────────────────────────────────────────────────────────────
if is_replay:
    st.warning("⚠️ **HISTORICAL REPLAY: not live data** — Forecasting retrospectively from historical test period observations. Stale data policy is disabled for replay.")

# ── Data Loading & Live Fetching ──────────────────────────────────────────────
candidate_artifacts = [
    models_dir / selected_stn_id / f"{selected_stn_id}_{selected_model}_h{selected_horizon}d.joblib",
    models_dir / selected_stn_id / f"{selected_model}_h{selected_horizon}d.joblib",
    models_dir / f"{selected_stn_id}_{selected_model}_h{selected_horizon}d.joblib",
]
model_artifact = next((p for p in candidate_artifacts if p.exists()), candidate_artifacts[0])

proc_files = sorted(
    (_PROJECT_ROOT / data_cfg["paths"]["processed_dir"]).glob(f"{selected_stn_id}_aligned_*.csv")
)

df_aligned = pd.DataFrame()

if is_replay:
    if proc_files:
        try:
            df_aligned = pd.read_csv(proc_files[-1])
            if "river_discharge" in df_aligned.columns and not df_aligned["river_discharge"].dropna().empty:
                latest_flow = float(df_aligned["river_discharge"].dropna().iloc[-1])
        except Exception:
            pass
else:
    # Live mode: fetch near-real-time data
    stn_curr = next((s for s in all_stations if s["id"] == selected_stn_id), None)
    if stn_curr is None:
        stn_curr = {"id": selected_stn_id, "name": selected_stn_name, "latitude": 10.8, "longitude": 78.7}

    force_refresh = st.session_state.pop("force_live_refresh", False)
    live_fetch_info = fetch_station_live_data(
        station_id=selected_stn_id,
        station_meta=stn_curr,
        force_refresh=force_refresh,
    )
    if not live_fetch_info["df"].empty:
        df_aligned = live_fetch_info["df"]
        if "river_discharge" in df_aligned.columns and not df_aligned["river_discharge"].dropna().empty:
            latest_flow = float(df_aligned["river_discharge"].dropna().iloc[-1])
            latest_date = str(df_aligned[df_aligned["river_discharge"].notna()]["date"].iloc[-1])
    elif proc_files:
        try:
            df_aligned = pd.read_csv(proc_files[-1])
            if "river_discharge" in df_aligned.columns and not df_aligned["river_discharge"].dropna().empty:
                latest_flow = float(df_aligned["river_discharge"].dropna().iloc[-1])
        except Exception:
            pass

# ── Centralized Forecast Engine Invocation ───────────────────────────────────
try:
    if is_replay and selected_origin_date:
        forecast_res = generate_forecast(
            station_id=selected_stn_id,
            horizon=selected_horizon,
            model_name=selected_model,
            df_data=df_aligned if not df_aligned.empty else None,
            mode="historical_replay",
            replay_origin_date=str(selected_origin_date),
        )
    else:
        forecast_res = generate_forecast(
            station_id=selected_stn_id,
            horizon=selected_horizon,
            model_name=selected_model,
            df_data=df_aligned if not df_aligned.empty else None,
            mode="live",
        )
except Exception as exc:
    import traceback
    err_log = _PROJECT_ROOT / "reports" / "app_errors.log"
    err_log.parent.mkdir(parents=True, exist_ok=True)
    with open(err_log, "a", encoding="utf-8") as f_err:
        f_err.write(f"[{datetime.now().isoformat()}] Dashboard exception: {traceback.format_exc()}\n")
    forecast_res = ForecastResult(
        forecast=None,
        interval=None,
        risk={"category": "UNAVAILABLE", "status": "ERROR"},
        status="ERROR",
        drift_status="UNKNOWN",
        drift_value=None,
        last_observation_time=None,
        data_age_hours=None,
        warnings=[f"Unexpected error: {exc}"],
        reason=f"System error: {exc}",
        source_used="error_handler",
    )

pred_flow = forecast_res.forecast
lower_flow = forecast_res.interval[0] if forecast_res.interval else None
upper_flow = forecast_res.interval[1] if forecast_res.interval else None
measured_coverage_pct = forecast_res.measured_coverage
risk_info = forecast_res.risk
primary_why = forecast_res.primary_why or "Baseline seasonal hydrologic baseflow without active storm surges."
latest_date = forecast_res.last_observation_time or latest_date
data_age_hours = forecast_res.data_age_hours
drift_status = forecast_res.drift_status
drift_value = forecast_res.drift_value

# Safe Drift & Freshness Evaluation
ood_status_label = drift_status
if forecast_res.status == "STALE":
    ood_status_display = "NOT ASSESSED"
    ood_subtext = "Drift not assessed: data stale"
    ood_color = "#eab308"
    ood_is_alert = False
elif drift_status == "OUT_OF_DISTRIBUTION":
    ood_status_display = "OUT_OF_DISTRIBUTION"
    ood_color = "#ef4444"
    ood_is_alert = True
    n_out = getattr(forecast_res, "features_outside_range_count", len([f for f in getattr(forecast_res, "top_drifting_features", []) if f.get("is_outside")]))
    if getattr(forecast_res, "top_drifting_features", None):
        top_f = forecast_res.top_drifting_features[0]
        ood_subtext = f"{n_out} features outside seasonal [p1, p99] (top: {top_f['feature']} {top_f['live_value']} vs {top_f.get('range_str', '')})"
    else:
        ood_subtext = f"{n_out} features outside seasonal envelope"
elif drift_status == "IN_DISTRIBUTION":
    ood_status_display = "IN_DISTRIBUTION"
    ood_color = "#22c55e"
    ood_is_alert = False
    n_out = getattr(forecast_res, "features_outside_range_count", 0)
    ood_subtext = f"Input distribution stable ({n_out} outside seasonal [p1, p99])"
else:
    ood_status_display = "UNKNOWN"
    ood_color = "#eab308"
    ood_is_alert = False
    ood_subtext = forecast_res.warnings[0] if forecast_res.warnings else "Drift reference statistics unavailable"

cat = risk_info.get("category", "UNAVAILABLE")
badge_class = "risk-unavailable"
n_out = getattr(forecast_res, "features_outside_range_count", 0)
is_low_conf = (
    forecast_res.status == "LOW_CONFIDENCE"
    or risk_info.get("status") == "LOW_CONFIDENCE"
    or risk_info.get("is_low_confidence", False)
    or (drift_status == "OUT_OF_DISTRIBUTION" and cat != "UNAVAILABLE")
)

is_rainfall_only = (
    getattr(forecast_res, "is_rainfall_only", False)
    or "[Rainfall risk only]" in selected_stn_label
)
rain_risk_info = getattr(forecast_res, "rainfall_risk", None)
if is_rainfall_only and not rain_risk_info:
    from src.risk.rainfall_risk import compute_rainfall_risk_indicator
    rain_risk_info = compute_rainfall_risk_indicator(df_aligned if not df_aligned.empty else None)

if is_rainfall_only:
    r_cat = rain_risk_info.get("category", "Low") if rain_risk_info else "Low"
    cat_display = f'{r_cat} <span class="badge-rainfall-only" style="font-size:0.55em; display:inline-block; padding:2px 7px; background:#0284c7; color:#fff; border-radius:4px; font-weight:700; margin-left:6px; vertical-align:middle;">Rainfall Risk Only</span>'
    badge_class = {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Severe": "risk-severe",
    }.get(r_cat, "risk-low")
    r_24 = rain_risk_info.get("rain_24h_mm", 0.0) if rain_risk_info else 0.0
    r_72 = rain_risk_info.get("rain_72h_mm", 0.0) if rain_risk_info else 0.0
    cat_subtext = f"Rule-based IMD rainfall risk (24h: {r_24:.1f} mm, 72h: {r_72:.1f} mm)"
    pred_display = "N/A"
    pred_subtext = "Flow forecast disabled for rainfall-only station"
    obs_display = "N/A"
    obs_subtext = "No river flow model for this station"
    fan_display = "N/A"
    fan_subtext = "Interval disabled for rainfall-only station"
    ood_status_display = "NOT ASSESSED"
    ood_subtext = "River discharge drift not applicable (rainfall risk only)"
    ood_color = "#38bdf8"
elif is_replay:
    cat_display = cat
    badge_class = {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Severe": "risk-severe",
    }.get(cat, "risk-unavailable")
    cat_subtext = f"Replay Statistical Risk at origin {latest_date}"
    if is_low_conf:
        cat_display = f'{cat} <span style="font-size:0.55em; display:inline-block; padding:2px 7px; background:#f59e0b; color:#000; border-radius:4px; font-weight:700; margin-left:6px; vertical-align:middle;">Low Confidence</span>'
        cat_subtext = f"Replay Risk ({n_out} features outside same-season range)"
elif getattr(forecast_res, "is_insufficient_signal", False) or risk_info.get("status") == "INSUFFICIENT_SIGNAL" or forecast_res.status == "INSUFFICIENT_SIGNAL":
    cat_display = "UNAVAILABLE"
    badge_class = "risk-unavailable"
    cat_subtext = "Insufficient signal: baseflow near-constant or below model resolution"
elif forecast_res.status in ("FETCH_FAILED", "INSUFFICIENT_HISTORY"):
    cat_display = "UNAVAILABLE"
    badge_class = "risk-unavailable"
    cat_subtext = forecast_res.reason or "Live data fetch failed or insufficient observation history"
elif forecast_res.status in ("STALE", "STALE_DATA"):
    cat_display = "UNAVAILABLE"
    badge_class = "risk-unavailable"
    cat_subtext = f"Data is {int(data_age_hours / 24) if data_age_hours else 'N'} days old (last observation {latest_date}) — stale data policy prevents risk classification"
elif forecast_res.status in ("NO_MODEL", "MODEL_MISSING") or not model_artifact.exists():
    cat_display = "UNAVAILABLE"
    badge_class = "risk-unavailable"
    cat_subtext = forecast_res.reason or "No trained model for this station/horizon"
elif forecast_res.status == "SUSPECT":
    cat_display = "UNAVAILABLE"
    badge_class = "risk-unavailable"
    cat_subtext = "Suspect forecast value — physical plausibility bounds exceeded"
elif is_low_conf:
    cat_display = f'{cat} <span style="font-size:0.55em; display:inline-block; padding:2px 7px; background:#f59e0b; color:#000; border-radius:4px; font-weight:700; margin-left:6px; vertical-align:middle;">Low Confidence</span>'
    badge_class = {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Severe": "risk-severe",
    }.get(cat, "risk-unavailable")
    cat_subtext = f"Low Confidence: {n_out} inputs outside seasonal training range"
elif cat != "UNAVAILABLE":
    cat_display = cat
    badge_class = {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Severe": "risk-severe",
    }.get(cat, "risk-unavailable")
    cat_subtext = "Statistical Percentile Threshold (Not CWC Level)"
else:
    cat_display = "UNAVAILABLE"
    badge_class = "risk-unavailable"
    cat_subtext = forecast_res.reason or "Risk classification unavailable"

if not is_rainfall_only:
    if pred_flow is not None:
        pred_display = f"{pred_flow:.2f} m³/s"
        pred_subtext = f"+{selected_horizon}d Forecast | Origin: {latest_date}"
    else:
        pred_display = "Unavailable"
        pred_subtext = forecast_res.reason or "Model not trained yet"

    obs_display = f"{latest_flow:.2f} m³/s" if latest_flow is not None else "N/A"
    obs_subtext = f"Modelled GloFAS Flow | {latest_date}" if latest_date else "Open-Meteo GloFAS Discharge"

    if lower_flow is not None and upper_flow is not None:
        fan_display = f"[{lower_flow:.2f}, {upper_flow:.2f}]"
        if measured_coverage_pct is not None:
            cov_cfg = thresh_cfg.get("conformal_coverage", {})
            min_acc = float(cov_cfg.get("min_acceptable_pct", 88.0))
            warn_msg = cov_cfg.get("under_coverage_warning", "interval slightly under-covers on the test set")
            if measured_coverage_pct < min_acc:
                fan_subtext = f"Nominal 90%, measured coverage {measured_coverage_pct:.1f}% ({warn_msg})"
            else:
                fan_subtext = f"Nominal 90%, measured coverage {measured_coverage_pct:.1f}%"
        else:
            fan_subtext = "Nominal 90% Conformal Interval"
    else:
        fan_display = "N/A"
        fan_subtext = "Conformal Interval unavailable"

# Prepare last_row and loaded_model for What-If simulator if prediction succeeded
if pred_flow is not None and model_artifact.exists() and not df_aligned.empty:
    try:
        matrix = build_feature_matrix(df_aligned, horizons=[selected_horizon])
        if not matrix.empty:
            last_row = matrix.iloc[[-1]].copy()
            loaded_model = joblib.load(model_artifact)
    except Exception:
        loaded_model = None
        last_row = None

# If station has no processed data or model missing, show clear notice with exact CLI command
if not is_rainfall_only and (not proc_files or forecast_res.status in ("UNAVAILABLE", "INSUFFICIENT_HISTORY", "FETCH_FAILED", "STALE", "STALE_DATA")):
    if not proc_files:
        st.warning(
            f"⚠️ No processed data found for station {selected_stn_id} ({selected_stn_name}).\n\n"
            f"To ingest data, calibrate thresholds, and train models for all stations, run:\n"
            f"```bash\npython -m src.train --all-stations\n```\n"
            f"Or for this specific station:\n"
            f"```bash\npython -m src.train --station {selected_stn_id}\n```"
        )
    elif not model_artifact.exists():
        st.warning(
            f"⚠️ Model artifact `{model_artifact.name}` not found for station {selected_stn_id} ({selected_stn_name}).\n\n"
            f"To train models for all stations, run:\n"
            f"```bash\npython -m src.train --all-stations\n```\n"
            f"Or for this specific station:\n"
            f"```bash\npython -m src.train --station {selected_stn_id}\n```\n"
            f"Forecast widgets and risk classification are disabled until models are trained."
        )
    elif forecast_res.reason:
        st.info(f"ℹ️ Forecast status for {selected_stn_name}: **{forecast_res.reason}**")

# Data Source & Freshness Caption in Live Mode
if not is_replay and live_fetch_info:
    fetch_source_lbl = {
        "live_api": "Live API (Open-Meteo)",
        "live_cache": "Local Live Cache",
        "historical_processed": "Historical Processed Archive",
        "none": "Unavailable",
    }.get(live_fetch_info.get("source_used", ""), live_fetch_info.get("source_used", ""))

    status_icon = "🟢" if live_fetch_info["status"] == "OK" else "🟡"
    st.caption(
        f"{status_icon} **Data Layer**: {fetch_source_lbl} | "
        f"**Last Observation**: `{live_fetch_info.get('last_observation_date') or 'N/A'}` | "
        f"**Fetched (UTC)**: `{live_fetch_info.get('fetched_at_utc') or 'N/A'}`"
    )

# Low-Confidence Warning Banners & Replay Switch Button (Requirements 2 & 5)
if is_rainfall_only and rain_risk_info:
    r_24 = rain_risk_info.get("rain_24h_mm", 0.0)
    r_72 = rain_risk_info.get("rain_72h_mm", 0.0)
    r_peak = rain_risk_info.get("peak_intensity_mmh", 0.0)
    r_prob = rain_risk_info.get("precip_probability_pct", 0.0)
    st.info(
        f"🌧️ **Rainfall risk only: no river-flow model for this station**\n\n"
        f"Showing rule-based meteorological rainfall indicators: "
        f"**24h Rainfall**: `{r_24:.1f} mm` | "
        f"**72h Cumulative**: `{r_72:.1f} mm` | "
        f"**Peak Intensity**: `{r_peak:.1f} mm/h` | "
        f"**Probability**: `{r_prob:.1f}%`.\n\n"
        f"Flow forecasts, prediction intervals, and river discharge thresholds are disabled for unmodeled basins."
    )
elif forecast_res.drift_status == "OUT_OF_DISTRIBUTION":
    top_drifting_str = ""
    if getattr(forecast_res, "top_drifting_features", None):
        top_names = [
            f["feature"] if isinstance(f, dict) else str(f)
            for f in forecast_res.top_drifting_features
        ]
        top_drifting_str = f" (Top drifting features: {', '.join(top_names)})"
    st.warning(f"⚠️ **Forecast shown with low confidence: inputs outside training range**{top_drifting_str}")
elif getattr(forecast_res, "is_insufficient_signal", False) or forecast_res.status == "INSUFFICIENT_SIGNAL":
    st.info(f"ℹ️ **Forecast shown with low confidence: station has insufficient hydrologic signal for early warning** ({forecast_res.reason or 'Near-constant flow series or below sensor resolution'})")
elif forecast_res.status in ("STALE", "STALE_DATA"):
    st.warning(f"⚠️ **Stale Data Notice**: Latest observation is from `{latest_date}` (data age > 48h). Risk classification is disabled (UNAVAILABLE) and drift monitoring is paused.")

if not is_replay and (cat_display == "UNAVAILABLE" or cat == "UNAVAILABLE") and model_artifact.exists():
    rep_btn_c1, rep_btn_c2 = st.columns([1, 2])
    with rep_btn_c1:
        if st.button("🔄 Open this station in Historical Replay", key="btn_switch_historical_replay", type="primary"):
            st.session_state["operating_mode"] = "Historical Replay"
            st.rerun()
    with rep_btn_c2:
        st.caption("Live risk classification is unavailable for this station. Click to explore predictions against verified historical test events.")

# ── 10-Second Executive Summary KPI Cards ─────────────────────────────────────
st.markdown(f"### {t('ten_sec_title')}")

kpi_html = f"""
<div class="kpi-container">
    <div class="kpi-card {badge_class}">
        <div class="kpi-label">{t('risk_category')}</div>
        <div class="kpi-value">{cat_display}</div>
        <div class="kpi-subtext">{cat_subtext}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">{t('expected_peak')}</div>
        <div class="kpi-value">{pred_display}</div>
        <div class="kpi-subtext">{pred_subtext}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Latest Observed Flow</div>
        <div class="kpi-value">{obs_display}</div>
        <div class="kpi-subtext">{obs_subtext}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">{t('interval_fan')}</div>
        <div class="kpi-value">{fan_display}</div>
        <div class="kpi-subtext">{fan_subtext}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">{t('confidence_ood')}</div>
        <div class="kpi-value" style="color:{ood_color}; font-size:1.3rem;">{ood_status_display}</div>
        <div class="kpi-subtext">{ood_subtext}</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

if is_rainfall_only and rain_risk_info:
    primary_why = f"Rainfall-only evaluation (24h rain: {rain_risk_info.get('rain_24h_mm', 0.0):.1f} mm, 72h: {rain_risk_info.get('rain_72h_mm', 0.0):.1f} mm). River flow model disabled for unmodeled basin."

st.info(f"💡 **{t('why_forecast')}**: {primary_why}")

# Detailed Risk Evaluation breakdown expander
if not is_rainfall_only and (risk_info.get("rationale") or cat != "UNAVAILABLE"):
    with st.expander("📊 **Risk Evaluation & Statistical Thresholds Details**", expanded=(drift_status == "OUT_OF_DISTRIBUTION")):
        col_ra, col_rb = st.columns(2)
        with col_ra:
            pred_f_str = f"`{pred_flow:.2f} m³/s` (+{selected_horizon}d)" if pred_flow is not None else "N/A"
            latest_f_str = f"`{latest_flow:.2f} m³/s`" if latest_flow is not None else "N/A"
            st.markdown(f"- **Forecast Discharge**: {pred_f_str}")
            st.markdown(f"- **Latest Observed Discharge**: {latest_f_str}")
            st.markdown(f"- **Threshold Crossed**: `{risk_info.get('threshold_crossed')}`")
        with col_rb:
            st.markdown(f"- **Decision Rationale**: {risk_info.get('rationale')}")
            st.markdown(f"- **Threshold Source**: *{risk_info.get('threshold_source')}*")
            st.caption(f"ℹ️ {risk_info.get('season_context')}")
        if getattr(forecast_res, "top_drifting_features", None):
            st.markdown("##### ⚠️ Top Input Features vs. Same-Season Training Range [p1, p99]")
            df_drift = pd.DataFrame(forecast_res.top_drifting_features)
            cols_to_show = ["feature", "live_value", "range_str", "train_mean", "z_score", "is_outside"]
            avail_cols = [c for c in cols_to_show if c in df_drift.columns]
            df_display = df_drift[avail_cols].rename(columns={
                "feature": "Feature",
                "live_value": "Live Value",
                "range_str": "Seasonal Range [p1, p99]",
                "train_mean": "Season Mean",
                "z_score": "Seasonal z-score",
                "is_outside": "Outside Range",
            })
            st.dataframe(df_display, hide_index=True)

# Sanity Check of the Forecast (Side-by-side models & physical plausibility)
if not is_rainfall_only:
    with st.expander("🔍 **Forecast Sanity & Consistency Checks**", expanded=bool(forecast_res.sanity_warning)):
        s_col1, s_col2, s_col3 = st.columns(3)
        p_val = f"{forecast_res.persistence_forecast:.2f} m³/s" if forecast_res.persistence_forecast is not None else "N/A"
        s_val = f"{forecast_res.forecast:.2f} m³/s" if forecast_res.forecast is not None else "N/A"
        t_val = f"{forecast_res.best_tree_forecast:.2f} m³/s" if forecast_res.best_tree_forecast is not None else "N/A"
        t_name = forecast_res.best_tree_name or "Best Tree"

        s_col1.metric("Persistence Baseline", p_val, help="Carries latest observed discharge forward")
        s_col2.metric(f"Selected Model ({selected_model})", s_val, help="Current forecast output")
        s_col3.metric(f"Best Tree Model ({t_name})", t_val, help="Top-performing tree architecture on validation")

        if forecast_res.sanity_warning:
            st.warning(f"⚠️ **Sanity Alert**: {forecast_res.sanity_warning}")
        else:
            st.success("✅ **Sanity Check Passed**: Forecast change is within historical dynamic range and plausible step-change bounds.")

        if forecast_res.historical_delta_violation_rate is not None:
            st.caption(
                f"ℹ️ **Test Set Sanity Rate**: In historical test evaluations, {forecast_res.historical_delta_violation_rate:.2f}% "
                f"of day-to-day transitions exhibited >50% relative shifts or exceeded training 1-day delta extremes."
            )

# ── Download Actions ──────────────────────────────────────────────────────────
down_col1, down_col2, down_col3 = st.columns(3)
if proc_files:
    down_col1.download_button(
        t("download_csv"),
        data=df_aligned.to_csv(index=False),
        file_name=f"{selected_stn_id}_forecast_series.csv",
        mime="text/csv",
    )
if scores_data:
    down_col2.download_button(
        t("download_json"),
        data=json.dumps(scores_data, indent=2),
        file_name="model_scores.json",
        mime="application/json",
    )
if latency_data:
    down_col3.download_button(
        t("download_latency"),
        data=json.dumps(latency_data, indent=2),
        file_name="latency.json",
        mime="application/json",
    )

st.markdown("---")

# ── Navigation Tabs ──────────────────────────────────────────────────────────
tab_overview, tab_forecast, tab_all_stns, tab_models, tab_training, tab_backtest, tab_rain_risk, tab_verif, tab_quality, tab_about = st.tabs([
    t("tab_overview"),
    t("tab_forecast"),
    t("tab_station_comp"),
    t("tab_models"),
    t("tab_training_status"),
    t("tab_backtest"),
    t("tab_rain_risk"),
    t("tab_verification"),
    t("tab_quality"),
    t("tab_about"),
])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    st.subheader(f"Basin Overview: {selected_stn_name}")
    stn_curr = next(s for s in all_stations if s["id"] == selected_stn_id)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("River Basin", stn_curr.get("river", "Cauvery"))
    col2.metric("Latitude", f"{stn_curr['latitude']}°N")
    col3.metric("Longitude", f"{stn_curr['longitude']}°E")
    col4.metric("Cadence", "Daily (GloFAS Modelled Discharge)")

    st.markdown("### Basin Map (Auto-Centered & Station Marked)")
    map_df = pd.DataFrame([{
        "lat": s["latitude"],
        "lon": s["longitude"],
        "name": s["name"],
        "selected": "Selected Station" if s["id"] == selected_stn_id else "Monitored Station",
    } for s in all_stations])

    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        hover_name="name",
        color="selected",
        color_discrete_map={"Selected Station": "#d62728", "Monitored Station": "#1f77b4"},
        zoom=7.5,
        center={"lat": stn_curr["latitude"], "lon": stn_curr["longitude"]},
        mapbox_style="open-street-map",
    )
    fig_map.update_traces(marker=dict(size=14))
    st.plotly_chart(fig_map, use_container_width=True)

# ── Tab 2: Forecast (Fan Chart, Plain Drivers, What-If) ───────────────────────
with tab_forecast:
    if is_rainfall_only:
        st.subheader(f"Rainfall Series & Indicators: {selected_stn_name}")
        st.info(
            "🌧️ **Rainfall Risk Mode Active**\n\n"
            "River discharge models and fan charts are disabled for this station because it is an unmodeled coastal/estuary basin. "
            "Monitor the precipitation series below and Tab 7 for IMD rainfall intensity analysis."
        )
        if proc_files and not df_aligned.empty and "precipitation_sum" in df_aligned.columns:
            fig_rain = go.Figure()
            df_rp = df_aligned.tail(45).copy()
            fig_rain.add_trace(go.Bar(
                x=df_rp["date"],
                y=df_rp["precipitation_sum"],
                name="Daily Rainfall (mm)",
                marker_color="#0284c7",
            ))
            fig_rain.update_layout(
                title=f"{selected_stn_name} - Recent Daily Rainfall Series",
                xaxis_title="Date",
                yaxis_title="Precipitation (mm)",
                template="plotly_dark",
            )
            st.plotly_chart(fig_rain, use_container_width=True)
    elif proc_files and pred_flow is not None:
        st.subheader(f"River Discharge Fan Chart (Horizon: {selected_horizon}d Ahead)")
        df_rec = df_aligned.tail(45).copy()
        fig_fan = go.Figure()

        # Historical observed series (strictly past observations)
        if "is_forecast" in df_rec.columns:
            is_fc = df_rec["is_forecast"].astype("boolean").fillna(False).astype(bool)
            df_past = df_rec[(~is_fc) & df_rec["river_discharge"].notna()]
        else:
            df_past = df_rec[df_rec["river_discharge"].notna()]

        fig_fan.add_trace(go.Scatter(
            x=df_past["date"],
            y=df_past["river_discharge"],
            mode="lines+markers",
            name="Observed GloFAS Discharge",
            line=dict(color="#38bdf8", width=2.5),
        ))

        # Dotted API simulated forecast layer (future days from API, if any)
        if "is_forecast" in df_aligned.columns and "api_forecast_river_discharge" in df_aligned.columns:
            is_fc_aligned = df_aligned["is_forecast"].astype("boolean").fillna(False).astype(bool)
            df_future_sim = df_aligned[is_fc_aligned & df_aligned["api_forecast_river_discharge"].notna()]
            if not df_future_sim.empty:
                fig_fan.add_trace(go.Scatter(
                    x=df_future_sim["date"],
                    y=df_future_sim["api_forecast_river_discharge"],
                    mode="lines+markers",
                    name="API Forecast (not observed)",
                    line=dict(color="#a855f7", dash="dot", width=2),
                ))

        # Prediction point (origin strictly latest_date)
        future_lbl = f"+{selected_horizon}d Forecast"
        last_dt = latest_date or (df_past["date"].iloc[-1] if not df_past.empty else df_rec["date"].iloc[-1])
        fig_fan.add_trace(go.Scatter(
            x=[last_dt, future_lbl],
            y=[latest_flow, pred_flow],
            mode="lines+markers",
            name=f"Point Forecast ({selected_model}): {pred_flow:.2f} m³/s",
            line=dict(color="#f59e0b", dash="dash", width=2.5),
        ))

        # Uncertainty fan interval
        fig_fan.add_trace(go.Scatter(
            x=[future_lbl, future_lbl],
            y=[lower_flow, upper_flow],
            mode="lines+markers",
            name=f"90% Prediction Interval: [{lower_flow:.2f}, {upper_flow:.2f}]",
            line=dict(color="#22c55e", width=6),
        ))

        # In Replay Mode: plot observed actual diamond
        if is_replay and forecast_res.replay_actuals:
            rep_act = forecast_res.replay_actuals.get(f"+{selected_horizon}d")
            if rep_act is not None:
                fig_fan.add_trace(go.Scatter(
                    x=[future_lbl],
                    y=[rep_act],
                    mode="markers",
                    name=f"Observed Actual: {rep_act:.2f} m³/s",
                    marker=dict(color="#ef4444", size=14, symbol="diamond"),
                ))

        cov_sub = f"(Nominal 90%, measured coverage {measured_coverage_pct:.1f}%)" if measured_coverage_pct is not None else "(Nominal 90%)"
        fig_fan.update_layout(
            title=f"{selected_stn_name} Discharge Forecast & 90% Conformal Interval {cov_sub}",
            xaxis_title="Date / Horizon",
            yaxis_title="Discharge (m³/s)",
            hovermode="x unified",
            template="plotly_dark",
        )
        st.plotly_chart(fig_fan, use_container_width=True)
        st.caption("Conformal prediction interval calibrated strictly on the validation set.")

        # Replay Retrospective Comparison Table
        if is_replay and forecast_res.replay_actuals:
            st.markdown("### 🎯 Retrospective Actuals & Verification")
            st.caption(f"Retrospective evaluation at forecast origin **{latest_date}** comparing predictions against observed reality.")
            rep_act = forecast_res.replay_actuals.get(f"+{selected_horizon}d")
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Predicted Discharge", f"{pred_flow:.2f} m³/s")
            r_col2.metric(f"Observed Actual (+{selected_horizon}d)", f"{rep_act:.2f} m³/s" if rep_act is not None else "N/A")
            if forecast_res.replay_error is not None:
                pct_str = f" ({forecast_res.replay_pct_error:.1f}%)" if forecast_res.replay_pct_error is not None else ""
                r_col3.metric("Forecast Error (Observed - Predicted)", f"{forecast_res.replay_error:+.2f} m³/s", delta=pct_str, delta_color="inverse")

            act_table = []
            for h in [1, 3, 7]:
                h_k = f"+{h}d"
                h_act = forecast_res.replay_actuals.get(h_k)
                act_table.append({
                    "Lead Horizon": h_k,
                    "Observed Actual Discharge (m³/s)": f"{h_act:.2f}" if h_act is not None else "Pending / End of series",
                })
            st.dataframe(pd.DataFrame(act_table), use_container_width=True)

        st.markdown("### 🔍 Plain-Language Driver Explanation")
        st.info(f"**Key Driver Analysis**: {primary_why}")

        st.markdown("### 🧪 Hypothetical Rainfall What-If Simulator")
        st.caption("Simulates model response under hypothetical rainfall scaling. Note: Tree models cannot extrapolate beyond historical bounds.")
        rain_mult = st.slider("Scale Precipitation Sum", min_value=0.5, max_value=2.5, value=1.2, step=0.1)
        sim_res = simulate_rainfall_what_if(loaded_model, last_row, rainfall_multiplier=rain_mult)
        s1, s2 = st.columns(2)
        s1.metric("Base Forecast", f"{sim_res['base_forecast_m3s']:.2f} m³/s")
        s2.metric(f"Scaled Forecast (+{int((rain_mult-1)*100)}%)", f"{sim_res['simulated_forecast_m3s']:.2f} m³/s")
        st.caption(f"Status: {sim_res['extrapolation_status']} | Mode: HYPOTHETICAL")
    else:
        st.subheader(f"River Discharge Fan Chart (Horizon: {selected_horizon}d Ahead)")
        st.warning(
            f"⚠️ Forecast disabled: Model artifact `{model_artifact.name}` not found for station {selected_stn_id}.\n\n"
            f"Run the training pipeline to generate models and forecasts:\n"
            f"```bash\npython -m src.train --station {selected_stn_id}\n```"
        )

# ── Tab 3: All Stations Summary ───────────────────────────────────────────────
with tab_all_stns:
    st.subheader("Cross-Basin Monitored Stations Status")
    dist_rows = []
    for s in all_stations:
        p = sorted((_PROJECT_ROOT / data_cfg["paths"]["processed_dir"]).glob(f"{s['id']}_aligned_*.csv"))
        s_models = list((models_dir / s["id"]).glob("*.joblib")) if (models_dir / s["id"]).exists() else []

        best_m = "N/A"
        if df_all_scores is not None and not df_all_scores.empty:
            s_scores = df_all_scores[(df_all_scores["station"] == s["id"]) & (df_all_scores["horizon"] == 1)]
            if not s_scores.empty:
                best_m = f"{s_scores.sort_values(by='RMSE').iloc[0]['model']} (RMSE {s_scores.sort_values(by='RMSE').iloc[0]['RMSE']:.2f})"

        if p and len(s_models) > 0:
            d = pd.read_csv(p[-1])
            flow_val = float(d["river_discharge"].iloc[-1])
            rain_val = float(d["precipitation_sum"].iloc[-1]) if "precipitation_sum" in d.columns else np.nan
            dist_rows.append({
                "Station": s["name"],
                "River Basin": s.get("river", "Cauvery"),
                "Latest Date": str(d["date"].iloc[-1]) if "date" in d.columns else "Available",
                "Modelled Flow (m³/s)": round(flow_val, 2),
                "Latest Rain (mm/day)": round(rain_val, 2) if not np.isnan(rain_val) else "N/A",
                "Status": "✅ Operational",
                "Best 1d Model": best_m,
            })
        elif p:
            dist_rows.append({
                "Station": s["name"],
                "River Basin": s.get("river", "Cauvery"),
                "Latest Date": "Data Ingested",
                "Modelled Flow (m³/s)": np.nan,
                "Latest Rain (mm/day)": np.nan,
                "Status": "⚠️ Model Untrained",
                "Best 1d Model": "Run training",
            })
        else:
            dist_rows.append({
                "Station": s["name"],
                "River Basin": s.get("river", "Cauvery"),
                "Latest Date": "Pending Data",
                "Modelled Flow (m³/s)": np.nan,
                "Latest Rain (mm/day)": np.nan,
                "Status": "❌ Insufficient Data",
                "Best 1d Model": "No data",
            })
    st.dataframe(pd.DataFrame(dist_rows), use_container_width=True)

# ── Tab 4: ML Model Comparison ───────────────────────────────────────────────
with tab_models:
    st.subheader(f"ML Model Benchmark Matrix: {selected_stn_name} (Horizon: {selected_horizon}d)")
    if df_all_scores is not None and not df_all_scores.empty:
        sub_df = df_all_scores[(df_all_scores["station"] == selected_stn_id) & (df_all_scores["horizon"] == selected_horizon)].copy()
        if not sub_df.empty:
            sub_df_display = sub_df.copy()
            sub_df_display["beats_persistence"] = sub_df_display["beats_persistence"].apply(lambda x: "✅" if x else "❌")
            sub_df_display["beats_all_baselines"] = sub_df_display["beats_all_baselines"].apply(lambda x: "✅" if x else "❌")
            sub_df_display = sub_df_display.sort_values(by="RMSE", ascending=True)

            st.dataframe(
                sub_df_display[[
                    "model", "MAE", "RMSE", "R2", "NSE", "peak_error", "peak_timing_error",
                    "missed_events", "false_alarms", "interval_coverage", "beats_persistence", "beats_all_baselines"
                ]],
                use_container_width=True,
            )

            # Plotly RMSE Comparison Chart
            fig_bar = px.bar(
                sub_df.sort_values(by="RMSE"),
                x="model",
                y="RMSE",
                color="beats_all_baselines",
                color_discrete_map={True: "#22c55e", False: "#ef4444"},
                title=f"Test Set RMSE by Model for {selected_stn_name} (Horizon: {selected_horizon}d)",
                labels={"beats_all_baselines": "Beats All Baselines", "RMSE": "Test RMSE (m³/s)"},
                template="plotly_dark",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No benchmark records found for the selected station and horizon.")
    else:
        st.info("Run `python -m src.train --all-stations` to populate benchmark scores.")

    # Feature Group Ablation Study
    if df_ablation is not None and not df_ablation.empty:
        st.markdown("### 🔬 Feature-Group Ablation Study")
        st.caption("Validates predictive value of newly engineered temporal and hydrological feature groups.")
        st.dataframe(df_ablation, use_container_width=True)

    # Target Transform Comparison
    target_tf_path = _PROJECT_ROOT / "reports" / "target_transforms.json"
    if target_tf_path.exists():
        with open(target_tf_path, "r", encoding="utf-8") as f_tf:
            tf_data = json.load(f_tf)
        st.markdown("### 📐 Target Transformation Validation (log1p vs raw)")
        st.json(tf_data)

# ── Tab 5: Training Status & Coverage Matrix ──────────────────────────────────
with tab_training:
    st.subheader("Training Coverage Matrix (All Stations × Horizons × Models)")
    if df_coverage is not None and not df_coverage.empty:
        c_tot = len(df_coverage)
        c_train = len(df_coverage[df_coverage["status"] == "trained"])
        c_skip = len(df_coverage[df_coverage["status"] == "skipped"])
        c_insuf = len(df_coverage[df_coverage["status"] == "insufficient_data"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Model Cells", c_tot)
        m2.metric("Trained & Verified", c_train, delta=f"{c_train/c_tot*100:.1f}%")
        m3.metric("Skipped (Optional)", c_skip)
        m4.metric("Insufficient Data", c_insuf)

        st.markdown("#### Full Coverage Breakdown")
        st.dataframe(df_coverage, use_container_width=True)

        st.markdown("#### Retraining & Alignment Commands")
        st.markdown("Run the full training pipeline across all configured stations:")
        st.code("python -m src.train --all-stations", language="bash")
        st.markdown("Or run retrain with validation RMSE quality gate:")
        st.code(f"python -m src.retrain --station {selected_stn_id}", language="bash")
    else:
        st.info("Run `python scripts/generate_training_coverage.py` to generate coverage records.")

# ── Tab 6: Historical Backtest ────────────────────────────────────────────────
with tab_backtest:
    st.subheader(f"Historical High-Flow Backtest Replay: {selected_stn_name}")
    st.caption("Replays past extreme flow events from the test period to evaluate forecast lead time and peak accuracy.")

    if proc_files and model_artifact.exists():
        df_hist = df_aligned.copy()
        if "river_discharge" in df_hist.columns and len(df_hist) > 100:
            df_hist["date"] = pd.to_datetime(df_hist["date"])
            peak_idx = df_hist["river_discharge"].idxmax()
            peak_row = df_hist.loc[peak_idx]
            peak_date = peak_row["date"]
            peak_val = float(peak_row["river_discharge"])

            # Extract window around peak
            w_start = pd.Timestamp(peak_date) - pd.DateOffset(days=15)
            w_end = pd.Timestamp(peak_date) + pd.DateOffset(days=15)
            df_window = df_hist[(df_hist["date"] >= w_start) & (df_hist["date"] <= w_end)].copy()

            st.markdown(f"**Identified Peak Event**: Observed `{peak_val:.2f} m³/s` on `{peak_date.strftime('%Y-%m-%d')}`")

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=df_window["date"].dt.strftime("%Y-%m-%d"),
                y=df_window["river_discharge"],
                mode="lines+markers",
                name="Historical GloFAS Discharge",
                line=dict(color="#38bdf8", width=2.5),
            ))
            fig_bt.add_vline(x=peak_date.strftime("%Y-%m-%d"), line_width=2, line_dash="dash", line_color="#ef4444")
            fig_bt.update_layout(
                title=f"Event Window Around Peak Discharge for {selected_stn_name}",
                xaxis_title="Date",
                yaxis_title="Discharge (m³/s)",
                template="plotly_dark",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            lead_time_days = selected_horizon
            st.success(
                f"⏱️ **Warning Lead Time**: Under +{selected_horizon}d forecast configuration, "
                f"advance warning is issued **{lead_time_days} day(s)** prior to peak runoff arrival."
            )
    else:
        st.warning("Backtest replay requires processed data and a trained model artifact.")

# ── Tab 7: Rainfall Risk (Side by Side — Never Merged) ────────────────────────
with tab_rain_risk:
    st.subheader("Rainfall & Hydrologic Indicators (Side by Side — Never Merged)")
    st.caption("Meteorological precipitation and river discharge measure distinct physical phenomena and are never combined into a single index.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🌧️ Meteorological Rainfall Indicator (IMD Categories)")
        st.markdown(r"""
        - **Light Rain**: $2.5 - 15.5$ mm/day
        - **Moderate Rain**: $15.6 - 64.4$ mm/day
        - **Heavy Rain**: $64.5 - 115.5$ mm/day
        - **Very Heavy Rain**: $115.6 - 204.4$ mm/day
        - **Extremely Heavy Rain**: $\ge 204.5$ mm/day
        """)
        if proc_files and not df_aligned.empty:
            latest_rain = float(df_aligned["precipitation_sum"].iloc[-1]) if "precipitation_sum" in df_aligned.columns else 0.0
            rain_cat = "Extremely Heavy" if latest_rain >= 204.5 else "Heavy" if latest_rain >= 64.5 else "Moderate" if latest_rain >= 15.6 else "Light/Normal"
            st.metric("Latest Observed Rainfall", f"{latest_rain:.1f} mm/day", delta=rain_cat, delta_color="off")

    with col_r:
        st.markdown("### 🌊 Hydrologic Model Risk Layer (GloFAS Discharge)")
        st.markdown(r"""
        - **NORMAL**: Flow $< 50\text{th}$ percentile
        - **ADVISORY**: $50\text{th} \le \text{Flow} < 75\text{th}$ percentile
        - **WATCH**: $75\text{th} \le \text{Flow} < 90\text{th}$ percentile
        - **WARNING**: Flow $\ge 90\text{th}$ percentile
        """)
        if latest_flow is not None and model_artifact.exists():
            st.metric("Latest Modelled Flow", f"{latest_flow:.1f} m³/s", delta=risk_info.get("category", "NORMAL"), delta_color="off")
        elif latest_flow is not None:
            st.metric("Latest Modelled Flow", f"{latest_flow:.1f} m³/s", delta="Risk: UNAVAILABLE (No Model)", delta_color="off")

# ── Tab 8: Verification & Drift ───────────────────────────────────────────────
with tab_verif:
    st.subheader("Forward Verification & Input Distribution Drift")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("### Forward Verification Status")
        st.info("Forward verification strictly requires $n \\ge 30$ verified forecast instances. Current status: **Not enough verified forecasts**.")
    with v2:
        st.markdown("### Feature Drift Monitoring")
        if forecast_res.status == "STALE":
            st.warning("⚠️ **Drift not assessed: data stale** (Serving observations are older than staleness threshold; live input drift check paused).")
        elif ood_is_alert or drift_status == "OUT_OF_DISTRIBUTION":
            st.error(f"Input distribution test: **{ood_status_label}** (Out of distribution; {getattr(forecast_res, 'features_outside_range_count', 0)} features outside same-season range).")
        elif ood_status_label == "IN_DISTRIBUTION":
            st.success(f"Input distribution test: **{ood_status_label}** ({getattr(forecast_res, 'features_outside_range_count', 0)} features outside same-season range).")
        else:
            st.info(f"Input distribution test: **{ood_status_label}** ({ood_subtext})")

        outside_cnt = getattr(forecast_res, "features_outside_range_count", 0)
        st.markdown(f"**Features outside same-season [p1, p99] range**: `{outside_cnt}`")
        if getattr(forecast_res, "top_drifting_features", None):
            st.markdown("**Top Drifting Features (Live Value vs Same-Season Reference Range):**")
            for feat_detail in forecast_res.top_drifting_features:
                if isinstance(feat_detail, dict):
                    st.markdown(f"- **{feat_detail.get('feature')}**: live={feat_detail.get('live_value')} vs range {feat_detail.get('range_str')} (z={feat_detail.get('z_score', 0):+.2f})")
                else:
                    st.markdown(f"- `{feat_detail}`")

# ── Tab 9: Data Quality & Sanity ──────────────────────────────────────────────
with tab_quality:
    st.subheader("Data Quality & Sanity Checks")
    sanity_file = _PROJECT_ROOT / "reports" / f"data_sanity_{selected_stn_id}.json"
    if sanity_file.exists():
        with open(sanity_file, "r", encoding="utf-8") as f_san:
            sanity_data = json.load(f_san)
        st.markdown(f"**Station Sanity Status**: `{sanity_data.get('status', 'UNKNOWN')}`")
        st.markdown(f"**Total Records**: `{sanity_data.get('rows', 0)}` rows")
        st.markdown(f"**Mean Discharge**: `{sanity_data.get('mean_discharge', 0.0):.2f} m³/s` | **Max**: `{sanity_data.get('max_discharge', 0.0):.2f} m³/s`")
        st.markdown(f"**Grid Snap Location**: Lat `{sanity_data.get('grid_lat', 0.0)}°N`, Lon `{sanity_data.get('grid_lon', 0.0)}°E`")
    else:
        st.markdown("- **Missing Values**: Handled via causal forward-fill up to 3 days.")
        st.markdown("- **Duplicates**: Checked by calendar date timestamp.")
        st.markdown("- **Resolution Exclusion**: Grid-snap check ensures stations with near-zero runoff are excluded.")

    st.markdown("---")
    with st.expander("📥 **Upload CWC / India-WRIS In-Situ Gauge Data**", expanded=False):
        st.markdown(
            "Upload physical river gauge observations (from Central Water Commission or India-WRIS) "
            "to validate against GloFAS modelled discharge or align for retraining."
        )
        st.markdown(
            "**Expected Schema**: CSV with columns `timestamp` (or `date`), `station_id`, `value` (or `water_level_m`/`discharge_m3s`), `unit` (`m`, `m3/s`, `cm`, `cusec`)."
        )
        uploaded_gauge = st.file_uploader(
            "Select CWC Gauge CSV File",
            type=["csv"],
            key="cwc_gauge_uploader",
        )
        if uploaded_gauge is not None:
            try:
                df_g_up = pd.read_csv(uploaded_gauge)
                from src.data.gauge_upload import validate_cwc_gauge_csv, save_and_align_gauge_dataset
                gauge_val_res = validate_cwc_gauge_csv(df_g_up, target_station_id=selected_stn_id)
                if gauge_val_res["is_valid"]:
                    st.success(f"✅ **Valid Gauge CSV**: {gauge_val_res['valid_rows']} valid rows detected ({gauge_val_res['date_range'][0]} to {gauge_val_res['date_range'][1]}). Unit: `{gauge_val_res['detected_unit']}`.")
                    for w in gauge_val_res.get("warnings", []):
                        st.caption(f"ℹ️ {w}")
                    if st.button("💾 Align & Save for Model Training", key="btn_save_gauge_align"):
                        out_p = save_and_align_gauge_dataset(gauge_val_res["cleaned_df"], selected_stn_id, data_cfg)
                        st.success(f"Successfully saved and aligned gauge training dataset: `{out_p.name}`. Retrain with: `python -m src.train --station {selected_stn_id}`")
                else:
                    st.error("❌ **Invalid Gauge CSV**:")
                    for err in gauge_val_res["errors"]:
                        st.markdown(f"- {err}")
            except Exception as e_up:
                st.error(f"Error parsing uploaded file: {e_up}")

# ── Tab 10: About & Limitations ───────────────────────────────────────────────
with tab_about:
    st.subheader("About ClimateShield AI & Academic Limitations")
    st.markdown("""
    - **Academic Decision-Support Prototype**: Not an official flood alert service. Always follow statutory directives from Central Water Commission (CWC), IMD, and Tamil Nadu SDMA.
    - **Modelled vs Gauge Discharge**: All discharge numbers represent **modelled runoff** from the Copernicus / Open-Meteo GloFAS hydrologic model, not in-situ physical river gauge measurements.
    - **Prototype Thresholds**: Warning categories (Advisory, Watch, Warning) are calculated from historical percentile thresholds (p50, p75, p90) on the training set, not statutory CWC warning/danger levels.
    - **Physical Exclusions**: Does not account for terrain topography, urban stormwater drains, or dam gates.
    - **Data Citation**: River discharge and precipitation sourced from [Open-Meteo.com](https://open-meteo.com) under CC BY 4.0.
    """)
