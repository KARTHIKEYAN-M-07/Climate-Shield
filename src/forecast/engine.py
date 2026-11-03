"""src/forecast/engine.py
───────────────────────
Centralized inference, conformal prediction interval, risk evaluation, and drift monitoring engine.

Features:
  1. Standardized ForecastResult dataclass with guaranteed dictionary keys.
  2. Mode Support:
       - 'live': Ingests near-real-time observations with staleness monitoring.
                 Strictly uses PAST observations (date <= today, Asia/Kolkata) for forecast origin.
       - 'historical_replay': Retrospective evaluation strictly bounded to the TEST split.
  3. Station Signal Check:
       - Applies station_signal_policy from configs/thresholds.yaml (min rows, min CV, min p90, max zero share).
       - Excludes unmodeled, trickle, or near-constant stations with status="INSUFFICIENT_SIGNAL" and risk="UNAVAILABLE".
  4. Multi-Feature Drift Engine & Top 3 Drifting Features:
       - Computes standardized feature z-scores and 5-bin window PSI on continuous non-cyclical features.
       - If drift is OUT_OF_DISTRIBUTION, policy sets risk to "UNAVAILABLE" and surfaces top 3 drifting features.
  5. Sanity Check of the Forecast:
       - Side-by-side comparison across Persistence, Selected Model, and Best Tree Model.
       - Evaluates 1-day step-change against historical maximum 1-day delta and 50% relative delta.
  6. Historical Replay Actuals Verification:
       - Extracts observed reality at +1d, +3d, +7d and computes absolute and percentage forecast error.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.features.temporal_features import build_feature_matrix, get_feature_columns
from src.monitoring.drift import compute_drift_status
from src.risk.risk_engine import evaluate_risk
from src.risk.safety_guards import explain_prediction
from src.utils.config import load_data_config, load_thresholds_config

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
KOLKATA_TZ = timezone(timedelta(hours=5, minutes=30))

FLAG_COLUMNS = [
    "precipitation_sum_ffill_flag",
    "precipitation_sum_gap_flag",
    "precipitation_sum_neg_flag",
    "precipitation_sum_extreme_flag",
    "river_discharge_ffill_flag",
    "river_discharge_gap_flag",
    "river_discharge_neg_flag",
    "river_discharge_extreme_flag",
]


@dataclass
class ForecastResult:
    """Strongly typed forecast result container."""

    forecast: Optional[float]
    interval: Optional[Tuple[float, float]]
    risk: Dict[str, Any]
    status: str  # 'OK' | 'LOW_CONFIDENCE' | 'STALE' | 'NO_MODEL' | 'INSUFFICIENT_SIGNAL' | 'FETCH_FAILED' | 'SUSPECT' | 'UNAVAILABLE' | 'ERROR'
    drift_status: str  # Strictly: 'IN_DISTRIBUTION' | 'OUT_OF_DISTRIBUTION' | 'UNKNOWN'
    drift_value: Optional[float]
    last_observation_time: Optional[str]
    data_age_hours: Optional[float]
    warnings: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    measured_coverage: Optional[float] = None
    primary_why: Optional[str] = None
    forecast_origin_date: Optional[str] = None
    source_used: str = "local_processed_aligned"

    # Station signal check
    is_insufficient_signal: bool = False
    is_rainfall_only: bool = False
    rainfall_risk: Optional[Dict[str, Any]] = None

    # Drift diagnostics
    top_drifting_features: Optional[List[Dict[str, Any]]] = None
    features_outside_range_count: int = 0

    # Sanity check fields
    persistence_forecast: Optional[float] = None
    best_tree_forecast: Optional[float] = None
    best_tree_name: Optional[str] = None
    sanity_warning: Optional[str] = None
    historical_delta_violation_rate: Optional[float] = None

    # Replay mode evaluation fields
    replay_actuals: Optional[Dict[str, Optional[float]]] = None
    replay_error: Optional[float] = None
    replay_pct_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        """Allow subscript access for backwards compatibility."""
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        """Safe dict-like get method."""
        return getattr(self, item, default)


def get_test_period_bounds(station_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return the start and end dates of the test period for a station."""
    data_cfg = load_data_config()
    proc_files = sorted(
        (_PROJECT_ROOT / data_cfg["paths"]["processed_dir"]).glob(f"{station_id}_aligned_*.csv")
    )
    if not proc_files:
        return None, None
    try:
        df = pd.read_csv(proc_files[-1])
        if "date" not in df.columns or len(df) < 20:
            return None, None
        test_start_idx = int(len(df) * 0.85)
        test_df = df.iloc[test_start_idx:]
        return str(test_df["date"].iloc[0]), str(test_df["date"].iloc[-1])
    except Exception:
        return None, None


def get_highest_flow_test_date(station_id: str) -> Optional[str]:
    """Return the date of the highest discharge event strictly within the test period."""
    data_cfg = load_data_config()
    proc_files = sorted(
        (_PROJECT_ROOT / data_cfg["paths"]["processed_dir"]).glob(f"{station_id}_aligned_*.csv")
    )
    if not proc_files:
        return None
    try:
        df = pd.read_csv(proc_files[-1])
        if "river_discharge" not in df.columns or "date" not in df.columns or len(df) < 20:
            return None
        test_start_idx = int(len(df) * 0.85)
        test_df = df.iloc[test_start_idx:].dropna(subset=["river_discharge"])
        if test_df.empty:
            return None
        peak_idx = test_df["river_discharge"].idxmax()
        return str(test_df.loc[peak_idx]["date"])
    except Exception:
        return None


def evaluate_station_signal(
    df: pd.DataFrame,
    station_id: str,
    thresh_cfg: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Evaluate whether station satisfies hydrologic signal criteria.

    Criteria configured in configs/thresholds.yaml:
      - min_rows: >= 500
      - min_cv: std / mean >= 0.80
      - min_p90_m3s: p90 >= 0.20 m³/s
      - max_near_zero_share: days <= 0.01 m³/s <= 20%
    """
    sig_policy = thresh_cfg.get("station_signal_policy", {
        "min_rows": 500,
        "min_cv": 0.80,
        "min_p90_m3s": 0.20,
        "max_near_zero_share": 0.20,
    })

    if df.empty or "river_discharge" not in df.columns:
        return False, "No discharge data available for station"

    n_train = int(len(df) * 0.70)
    q_tr = df["river_discharge"].iloc[:n_train].dropna()
    n_rows = len(q_tr)

    if n_rows < sig_policy.get("min_rows", 500):
        return False, f"Insufficient records ({n_rows} < {sig_policy.get('min_rows', 500)})"

    m_tr = float(q_tr.mean())
    s_tr = float(q_tr.std())
    cv_tr = s_tr / m_tr if m_tr > 1e-6 else 0.0
    p90_tr = float(q_tr.quantile(0.90))
    zero_share = float((q_tr <= 0.01).mean())

    if cv_tr < sig_policy.get("min_cv", 0.80):
        return False, f"Near-constant flow series (CV={cv_tr:.2f} < {sig_policy.get('min_cv', 0.80):.2f})"
    if p90_tr < sig_policy.get("min_p90_m3s", 0.20):
        return False, f"Flow magnitude below resolution (p90={p90_tr:.2f} < {sig_policy.get('min_p90_m3s', 0.20):.2f} m³/s)"
    if zero_share > sig_policy.get("max_near_zero_share", 0.20):
        return False, f"Predominantly dry bed ({zero_share * 100:.1f}% > {sig_policy.get('max_near_zero_share', 0.20) * 100:.0f}% days <= 0.01 m³/s)"

    return True, None


def generate_forecast(
    station_id: str,
    horizon: int,
    model_name: str,
    df_data: Optional[pd.DataFrame] = None,
    mode: str = "live",  # 'live' | 'historical_replay'
    replay_origin_date: Optional[str] = None,
    max_data_stale_hours: float = 48.0,
    max_multiplier_threshold: float = 5.0,
) -> ForecastResult:
    """Generate forecast, conformal interval, risk evaluation, and drift status.

    Handles both 'live' and 'historical_replay' modes.
    Never raises an uncaught exception.
    """
    error_log_path = _PROJECT_ROOT / "reports" / "app_errors.log"
    data_cfg = load_data_config()
    thresh_cfg = load_thresholds_config()
    models_dir = _PROJECT_ROOT / data_cfg["paths"]["models_dir"]

    now_utc = datetime.now(timezone.utc)
    today_kolkata_str = datetime.now(KOLKATA_TZ).strftime("%Y-%m-%d")

    warnings: List[str] = []

    try:
        # 1. Load full station dataset
        source_used = "local_processed_aligned"
        if df_data is None:
            if mode == "live":
                cache_dir = _PROJECT_ROOT / "data" / "live_cache"
                cached_live = sorted(cache_dir.glob(f"{station_id}_*.csv")) if cache_dir.exists() else []
                if cached_live:
                    df_full = pd.read_csv(cached_live[-1])
                    source_used = "live_cache"
                else:
                    proc_files = sorted(
                        (_PROJECT_ROOT / data_cfg["paths"]["processed_dir"]).glob(f"{station_id}_aligned_*.csv")
                    )
                    if not proc_files:
                        return ForecastResult(
                            forecast=None,
                            interval=None,
                            risk={"category": "UNAVAILABLE", "status": "NO_DATA"},
                            status="UNAVAILABLE",
                            drift_status="UNKNOWN",
                            drift_value=None,
                            last_observation_time=None,
                            data_age_hours=None,
                            warnings=["No processed data found for station"],
                            reason="No processed data file found",
                            source_used="none",
                        )
                    df_full = pd.read_csv(proc_files[-1])
            else:
                proc_files = sorted(
                    (_PROJECT_ROOT / data_cfg["paths"]["processed_dir"]).glob(f"{station_id}_aligned_*.csv")
                )
                if not proc_files:
                    return ForecastResult(
                        forecast=None,
                        interval=None,
                        risk={"category": "UNAVAILABLE", "status": "NO_DATA"},
                        status="UNAVAILABLE",
                        drift_status="UNKNOWN",
                        drift_value=None,
                        last_observation_time=None,
                        data_age_hours=None,
                        warnings=["No processed data found for station"],
                        reason="No processed data file found",
                        source_used="none",
                    )
                df_full = pd.read_csv(proc_files[-1])
        else:
            df_full = df_data.copy()

        # 2. Basic dataset sanity
        if df_full.empty:
            return ForecastResult(
                forecast=None,
                interval=None,
                risk={"category": "UNAVAILABLE", "status": "FETCH_FAILED"},
                status="FETCH_FAILED",
                drift_status="UNKNOWN",
                drift_value=None,
                last_observation_time=None,
                data_age_hours=None,
                warnings=["Dataset is empty"],
                reason="Dataset is empty or live fetch failed",
                source_used="local_processed_aligned",
            )

        if "river_discharge" not in df_full.columns or df_full["river_discharge"].dropna().empty:
            from src.risk.rainfall_risk import compute_rainfall_risk_indicator
            rain_indicator = compute_rainfall_risk_indicator(df_full)
            return ForecastResult(
                forecast=None,
                interval=None,
                risk={
                    "category": "UNAVAILABLE",
                    "rainfall_category": rain_indicator["category"],
                    "status": "RAINFALL_ONLY",
                    "rationale": rain_indicator["rationale"],
                    "disclaimer": rain_indicator["disclaimer"],
                },
                status="RAINFALL_ONLY",
                drift_status="UNKNOWN",
                drift_value=None,
                last_observation_time=str(df_full["date"].dropna().iloc[-1]) if "date" in df_full.columns and not df_full["date"].dropna().empty else None,
                data_age_hours=None,
                warnings=["River flow model unsupported for coastal grid; rule-based rainfall risk active"],
                reason="Rainfall risk only: no river-flow model for this station",
                source_used="local_processed_aligned",
                is_insufficient_signal=True,
                is_rainfall_only=True,
                rainfall_risk=rain_indicator,
            )

        # 3. Station Signal Check (Identically applied across all stations)
        is_signal_ok, signal_reason = evaluate_station_signal(df_full, station_id, thresh_cfg)
        if not is_signal_ok:
            # Insufficient signal: cannot provide reliable early warning
            warnings.append(f"Insufficient station signal: {signal_reason}")

        # 4. Tag forecast rows and slice data strictly to PAST observations (today is incomplete: date >= today is forecast)
        if "is_forecast" not in df_full.columns and "date" in df_full.columns:
            df_full["is_forecast"] = (df_full["date"] >= today_kolkata_str).astype("boolean").astype(bool)
        elif "is_forecast" in df_full.columns:
            df_full["is_forecast"] = df_full["is_forecast"].astype("boolean").fillna(False).astype(bool)

        # Fill cleaning flag columns if absent or NaN
        for fc in FLAG_COLUMNS:
            if fc not in df_full.columns:
                df_full[fc] = 0.0
            else:
                df_full[fc] = pd.to_numeric(df_full[fc], errors="coerce").fillna(0.0)

        min_required_past_days = int(thresh_cfg.get("drift_risk_policy", {}).get("min_required_past_days", 30))

        if mode == "historical_replay":
            if not replay_origin_date:
                replay_origin_date = get_highest_flow_test_date(station_id) or str(df_full["date"].iloc[-1])
            forecast_origin = replay_origin_date
            df_sliced = df_full[df_full["date"] <= forecast_origin].copy()
        else:
            # LIVE MODE: Strictly PAST complete observations only (date < today_kolkata_str)
            is_fc = df_full["is_forecast"].astype("boolean").fillna(False).astype(bool)
            past_df = df_full[(~is_fc) & (df_full["date"] < today_kolkata_str)].copy()
            if past_df.empty:
                past_df = df_full[df_full["date"] < today_kolkata_str].copy()
            df_sliced = past_df.copy()
            forecast_origin = str(df_sliced["date"].iloc[-1]) if not df_sliced.empty and "date" in df_sliced.columns else None

        if df_sliced.empty or len(df_sliced) < min_required_past_days:
            missing_count = max(0, min_required_past_days - len(df_sliced))
            reason = f"Insufficient history: {len(df_sliced)} past days available, required {min_required_past_days} (missing {missing_count} days)"
            return ForecastResult(
                forecast=None,
                interval=None,
                risk={
                    "category": "UNAVAILABLE",
                    "status": "INSUFFICIENT_HISTORY",
                    "rationale": f"Station {station_id} has insufficient observation history ({len(df_sliced)}/{min_required_past_days} days, missing {missing_count} days). Early warning disabled.",
                    "threshold_crossed": "None (Insufficient History)",
                    "threshold_source": "configs/thresholds.yaml (drift_risk_policy.min_required_past_days)",
                    "season_context": "Incomplete historical window",
                },
                status="INSUFFICIENT_HISTORY",
                drift_status="UNKNOWN",
                drift_value=None,
                last_observation_time=forecast_origin,
                data_age_hours=None,
                warnings=warnings + [reason],
                reason=reason,
                source_used="historical_replay" if mode == "historical_replay" else "local_processed_aligned",
                features_outside_range_count=0,
            )

        # Date-gap check: verify no missing calendar days within the feature window (last min_required_past_days)
        window_df = df_sliced.iloc[-min_required_past_days:].copy()
        if "date" in window_df.columns and len(window_df) > 1:
            dates = pd.to_datetime(window_df["date"])
            date_diffs = dates.diff().dropna()
            if (date_diffs > pd.Timedelta(days=1)).any():
                max_gap = int(date_diffs.max().days)
                gap_reason = f"Insufficient history: feature window crosses date gap ({max_gap} days missing)"
                return ForecastResult(
                    forecast=None,
                    interval=None,
                    risk={
                        "category": "UNAVAILABLE",
                        "status": "INSUFFICIENT_HISTORY",
                        "rationale": f"Station {station_id} feature window has a date gap of {max_gap} days. Early warning disabled.",
                        "threshold_crossed": "None (Date Gap in Feature Window)",
                        "threshold_source": "data continuity check",
                        "season_context": "Discontinuous feature window",
                    },
                    status="INSUFFICIENT_HISTORY",
                    drift_status="UNKNOWN",
                    drift_value=None,
                    last_observation_time=forecast_origin,
                    data_age_hours=None,
                    warnings=warnings + [gap_reason],
                    reason=gap_reason,
                    source_used="historical_replay" if mode == "historical_replay" else "local_processed_aligned",
                    features_outside_range_count=0,
                )

        last_obs_date_str = forecast_origin
        data_age_hours: Optional[float] = None
        is_stale = False

        # Staleness applies strictly to Live mode
        if mode == "live" and last_obs_date_str:
            try:
                last_dt = datetime.strptime(last_obs_date_str, "%Y-%m-%d").replace(tzinfo=KOLKATA_TZ)
                now_kolkata = now_utc.astimezone(KOLKATA_TZ)
                diff_sec = (now_kolkata - last_dt).total_seconds()
                data_age_hours = round(max(0.0, diff_sec / 3600.0), 1)
                if data_age_hours > max_data_stale_hours:
                    is_stale = True
                    warnings.append(f"Data is {int(data_age_hours / 24)} days old (last observation {last_obs_date_str})")
            except Exception:
                data_age_hours = None

        # Extract latest observed flow from the sliced past dataset
        valid_flows = df_sliced["river_discharge"].dropna()
        latest_flow = float(valid_flows.iloc[-1]) if not valid_flows.empty else 0.0

        # 5. Check model artifacts
        candidate_artifacts = [
            models_dir / station_id / f"{station_id}_{model_name}_h{horizon}d.joblib",
            models_dir / station_id / f"{model_name}_h{horizon}d.joblib",
            models_dir / f"{station_id}_{model_name}_h{horizon}d.joblib",
        ]
        model_artifact = next((p for p in candidate_artifacts if p.exists()), None)
        if not model_artifact:
            return ForecastResult(
                forecast=None,
                interval=None,
                risk={"category": "UNAVAILABLE", "status": "NO_MODEL"},
                status="NO_MODEL",
                drift_status="UNKNOWN",
                drift_value=None,
                last_observation_time=last_obs_date_str,
                data_age_hours=data_age_hours,
                warnings=warnings + ["No trained model artifact found on disk"],
                reason=f"No trained model artifact for {model_name} at horizon {horizon}d",
                source_used="local_processed_aligned",
            )

        # 6. Load model metadata & conformal parameters
        meta_candidates = [
            models_dir / station_id / f"{station_id}_{model_name}_h{horizon}d.meta.json",
            models_dir / station_id / f"{model_name}_h{horizon}d.meta.json",
            models_dir / f"{station_id}_{model_name}_h{horizon}d.meta.json",
        ]
        meta_file = next((p for p in meta_candidates if p.exists()), None)
        meta_data: Dict[str, Any] = {}
        if meta_file:
            try:
                with open(meta_file, "r", encoding="utf-8") as f_meta:
                    meta_data = json.load(f_meta)
            except Exception:
                meta_data = {}

        conformal_file = models_dir / station_id / f"{model_name}_h{horizon}d_conformal.json"
        q_hat: Optional[float] = None
        measured_coverage: Optional[float] = None
        if conformal_file.exists():
            try:
                with open(conformal_file, "r", encoding="utf-8") as f_conf:
                    cdata = json.load(f_conf)
                    q_hat = float(cdata.get("q_hat", 0.0))
                    measured_coverage = float(cdata.get("empirical_coverage_pct", 90.0))
            except Exception:
                pass
        if q_hat is None and meta_data:
            q_hat = meta_data.get("conformal_q_hat")
            measured_coverage = meta_data.get("conformal_test_coverage_pct")

        # 7. Feature Matrix Generation using sliced observations only
        matrix = build_feature_matrix(df_sliced, horizons=[horizon], include_targets=False)
        if matrix.empty:
            return ForecastResult(
                forecast=None,
                interval=None,
                risk={"category": "UNAVAILABLE", "status": "FEATURE_ERROR"},
                status="UNAVAILABLE",
                drift_status="UNKNOWN",
                drift_value=None,
                last_observation_time=last_obs_date_str,
                data_age_hours=data_age_hours,
                warnings=warnings + ["Feature generation returned no rows"],
                reason="Insufficient rows for feature engineering lags",
                source_used="local_processed_aligned",
            )

        feat_cols = get_feature_columns(matrix, horizons=[horizon])
        last_row = matrix.iloc[[-1]].copy()

        # 8. Model Execution
        loaded_model = joblib.load(model_artifact)
        if model_name == "persistence":
            X_infer = last_row[["river_discharge"]].copy()
        else:
            X_infer = last_row[feat_cols].copy()

        if hasattr(loaded_model, "feature_names_in_"):
            expected = list(loaded_model.feature_names_in_)
            X_infer = X_infer.loc[:, expected]

        raw_pred = float(loaded_model.predict(X_infer)[0])

        # 9. Sanity Guard against non-physical spikes
        training_max = float(meta_data.get("drift_reference_stats", {}).get("max", 1000.0))
        pred_flow = max(0.0, raw_pred) if np.isfinite(raw_pred) else None
        if not np.isfinite(raw_pred) or raw_pred > (training_max * max_multiplier_threshold):
            status = "SUSPECT"
            reason = f"Forecast flow ({raw_pred:.2f} m³/s) failed physical sanity bounds [0, {training_max * max_multiplier_threshold:.2f}]"
            warnings.append(reason)
        else:
            if raw_pred < 0.0:
                warnings.append(f"Model raw prediction was negative ({raw_pred:.2f} m³/s); non-negativity constraint applied (0.0 m³/s)")
            if mode == "live":
                status = "STALE" if is_stale else ("UNAVAILABLE" if not is_signal_ok else "OK")
            else:
                status = "UNAVAILABLE" if not is_signal_ok else "OK"
            reason = "Forecast computed successfully" if is_signal_ok else f"Insufficient signal: {signal_reason}"

        # 10. Conformal Interval Calculation
        if pred_flow is not None:
            if q_hat is not None and q_hat > 0:
                lower_flow = max(0.0, pred_flow - q_hat)
                upper_flow = pred_flow + q_hat
            else:
                lower_flow = max(0.0, pred_flow * 0.85)
                upper_flow = max(lower_flow, pred_flow * 1.15)
            interval: Optional[Tuple[float, float]] = (round(lower_flow, 2), round(upper_flow, 2))
        else:
            interval = None

        # 11. Multi-Feature Same-Season Drift Monitoring & Top 3 Drifting Features
        drift_policy = thresh_cfg.get("drift_risk_policy", {
            "out_of_distribution_action": "UNAVAILABLE",
            "method": "same_season_percentiles",
            "season_window_days": 30,
            "reference_percentiles": [1.0, 99.0],
            "min_features_outside_threshold": 2,
            "top_drifting_features_count": 3,
            "min_feature_std_floor": 0.05,
        })
        season_window = int(drift_policy.get("season_window_days", 30))
        ref_percentiles = drift_policy.get("reference_percentiles", [1.0, 99.0])
        p_low = float(ref_percentiles[0])
        p_high = float(ref_percentiles[1])
        min_outside_thresh = int(drift_policy.get("min_features_outside_threshold", 2))
        std_floor = float(drift_policy.get("min_feature_std_floor", 0.05))
        excluded_patterns = drift_policy.get("excluded_feature_patterns", ["_flag"])

        # Compute training feature reference statistics from training split
        n_tr = int(len(df_full) * 0.70)
        df_tr_split = df_full.iloc[:n_tr].copy()
        X_tr_matrix = build_feature_matrix(df_tr_split, horizons=[horizon])

        origin_str = last_obs_date_str or today_kolkata_str
        try:
            origin_doy = pd.to_datetime(origin_str).dayofyear
        except Exception:
            origin_doy = pd.to_datetime(today_kolkata_str).dayofyear

        # Determine same-season rows across training split
        if "date" in X_tr_matrix.columns:
            doy_series = pd.to_datetime(X_tr_matrix["date"]).dt.dayofyear
        else:
            doy_series = pd.to_datetime(df_tr_split["date"].iloc[:len(X_tr_matrix)]).dt.dayofyear

        doy_diff = (doy_series - origin_doy).abs()
        circ_diff = np.minimum(doy_diff, 365 - doy_diff)
        season_mask = circ_diff <= season_window
        X_season = X_tr_matrix[season_mask].copy()
        if len(X_season) < 30:
            X_season = X_tr_matrix.copy()

        feature_cols_to_check = [
            c for c in (list(loaded_model.feature_names_in_) if hasattr(loaded_model, "feature_names_in_") else feat_cols)
            if c not in ("source", "source_used", "fetched_at_utc", "is_forecast", "date", "date_time")
            and not any(pat in c for pat in excluded_patterns)
        ]

        feature_drift_scores: List[Dict[str, Any]] = []
        outside_features: List[Dict[str, Any]] = []

        for col in feature_cols_to_check:
            if col in X_season.columns and col in last_row.columns:
                s_series = pd.to_numeric(X_season[col], errors="coerce").dropna()
                lv_series = pd.to_numeric(last_row[col], errors="coerce").dropna()
                if s_series.empty or lv_series.empty:
                    continue
                lv_val = float(lv_series.iloc[0])
                s_vals = s_series.astype(float).values
                if len(s_vals) > 10 and np.isfinite(lv_val):
                    tr_s = float(np.std(s_vals))
                    if tr_s < 1e-4:
                        # Exclude zero or near-zero variance features from drift test
                        continue
                    tr_m = float(np.mean(s_vals))
                    tr_s_guarded = max(tr_s, std_floor)
                    p1 = float(np.percentile(s_vals, p_low))
                    p99 = float(np.percentile(s_vals, p_high))
                    min_val = float(np.min(s_vals))
                    max_val = float(np.max(s_vals))

                    is_outside = (lv_val < p1) or (lv_val > p99)
                    z_sea = float((lv_val - tr_m) / tr_s_guarded)

                    dist = 0.0
                    if lv_val < p1:
                        dist = p1 - lv_val
                    elif lv_val > p99:
                        dist = lv_val - p99

                    item = {
                        "feature": col,
                        "live_value": round(lv_val, 2),
                        "train_mean": round(tr_m, 2),
                        "train_std": round(tr_s, 2),
                        "guarded_std": round(tr_s_guarded, 2),
                        "p_low": round(p1, 2),
                        "p_high": round(p99, 2),
                        "range_str": f"[{p1:.2f}, {p99:.2f}]",
                        "min_val": round(min_val, 2),
                        "max_val": round(max_val, 2),
                        "is_outside": is_outside,
                        "violation_dist": round(dist, 2),
                        "z_score": round(z_sea, 2),
                        "abs_z": abs(z_sea),
                    }
                    feature_drift_scores.append(item)
                    if is_outside:
                        outside_features.append(item)

        # Sort: features outside range first by violation distance and |z|, then remaining by |z|
        feature_drift_scores.sort(
            key=lambda x: (1 if x["is_outside"] else 0, x["violation_dist"], x["abs_z"]),
            reverse=True,
        )
        top_drifting_features = feature_drift_scores[:int(drift_policy.get("top_drifting_features_count", 3))]
        n_outside = len(outside_features)

        # Window PSI on discharge
        serving_discharge = (
            df_sliced["river_discharge"].iloc[-30:].dropna().to_numpy(dtype=float)
            if len(df_sliced) >= 30
            else df_sliced["river_discharge"].dropna().to_numpy(dtype=float)
        )
        ref_stats = meta_data.get("drift_reference_stats")
        if not ref_stats:
            ref_file = models_dir / station_id / "drift_reference.json"
            if ref_file.exists():
                try:
                    with open(ref_file, "r", encoding="utf-8") as rf:
                        ref_stats = json.load(rf)
                except Exception:
                    ref_stats = None

        drift_info = compute_drift_status(
            serving_values=serving_discharge,
            reference_stats=ref_stats,
            feature_name="river_discharge",
        )
        base_psi = drift_info.get("psi") or 0.0

        # Determine drift status
        if mode == "live" and status == "STALE":
            drift_status = "UNKNOWN"
            drift_value = None
            warnings.append("Drift not assessed: data stale")
        elif n_outside >= min_outside_thresh:
            drift_status = "OUT_OF_DISTRIBUTION"
            drift_value = float(n_outside)
        else:
            drift_status = "IN_DISTRIBUTION"
            drift_value = float(n_outside)

        # 12. Decoupled Risk Classification & Status Assignment
        if not is_signal_ok:
            status = "INSUFFICIENT_SIGNAL"
            reason = f"Insufficient hydrologic signal: {signal_reason}"
            risk_result = {
                "category": "UNAVAILABLE",
                "status": "INSUFFICIENT_SIGNAL",
                "rationale": f"Station {station_id} has insufficient hydrologic signal: {signal_reason}. Early warning disabled.",
                "threshold_crossed": "None (Insufficient Signal)",
                "threshold_source": "configs/thresholds.yaml (station_signal_policy)",
                "season_context": "Insufficient signal",
            }
        elif status == "SUSPECT":
            reason = reason or f"Forecast flow ({pred_flow} m³/s) exceeds physical plausibility bounds"
            risk_result = {
                "category": "UNAVAILABLE",
                "status": "SUSPECT_VALUE",
                "rationale": reason,
                "threshold_crossed": "None (Suspect)",
                "threshold_source": "None",
                "season_context": "Suspect value guard triggered",
            }
        elif mode == "live" and status == "STALE":
            reason = f"Data is {int(data_age_hours / 24.0) if data_age_hours else 'N'} days old (last observation {last_obs_date_str}) — stale data policy triggered"
            risk_result = {
                "category": "UNAVAILABLE",
                "status": "STALE_DATA",
                "rationale": reason,
                "threshold_crossed": "None (Data Stale)",
                "threshold_source": "None",
                "season_context": "Stale data mode active",
            }
        elif drift_status == "OUT_OF_DISTRIBUTION":
            # Risk Policy: LOW_CONFIDENCE (model exists, data fresh, drift flagged)
            if status in ("OK", "OK_DRIFT_BLOCKED"):
                status = "LOW_CONFIDENCE"
            raw_risk = evaluate_risk(
                predicted_level=pred_flow,
                station_id=station_id,
                thresholds_cfg=thresh_cfg,
                horizon_days=horizon,
                current_level=latest_flow,
                model_exists=True,
            )
            raw_cat = raw_risk.get("category", "UNAVAILABLE")
            raw_crossed = raw_risk.get("threshold_crossed", "None")
            top_3_str = ", ".join([f"{f['feature']} (live={f['live_value']} vs {f['range_str']}, z={f['z_score']:+.1f})" for f in top_drifting_features])
            reason = (
                f"{n_outside} features outside same-season range (top: {top_drifting_features[0]['feature']} live={top_drifting_features[0]['live_value']} vs {top_drifting_features[0]['range_str']}); forecast shown with low confidence"
                if top_drifting_features
                else f"{n_outside} features outside same-season range; forecast shown with low confidence"
            )

            ood_action = str(drift_policy.get("out_of_distribution_action", "LOW_CONFIDENCE")).upper()
            if ood_action == "UNAVAILABLE":
                cat_val = "UNAVAILABLE"
                thresh_crossed_val = f"{raw_crossed} (Confidence: UNAVAILABLE due to drift)"
                status_val = "OUT_OF_DISTRIBUTION"
            else:
                cat_val = raw_cat
                thresh_crossed_val = f"{raw_crossed} (Low Confidence)"
                status_val = "LOW_CONFIDENCE"

            risk_result = {
                "category": cat_val,
                "status": status_val,
                "badge": "Low Confidence",
                "is_low_confidence": True,
                "rationale": (
                    f"Inputs out of seasonal training range (OUT_OF_DISTRIBUTION, {n_outside} features outside range [p1, p99]). "
                    f"Top drifting features: {top_3_str}. "
                    f"Statistical percentile model classified raw risk as {raw_cat} ({raw_crossed}) with low confidence."
                ),
                "threshold_crossed": thresh_crossed_val,
                "threshold_source": raw_risk.get("threshold_source", "configs/thresholds.yaml"),
                "season_context": f"Seasonal distribution drift active ({n_outside} features outside same-season [p1, p99] envelope)",
                "top_drifting_features": top_drifting_features,
            }
        elif pred_flow is not None:
            reason = f"Forecast computed successfully; inputs in distribution ({n_outside} features outside same-season range)"
            risk_result = evaluate_risk(
                predicted_level=pred_flow,
                station_id=station_id,
                thresholds_cfg=thresh_cfg,
                horizon_days=horizon,
                current_level=latest_flow,
                model_exists=True,
            )
        else:
            reason = reason or "Forecast could not be generated"
            risk_result = {
                "category": "UNAVAILABLE",
                "status": "ERROR",
                "rationale": reason,
                "threshold_crossed": "None (Error)",
                "threshold_source": "None",
                "season_context": "Prediction failed",
            }

        # 13. Sanity Check of the Forecast (Side by Side & Delta Threshold)
        persistence_forecast = latest_flow

        # Best tree model for comparison
        tree_models = ["xgboost", "random_forest", "gradient_boosting", "extra_trees"]
        best_tree_name = "random_forest"
        best_tree_pred = None

        scores_all_file = _PROJECT_ROOT / "reports" / "model_scores_all.csv"
        if scores_all_file.exists():
            try:
                df_scores = pd.read_csv(scores_all_file)
                tree_sub = df_scores[
                    (df_scores["station"] == station_id)
                    & (df_scores["horizon"] == horizon)
                    & (df_scores["model"].isin(tree_models))
                ]
                if not tree_sub.empty:
                    best_tree_name = str(tree_sub.sort_values(by="RMSE").iloc[0]["model"])
            except Exception:
                pass

        tree_artifact = models_dir / station_id / f"{best_tree_name}_h{horizon}d.joblib"
        if tree_artifact.exists():
            try:
                tree_m = joblib.load(tree_artifact)
                if hasattr(tree_m, "feature_names_in_"):
                    X_tree = last_row.loc[:, list(tree_m.feature_names_in_)]
                else:
                    X_tree = last_row[feat_cols]
                best_tree_pred = round(float(tree_m.predict(X_tree)[0]), 2)
            except Exception:
                best_tree_pred = None

        # Step delta checks
        dis_series = df_full["river_discharge"].dropna().to_numpy(dtype=float)
        n_train_d = int(len(dis_series) * 0.70)
        train_deltas = np.abs(np.diff(dis_series[:n_train_d]))
        max_train_delta = float(np.max(train_deltas)) if len(train_deltas) > 0 else 50.0

        test_deltas = np.abs(np.diff(dis_series[int(len(dis_series) * 0.85):])) if len(dis_series) > 100 else np.array([])
        historical_violation_rate = (
            round(float((test_deltas > max_train_delta).mean() * 100.0), 2)
            if len(test_deltas) > 0
            else 0.0
        )

        sanity_warning = None
        if pred_flow is not None:
            delta_from_obs = abs(pred_flow - latest_flow)
            pct_change = delta_from_obs / max(latest_flow, 1.0)
            if delta_from_obs > max_train_delta:
                sanity_warning = (
                    f"Large change: verify — forecast day-to-day delta ({delta_from_obs:.2f} m³/s) "
                    f"exceeds largest 1-day change in training ({max_train_delta:.2f} m³/s)."
                )
            elif pct_change > 0.50 and delta_from_obs > 2.0:
                sanity_warning = (
                    f"Large change: verify — forecast differs by {int(pct_change * 100)}% from latest observation "
                    f"({latest_flow:.2f} m³/s → {pred_flow:.2f} m³/s)."
                )

        # 14. Replay actuals comparison (if in historical_replay mode)
        replay_actuals = None
        replay_error = None
        replay_pct_error = None

        if mode == "historical_replay" and forecast_origin:
            try:
                origin_dt = datetime.strptime(forecast_origin, "%Y-%m-%d")
                d_target = (origin_dt + timedelta(days=horizon)).strftime("%Y-%m-%d")
                match_row = df_full[df_full["date"] == d_target]

                actual_val = None
                if not match_row.empty and "river_discharge" in match_row.columns:
                    actual_val = float(match_row["river_discharge"].iloc[0])
                    if pred_flow is not None and np.isfinite(actual_val):
                        replay_error = round(abs(pred_flow - actual_val), 2)
                        replay_pct_error = round(replay_error / max(actual_val, 0.1) * 100.0, 1)

                replay_actuals = {
                    f"+{horizon}d": round(actual_val, 2) if actual_val is not None else None,
                }
                for h_step in [1, 3, 7]:
                    dt_step = (origin_dt + timedelta(days=h_step)).strftime("%Y-%m-%d")
                    m_step = df_full[df_full["date"] == dt_step]
                    if not m_step.empty and "river_discharge" in m_step.columns:
                        replay_actuals[f"+{h_step}d"] = round(float(m_step["river_discharge"].iloc[0]), 2)
            except Exception:
                pass

        # 15. Plain-Language Hydrologic Rationale
        primary_why = f"Baseline seasonal hydrologic baseflow evaluated at origin {last_obs_date_str}."
        try:
            train_stats = {}
            for col in feat_cols:
                if col in matrix.columns:
                    s = pd.to_numeric(matrix[col], errors="coerce").dropna()
                    if not s.empty:
                        train_stats[col] = {"mean": float(s.mean()), "p90": float(s.quantile(0.90))}
            expls = explain_prediction(
                sample=last_row.to_dict(orient="records")[0],
                training_summary=train_stats,
            )
            if expls:
                primary_why = f"Forecast origin ({last_obs_date_str}): {expls[0]}"
        except Exception:
            pass

        return ForecastResult(
            forecast=round(pred_flow, 2) if pred_flow is not None else None,
            interval=interval,
            risk=risk_result,
            status=status,
            drift_status=drift_status,
            drift_value=drift_value,
            last_observation_time=last_obs_date_str,
            data_age_hours=data_age_hours,
            warnings=warnings,
            reason=reason,
            measured_coverage=measured_coverage,
            primary_why=primary_why,
            forecast_origin_date=last_obs_date_str,
            source_used="historical_replay" if mode == "historical_replay" else "local_processed_aligned",
            is_insufficient_signal=not is_signal_ok,
            top_drifting_features=top_drifting_features,
            features_outside_range_count=n_outside,
            persistence_forecast=round(persistence_forecast, 2) if persistence_forecast is not None else None,
            best_tree_forecast=best_tree_pred,
            best_tree_name=best_tree_name,
            sanity_warning=sanity_warning,
            historical_delta_violation_rate=historical_violation_rate,
            replay_actuals=replay_actuals,
            replay_error=replay_error,
            replay_pct_error=replay_pct_error,
        )

    except Exception as exc:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        tb_str = traceback.format_exc()
        with open(error_log_path, "a", encoding="utf-8") as f_err:
            f_err.write(
                f"\n[{now_utc.isoformat()}] Error generating forecast for station={station_id}, "
                f"horizon={horizon}, model={model_name}:\n{tb_str}\n"
            )

        return ForecastResult(
            forecast=None,
            interval=None,
            risk={"category": "UNAVAILABLE", "status": "SYSTEM_ERROR"},
            status="ERROR",
            drift_status="UNKNOWN",
            drift_value=None,
            last_observation_time=None,
            data_age_hours=None,
            warnings=[f"Unexpected forecast error: {exc}"],
            reason=f"System error: {exc}",
            source_used="error_handler",
        )
