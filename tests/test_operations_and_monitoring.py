"""Tests for Phase 19: Operations and Monitoring.

Tests:
1. SQLite prediction log: log prediction, log verification, require n >= 30 before computing scores
2. Drift monitoring: PSI calculation, KS-test, and freshness checks
3. Model registry: register model, hashes, promotion requiring validation win AND significance test
4. Scheduled refresh: idempotent and dry-run execution
5. Latency benchmark: execution and schema of reports/latency.json
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pytest

from src.monitoring.prediction_log import PredictionLogger
from src.monitoring.drift import compute_psi, compute_ks_test, check_data_and_model_freshness
from src.monitoring.registry import ModelRegistry
from src.monitoring.scheduler import run_pipeline_refresh
from src.monitoring.benchmark_latency import benchmark_pipeline_latency


class TestPredictionLogger:
    """Test SQLite prediction logging and verification rules."""

    def test_log_and_verification_workflow(self, tmp_path):
        db_path = str(tmp_path / "test_preds.db")
        logger = PredictionLogger(db_path=db_path)

        # Log 10 predictions
        for i in range(10):
            logger.log_prediction(
                origin_time="2024-10-01",
                target_time=f"2024-10-0{i+1}",
                station_id="cauvery_trichy",
                horizon_days=1,
                model_version="v2.0.0",
                prediction=120.0 + i,
                lower_bound=100.0,
                upper_bound=150.0,
                nominal_coverage=0.9,
                data_freshness_hours=3.5,
            )

        # Verify 5 of them
        for i in range(5):
            updated = logger.log_verification(
                station_id="cauvery_trichy",
                target_time=f"2024-10-0{i+1}",
                actual_discharge=122.0 + i,
            )
            assert updated == 1

        # With min_count=30, should return 'not enough verified forecasts'
        res = logger.get_forward_verification_metrics(
            station_id="cauvery_trichy",
            min_count=30,
        )
        assert res["status"] == "not enough verified forecasts"
        assert res["verified_count"] == 5
        assert res["required_count"] == 30
        assert res["metrics"] is None

        # With min_count=5, should successfully compute verified metrics
        res_met = logger.get_forward_verification_metrics(
            station_id="cauvery_trichy",
            min_count=5,
        )
        assert res_met["status"] == "verified"
        assert res_met["verified_count"] == 5
        assert "mae" in res_met["metrics"]
        assert "rmse" in res_met["metrics"]
        assert "nse" in res_met["metrics"]
        assert res_met["metrics"]["interval_coverage"] == 1.0


class TestDriftMonitoring:
    """Test PSI, KS-test, and freshness checks."""

    def test_psi_identical_distribution(self):
        data = np.random.normal(loc=100.0, scale=15.0, size=500)
        psi = compute_psi(data, data)
        assert psi < 0.05  # Virtually zero drift

    def test_psi_shifted_distribution(self):
        baseline = np.random.normal(loc=100.0, scale=10.0, size=500)
        shifted = np.random.normal(loc=180.0, scale=25.0, size=500)
        psi = compute_psi(baseline, shifted)
        assert psi > 0.25  # Significant drift detected

    def test_ks_test(self):
        baseline = np.random.normal(loc=50.0, scale=5.0, size=200)
        same = np.random.normal(loc=50.0, scale=5.0, size=200)
        res_same = compute_ks_test(baseline, same)
        assert not res_same["drift_detected"]

        diff = np.random.normal(loc=150.0, scale=20.0, size=200)
        res_diff = compute_ks_test(baseline, diff)
        assert res_diff["drift_detected"]
        assert res_diff["p_value"] < 0.05

    def test_data_and_model_freshness(self):
        now_dt = datetime.now(timezone.utc)
        fresh_data_ts = (now_dt - timedelta(hours=1)).isoformat()
        fresh_model_ts = (now_dt - timedelta(days=5)).isoformat()

        # Fresh data (1 hour ago), fresh model (5 days ago)
        fresh_check = check_data_and_model_freshness(
            latest_data_timestamp=fresh_data_ts,
            model_trained_timestamp=fresh_model_ts,
        )
        assert not fresh_check["is_data_stale"]
        assert not fresh_check["is_model_stale"]
        assert len(fresh_check["warnings"]) == 0

        # Stale data (100 hours ago), stale model (60 days ago)
        stale_data_ts = (now_dt - timedelta(hours=100)).isoformat()
        stale_model_ts = (now_dt - timedelta(days=60)).isoformat()
        stale_check = check_data_and_model_freshness(
            latest_data_timestamp=stale_data_ts,
            model_trained_timestamp=stale_model_ts,
        )
        assert stale_check["is_data_stale"]
        assert stale_check["is_model_stale"]
        assert len(stale_check["warnings"]) == 2


class TestModelRegistry:
    """Test registry and champion/challenger promotion rules."""

    def test_registration_and_promotion_rule(self, tmp_path):
        reg_file = str(tmp_path / "registry.json")
        registry = ModelRegistry(registry_path=reg_file)

        # 1. Register Champion
        champ_entry = registry.register_model(
            model_name="ridge",
            version="1.0.0",
            station_id="cauvery_trichy",
            horizon_days=1,
            validation_metrics={"rmse": 45.0, "mae": 30.0},
            data_hash="abc123hash",
            config_hash="cfg456hash",
            is_champion=True,
        )
        assert registry.get_champion()["model_name"] == "ridge"

        # 2. Register Challenger with worse RMSE -> should not promote
        registry.register_model(
            model_name="decision_tree",
            version="1.0.0",
            station_id="cauvery_trichy",
            horizon_days=1,
            validation_metrics={"rmse": 55.0, "mae": 40.0},
            data_hash="abc123hash",
            config_hash="cfg456hash",
            is_champion=False,
        )
        promo_fail = registry.evaluate_and_promote(
            challenger_key="decision_tree_1.0.0_cauvery_trichy_h1",
            metric="rmse",
            smaller_is_better=True,
            dm_p_value=0.01,
        )
        assert not promo_fail["promoted"]
        assert registry.get_champion()["model_name"] == "ridge"

        # 3. Register Challenger with better RMSE but insignificant DM test (p >= 0.05) -> should not promote
        registry.register_model(
            model_name="xgboost",
            version="1.0.0",
            station_id="cauvery_trichy",
            horizon_days=1,
            validation_metrics={"rmse": 40.0, "mae": 25.0},
            data_hash="abc123hash",
            config_hash="cfg456hash",
            is_champion=False,
        )
        promo_fail_sig = registry.evaluate_and_promote(
            challenger_key="xgboost_1.0.0_cauvery_trichy_h1",
            metric="rmse",
            smaller_is_better=True,
            dm_p_value=0.12,  # p >= 0.05
        )
        assert not promo_fail_sig["promoted"]
        assert "failed significance test" in promo_fail_sig["reason"]
        assert registry.get_champion()["model_name"] == "ridge"

        # 4. Challenger with better RMSE AND significant DM test (p < 0.05) -> PROMOTED
        promo_success = registry.evaluate_and_promote(
            challenger_key="xgboost_1.0.0_cauvery_trichy_h1",
            metric="rmse",
            smaller_is_better=True,
            dm_p_value=0.02,  # p < 0.05
        )
        assert promo_success["promoted"]
        assert registry.get_champion()["model_name"] == "xgboost"


class TestOperationsAndBenchmark:
    """Test scheduler dry-run and latency benchmark."""

    def test_scheduler_dry_run(self):
        res = run_pipeline_refresh(dry_run=True, station_id="TN_CAU_TRICHY")
        assert res["status"] == "success"
        assert res["dry_run"] is True
        assert len(res["actions"]) > 0
        assert any("[DRY-RUN]" in act for act in res["actions"])

    def test_benchmark_latency_output(self, tmp_path):
        out_file = str(tmp_path / "latency.json")
        res = benchmark_pipeline_latency(n_runs=10, output_path=out_file)
        assert Path(out_file).exists()
        assert "median_total_ms" in res["summary"]
        assert "p95_total_ms" in res["summary"]
        assert "fetch" in res["stage_breakdown"]
        assert "features" in res["stage_breakdown"]
        assert "predict" in res["stage_breakdown"]
        assert "render" in res["stage_breakdown"]
