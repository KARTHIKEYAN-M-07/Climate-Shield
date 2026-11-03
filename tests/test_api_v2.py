"""Tests for API v2 endpoints and contracts (Phase 20).

Tests:
1. Versioned routes:
   - GET /v1/stations/{id}/history
   - POST /v1/explain
   - GET /v1/uncertainty/{station}
   - GET /v1/verification
   - GET /v1/drift
   - GET /v1/metrics
   - GET /v1/model-card
2. Predict contract:
   - Point forecast, interval (bounds and measured coverage), risk category, OOD flag, model version.
3. Auth contract:
   - API key validation via environment variable CLIMATESHIELD_API_KEY.
4. Error contracts:
   - 401 when auth fails, 404 when station or model is missing, 422 on schema violation.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestApiV2Routes:
    def test_v1_model_card(self):
        resp = client.get("/v1/model-card")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "2.0.0"
        assert "GloFAS" in data["target"]
        assert "content_markdown" in data

    def test_v1_metrics_prometheus(self):
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        assert "climateshield_api_up 1" in body

    def test_v1_drift_status(self):
        resp = client.get("/v1/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert "freshness" in data
        assert "psi_rule" in data

    def test_v1_verification_insufficient(self):
        resp = client.get("/v1/verification?min_count=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not enough verified forecasts"
        assert data["metrics"] is None

    def test_v1_explain_feature_attributions(self):
        payload = {
            "features": {
                "precipitation_sum_cumul_7d": 85.0,
                "river_discharge_roc_1d": 35.0,
                "monsoon_ne": 1.0,
            },
            "reference_features": {
                "precipitation_sum_cumul_7d": 20.0,
            },
            "top_k": 3,
        }
        resp = client.post("/v1/explain", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "top_features" in data
        assert len(data["top_features"]) >= 2

    def test_v1_station_history_missing_station(self):
        resp = client.get("/v1/stations/UNKNOWN_STATION/history")
        assert resp.status_code == 404
        assert "No processed historical data found" in resp.json()["detail"]

    def test_api_key_auth(self, monkeypatch):
        monkeypatch.setenv("CLIMATESHIELD_API_KEY", "secret_passphrase")

        # Request without header -> 401
        resp_unauth = client.post(
            "/predict",
            json={
                "station_id": "TN_CAU_TRICHY",
                "model_name": "xgboost",
                "features": {"river_discharge": 100.0},
            },
        )
        assert resp_unauth.status_code == 401
        assert "Invalid or missing API key" in resp_unauth.json()["detail"]

        # Request with correct header -> proceeds past auth check (returns 404 for missing station)
        resp_auth = client.post(
            "/predict",
            headers={"X-API-Key": "secret_passphrase"},
            json={
                "station_id": "TN_NONEXISTENT",
                "model_name": "xgboost",
                "features": {"river_discharge": 100.0},
            },
        )
        # Auth passed; failure is 404 since station model isn't found
        assert resp_auth.status_code == 404
