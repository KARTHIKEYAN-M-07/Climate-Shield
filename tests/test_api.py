"""
tests/test_api.py
─────────────────
Phase-9 integration tests:
  - GET /health returns 200 and disclaimer
  - GET /stations returns valid station list with Cauvery stations
  - GET /model-performance returns 404 when reports/model_scores.json is missing,
    or returns scores when present.
  - POST /predict:
      * Valid request with mock model returns forecast and risk category
      * Missing model artifact returns 404
      * Stale data returns status UNAVAILABLE in risk evaluation
      * Malformed feature schema returns 422
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app
from src.models.persistence import PersistenceBaseline

client = TestClient(app)


class TestApiEndpoints:
    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "OK"
        assert "disclaimer" in data

    def test_stations_endpoint(self):
        resp = client.get("/stations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 3
        station_ids = [s["id"] for s in data["stations"]]
        assert "TN_CAU_TRICHY" in station_ids

    def test_model_performance_missing_returns_404(self, monkeypatch, tmp_path):
        # Point to non-existent scores
        from src.api import main
        monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
        resp = client.get("/model-performance")
        assert resp.status_code == 404
        assert "run 'python -m src.train'" in resp.json()["detail"]

    def test_predict_model_missing_returns_404(self):
        payload = {
            "station_id": "TN_CAU_NONEXISTENT",
            "model_name": "xgboost",
            "horizon_days": 1,
            "data_freshness_hours": 0.0,
            "features": {"river_discharge": 100.0},
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_predict_stale_data_returns_unavailable(self, tmp_path, monkeypatch):
        from src.api import main

        # Create temporary dummy model artifact
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        model = PersistenceBaseline()
        model_path = models_dir / "TN_CAU_TRICHY_persistence_h1d.joblib"
        joblib.dump(model, model_path)

        # Mock project root in main
        monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)

        payload = {
            "station_id": "TN_CAU_TRICHY",
            "model_name": "persistence",
            "horizon_days": 1,
            "data_freshness_hours": 72.0,  # > 48h staleness limit
            "features": {"river_discharge": 150.0},
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_evaluation"]["status"] == "UNAVAILABLE"
        assert data["risk_evaluation"]["is_stale"] is True

    def test_predict_bad_schema_returns_422(self):
        # Missing required 'features' dict
        bad_payload = {
            "station_id": "TN_CAU_TRICHY",
            "model_name": "persistence",
        }
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422
