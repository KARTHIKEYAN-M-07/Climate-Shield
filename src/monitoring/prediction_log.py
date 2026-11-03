"""Prediction logging and verification storage using SQLite.

Tracks:
- origin_time, station, horizon, model_version, prediction, interval, data_freshness
- forward verification once actual modelled discharge arrives
- requires n >= 30 verified predictions before reporting forward verification scores
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class PredictionLogger:
    """Manages SQLite storage for model predictions and forward verification."""

    def __init__(self, db_path: str = "reports/predictions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the predictions table if it doesn't already exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    origin_time TEXT NOT NULL,
                    target_time TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    lower_bound REAL,
                    upper_bound REAL,
                    nominal_coverage REAL,
                    data_freshness_hours REAL,
                    actual_discharge REAL,
                    verified_at TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log_prediction(
        self,
        origin_time: str,
        target_time: str,
        station_id: str,
        horizon_days: int,
        model_version: str,
        prediction: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        nominal_coverage: Optional[float] = None,
        data_freshness_hours: Optional[float] = None,
    ) -> int:
        """Log a new forecast prediction."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    origin_time, target_time, station_id, horizon_days,
                    model_version, prediction, lower_bound, upper_bound,
                    nominal_coverage, data_freshness_hours, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    origin_time,
                    target_time,
                    station_id,
                    horizon_days,
                    model_version,
                    float(prediction),
                    float(lower_bound) if lower_bound is not None else None,
                    float(upper_bound) if upper_bound is not None else None,
                    float(nominal_coverage) if nominal_coverage is not None else None,
                    float(data_freshness_hours) if data_freshness_hours is not None else None,
                    now_str,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def log_verification(
        self,
        station_id: str,
        target_time: str,
        actual_discharge: float,
    ) -> int:
        """Log the actual verified discharge for target_time and station."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE predictions
                SET actual_discharge = ?, verified_at = ?
                WHERE station_id = ? AND target_time = ? AND actual_discharge IS NULL
                """,
                (float(actual_discharge), now_str, station_id, target_time),
            )
            conn.commit()
            return cursor.rowcount

    def get_forward_verification_metrics(
        self,
        station_id: Optional[str] = None,
        horizon_days: Optional[int] = None,
        min_count: int = 30,
    ) -> Dict[str, Any]:
        """Compute verification metrics only if verified count >= min_count.
        
        Otherwise returns a clear 'not enough verified forecasts' status.
        """
        query = """
            SELECT prediction, actual_discharge, lower_bound, upper_bound
            FROM predictions
            WHERE actual_discharge IS NOT NULL
        """
        params: List[Any] = []
        if station_id:
            query += " AND station_id = ?"
            params.append(station_id)
        if horizon_days is not None:
            query += " AND horizon_days = ?"
            params.append(horizon_days)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        count = len(rows)
        if count < min_count:
            return {
                "status": "not enough verified forecasts",
                "verified_count": count,
                "required_count": min_count,
                "metrics": None,
            }

        preds = np.array([r["prediction"] for r in rows], dtype=float)
        actuals = np.array([r["actual_discharge"] for r in rows], dtype=float)

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        mean_actual = np.mean(actuals)
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - mean_actual) ** 2)
        nse = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Empirical interval coverage if bounds are present
        covered = [
            r["lower_bound"] <= r["actual_discharge"] <= r["upper_bound"]
            for r in rows
            if r["lower_bound"] is not None and r["upper_bound"] is not None
        ]
        interval_coverage = float(np.mean(covered)) if covered else None

        return {
            "status": "verified",
            "verified_count": count,
            "required_count": min_count,
            "metrics": {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "nse": round(nse, 4),
                "interval_coverage": round(interval_coverage, 4) if interval_coverage is not None else None,
            },
        }
