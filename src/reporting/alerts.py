"""Opt-in webhook / notification dispatch for threshold alerts.

Rules:
- Strictly opt-in: no alerts dispatched unless destination URL or email recipient is configured.
- Dry-run mode for testing payload delivery without network transmission.
- Dispatches alert payload when risk category meets or exceeds configured threshold (e.g. WATCH, WARNING).
- Always includes prototype disclaimer and official agency referral in payload.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests


class AlertDispatcher:
    """Manages opt-in alerting via webhooks or notification sinks."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        min_risk_level: str = "WATCH",
        dry_run: bool = False,
    ):
        self.webhook_url = webhook_url
        self.min_risk_level = min_risk_level
        self.dry_run = dry_run
        self._level_ranks = {"NORMAL": 0, "ADVISORY": 1, "WATCH": 2, "WARNING": 3}

    def should_trigger(self, risk_category: str) -> bool:
        """Evaluate if risk category qualifies for alert dispatch."""
        rank = self._level_ranks.get(risk_category.upper(), -1)
        min_rank = self._level_ranks.get(self.min_risk_level.upper(), 2)
        return rank >= min_rank

    def build_payload(
        self,
        station_id: str,
        station_name: str,
        river_basin: str,
        predicted_flow: float,
        horizon_days: int,
        risk_category: str,
        rationale: str,
    ) -> Dict[str, Any]:
        """Construct alert notification payload."""
        return {
            "event": "CLIMATESHIELD_FLOOD_RISK_ALERT",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "station": {
                "id": station_id,
                "name": station_name,
                "river_basin": river_basin,
            },
            "forecast": {
                "predicted_discharge_m3s": round(predicted_flow, 2),
                "horizon_days": horizon_days,
                "risk_category": risk_category,
                "primary_rationale": rationale,
            },
            "advisory": (
                "NOT an official warning. Academic decision-support prototype. "
                "Follow official IMD, CWC, and State Disaster Management Authority advisories."
            ),
        }

    def dispatch(
        self,
        station_id: str,
        station_name: str,
        river_basin: str,
        predicted_flow: float,
        horizon_days: int,
        risk_category: str,
        rationale: str,
    ) -> Dict[str, Any]:
        """Dispatch alert if opt-in configured and threshold met."""
        if not self.should_trigger(risk_category):
            return {"dispatched": False, "reason": f"Risk {risk_category} below min level {self.min_risk_level}"}

        if not self.webhook_url and not self.dry_run:
            return {"dispatched": False, "reason": "No opt-in webhook_url configured"}

        payload = self.build_payload(
            station_id=station_id,
            station_name=station_name,
            river_basin=river_basin,
            predicted_flow=predicted_flow,
            horizon_days=horizon_days,
            risk_category=risk_category,
            rationale=rationale,
        )

        if self.dry_run:
            return {"dispatched": True, "dry_run": True, "payload": payload}

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=5.0)
            return {
                "dispatched": True,
                "dry_run": False,
                "status_code": resp.status_code,
                "payload": payload,
            }
        except Exception as e:
            return {
                "dispatched": False,
                "dry_run": False,
                "error": str(e),
                "payload": payload,
            }
