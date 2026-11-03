"""Tests for Phase 22: Outputs and Opt-In Alerts.

Tests:
1. Station brief generation in English and Tamil (HTML formatting, metadata, caveats).
2. Alert dispatcher:
   - Opt-in guard (does not dispatch if unconfigured)
   - Threshold filter (triggers on WATCH/WARNING, ignores NORMAL)
   - Dry-run dispatch payload structure and mandatory advisory disclaimer.
"""

from pathlib import Path
import pytest

from src.reporting.station_brief import generate_station_brief_html
from src.reporting.alerts import AlertDispatcher


class TestStationBrief:
    def test_generate_station_brief_english(self, tmp_path):
        out_file = str(tmp_path / "brief_en.html")
        html = generate_station_brief_html(
            station_id="TN_CAU_TRICHY",
            station_name="Cauvery at Tiruchirappalli",
            river_basin="Cauvery",
            latest_flow_m3s=210.5,
            predicted_flow_m3s=340.2,
            lower_bound_m3s=280.0,
            upper_bound_m3s=410.0,
            risk_category="WATCH",
            rationale="7-day cumulative rainfall is in the top 10% historical bracket.",
            horizon_days=3,
            lang="en",
            output_path=out_file,
        )
        assert Path(out_file).exists()
        assert "Cauvery at Tiruchirappalli" in html
        assert "WATCH" in html
        assert "340.2 m³/s" in html
        assert "ACADEMIC PROTOTYPE NOTICE" in html
        assert "Open-Meteo.com" in html

    def test_generate_station_brief_tamil(self, tmp_path):
        out_file = str(tmp_path / "brief_ta.html")
        html = generate_station_brief_html(
            station_id="TN_CAU_TRICHY",
            station_name="Cauvery at Tiruchirappalli",
            river_basin="Cauvery",
            latest_flow_m3s=210.5,
            predicted_flow_m3s=340.2,
            lower_bound_m3s=280.0,
            upper_bound_m3s=410.0,
            risk_category="WARNING",
            rationale="வடகிழக்கு பருவமழை தீவிரமடைந்துள்ளது.",
            horizon_days=1,
            lang="ta",
            output_path=out_file,
        )
        assert Path(out_file).exists()
        assert "தமிழ்நாடு நதி படுகை வெள்ள முன்னறிவிப்பு அமைப்பு" in html
        assert "அறிவிப்பு" in html


class TestAlertDispatcher:
    def test_unconfigured_dispatcher_does_not_fire(self):
        disp = AlertDispatcher(webhook_url=None, min_risk_level="WATCH", dry_run=False)
        res = disp.dispatch(
            station_id="TN_CAU_TRICHY",
            station_name="Trichy",
            river_basin="Cauvery",
            predicted_flow=500.0,
            horizon_days=1,
            risk_category="WARNING",
            rationale="Heavy upstream rain",
        )
        assert not res["dispatched"]
        assert "No opt-in webhook_url configured" in res["reason"]

    def test_risk_below_threshold_does_not_fire(self):
        disp = AlertDispatcher(webhook_url="https://example.com/webhook", min_risk_level="WATCH")
        res = disp.dispatch(
            station_id="TN_CAU_TRICHY",
            station_name="Trichy",
            river_basin="Cauvery",
            predicted_flow=50.0,
            horizon_days=1,
            risk_category="NORMAL",
            rationale="Baseflow",
        )
        assert not res["dispatched"]
        assert "below min level" in res["reason"]

    def test_dry_run_dispatch_payload_structure(self):
        disp = AlertDispatcher(min_risk_level="WATCH", dry_run=True)
        res = disp.dispatch(
            station_id="TN_CAU_TRICHY",
            station_name="Trichy",
            river_basin="Cauvery",
            predicted_flow=450.0,
            horizon_days=3,
            risk_category="WARNING",
            rationale="Extreme precipitation surge",
        )
        assert res["dispatched"]
        assert res["dry_run"] is True
        payload = res["payload"]
        assert payload["event"] == "CLIMATESHIELD_FLOOD_RISK_ALERT"
        assert payload["forecast"]["risk_category"] == "WARNING"
        assert "NOT an official warning" in payload["advisory"]
        assert "CWC" in payload["advisory"]
