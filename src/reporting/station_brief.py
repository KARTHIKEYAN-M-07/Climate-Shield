"""Generate self-contained, print-ready HTML station briefs in English and Tamil.

Includes:
- Station metadata and river basin
- Latest observed discharge and precipitation
- Model forecast and 90% conformal interval
- Statistical risk tier and top hydrologic rationale
- Regulatory caveat stating prototype status
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.dashboard.localization import get_text


def generate_station_brief_html(
    station_id: str,
    station_name: str,
    river_basin: str,
    latest_flow_m3s: float,
    predicted_flow_m3s: float,
    lower_bound_m3s: float,
    upper_bound_m3s: float,
    risk_category: str,
    rationale: str,
    horizon_days: int = 1,
    model_name: str = "xgboost",
    lang: str = "en",
    output_path: Optional[str] = None,
) -> str:
    """Generate self-contained HTML station brief."""
    t_title = get_text(lang, "title")
    t_warning = get_text(lang, "warning_notice")
    t_risk = get_text(lang, "risk_category")
    t_peak = get_text(lang, "expected_peak")
    t_interval = get_text(lang, "interval_fan")
    t_why = get_text(lang, "why_forecast")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    badge_color = (
        "#2ca02c"
        if risk_category == "NORMAL"
        else "#ff7f0e"
        if risk_category in ["ADVISORY", "WATCH"]
        else "#d62728"
    )

    html_content = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<title>Station Brief - {station_name}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1a1a1a;
    line-height: 1.4;
    padding: 30px;
    max-width: 800px;
    margin: 0 auto;
  }}
  .header {{
    border-bottom: 2px solid #005a9c;
    padding-bottom: 12px;
    margin-bottom: 20px;
  }}
  .header h1 {{
    margin: 0;
    color: #005a9c;
    font-size: 22px;
  }}
  .header p {{
    margin: 4px 0 0;
    color: #555;
    font-size: 13px;
  }}
  .alert-box {{
    background-color: #fff8e6;
    border-left: 4px solid #ffb020;
    padding: 10px 14px;
    font-size: 12px;
    margin-bottom: 20px;
    color: #664d03;
  }}
  .grid {{
    display: flex;
    gap: 15px;
    margin-bottom: 20px;
  }}
  .card {{
    flex: 1;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 14px;
    background: #fdfdfd;
  }}
  .card-label {{
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    font-weight: bold;
    margin-bottom: 6px;
  }}
  .card-value {{
    font-size: 20px;
    font-weight: bold;
    margin: 0;
  }}
  .card-sub {{
    font-size: 11px;
    color: #777;
    margin-top: 4px;
  }}
  .section {{
    margin-bottom: 20px;
  }}
  .section h2 {{
    font-size: 15px;
    border-bottom: 1px solid #eee;
    padding-bottom: 4px;
    color: #333;
  }}
  .footer {{
    border-top: 1px solid #eee;
    padding-top: 10px;
    font-size: 11px;
    color: #777;
    text-align: center;
  }}
  @media print {{
    body {{ padding: 0; }}
    .alert-box {{ border-left-color: #333; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>{t_title}</h1>
  <p>Station Operational Brief: <strong>{station_name}</strong> ({station_id}) | Basin: {river_basin} | Generated: {timestamp_str}</p>
</div>

<div class="alert-box">
  {t_warning}
</div>

<div class="grid">
  <div class="card">
    <div class="card-label">{t_risk}</div>
    <div class="card-value" style="color: {badge_color};">{risk_category}</div>
    <div class="card-sub">Statistical percentile threshold</div>
  </div>
  <div class="card">
    <div class="card-label">{t_peak} (+{horizon_days}d)</div>
    <div class="card-value">{predicted_flow_m3s:.1f} m³/s</div>
    <div class="card-sub">Latest: {latest_flow_m3s:.1f} m³/s ({model_name})</div>
  </div>
  <div class="card">
    <div class="card-label">{t_interval}</div>
    <div class="card-value">[{lower_bound_m3s:.1f}, {upper_bound_m3s:.1f}]</div>
    <div class="card-sub">Nominal 90% split-conformal bounds</div>
  </div>
</div>

<div class="section">
  <h2>{t_why}</h2>
  <p style="font-size: 13px; color: #222;">{rationale}</p>
</div>

<div class="section">
  <h2>Operational Notes & Data Provenance</h2>
  <ul style="font-size: 12px; color: #444;">
    <li>Modelled GloFAS numerical river discharge target. Gauge height / river bed depth not represented.</li>
    <li>Meteorological precipitation and hydrologic risk are evaluated independently; never merged.</li>
    <li>Data courtesy of Open-Meteo.com under Creative Commons Attribution 4.0 International (CC BY 4.0).</li>
  </ul>
</div>

<div class="footer">
  ClimateShield AI v2 — Tamil Nadu River Basins Decision Support Prototype | Not an official flood alert system
</div>

</body>
</html>
"""

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
