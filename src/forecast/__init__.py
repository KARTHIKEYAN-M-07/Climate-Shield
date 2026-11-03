"""src/forecast/__init__.py
─────────────────────────
Inference and forecast engine package for ClimateShield AI.
"""

from src.forecast.engine import ForecastResult, generate_forecast

__all__ = ["ForecastResult", "generate_forecast"]
