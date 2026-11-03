"""Generators for Model Cards and Data Sheets.

Outputs:
- reports/model_card.md
- reports/data_sheet.md
"""

from pathlib import Path


def generate_model_card(output_path: str = "reports/model_card.md") -> str:
    """Generate comprehensive Model Card markdown."""
    content = r"""# ClimateShield AI — Model Card v2.0
**Target**: Modelled GloFAS River Discharge ($m^3/s$)
**Spatial Domain**: Tamil Nadu River Basins (Cauvery, Vaigai, Thamirabarani, Palar, Adyar, Bhavani, Noyyal)

---

## 1. Model Details
- **Architectures**: XGBoost, Random Forest, Gradient Boosting, HistGradientBoosting, Extra Trees, ElasticNet, Ridge, Seasonal Climatology, Damped Persistence AR(1), Persistence Baseline.
- **Input Features**: Catchment-areal rainfall, Antecedent Precipitation Index (API decay 0.85/0.90), days-since-last-heavy-rain, wet-spell length, seasonal monsoon flags (SW: Jun-Sep, NE: Oct-Dec), historical discharge lags.
- **Forecast Horizons**: Direct multi-horizon forecasting for $h \in \{1, 3, 7\}$ days ahead.
- **Uncertainty Calibration**: Split-conformal prediction intervals calibrated strictly on the validation partition (nominal $90\%$ target).

---

## 2. Intended Use & Target Audience
- **Intended Purpose**: Academic decision-support prototype exploring statistical and machine learning methodologies for river flow forecasting in peninsular India.
- **Target Audience**: Researchers, hydrology students, and data scientists studying basin-scale streamflow dynamics.

---

## 3. Important Limitations & Out-of-Scope Uses
- **Physical Exclusions**: The model strictly predicts numerical grid reanalysis river discharge; it does NOT simulate localized water depth, levee breaches, storm sewers, tidal surges, or dynamic dam reservoir sluice gate operations.
- **Tree Model Extrapolation**: Decision trees and tree ensembles cannot extrapolate to streamflow magnitudes exceeding historical training extremes. Inputs exceeding historical ranges trigger Out-of-Distribution (OOD) guards returning `LOW CONFIDENCE`.
- **Statutory Notice**: This software is NOT an official flood alert service. Civil protection authorities and the public must follow advisories from IMD, CWC, and the Tamil Nadu State Disaster Management Authority (TNSDMA).
"""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)
    return str(out_file)


def generate_data_sheet(output_path: str = "reports/data_sheet.md") -> str:
    """Generate comprehensive Data Sheet markdown."""
    content = r"""# ClimateShield AI — Data Sheet for Datasets
**Datasets**: Open-Meteo Historical Weather API & GloFAS River Discharge API

---

## 1. Motivation
- **Purpose**: Provide consistent daily time-series of meteorological precipitation and hydrologic river discharge across Tamil Nadu river basins for predictive modelling benchmarks.

---

## 2. Composition & Provenance
- **Precipitation**: Daily total precipitation sum ($mm/day$) derived from ERA5 / ERA5-Land reanalysis via Open-Meteo.
- **River Discharge**: Daily mean modelled river discharge ($m^3/s$) from the Global Flood Awareness System (GloFAS) grid product.
- **Resolution**: $\sim 5\text{ km}$ spatial grid. Grid-snapping checks are conducted to exclude locations where stream discharge is negligible or unresolved.

---

## 3. Preprocessing & Temporal Alignment
- **Chronological Alignment**: Cleaned and causally aligned via strict inner join on UTC calendar dates.
- **Missing Data Treatment**: Short gaps ($\le 3$ days) forward-filled causally; long gaps leave records uncorrupted.
- **Leakage Prevention**: All features (lags, rolling sums, API memory) use only observations up to origin time $t$.

---

## 4. Attribution & License
- Weather and discharge data courtesy of [Open-Meteo.com](https://open-meteo.com) under Creative Commons Attribution 4.0 International (CC BY 4.0).
"""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)
    return str(out_file)
