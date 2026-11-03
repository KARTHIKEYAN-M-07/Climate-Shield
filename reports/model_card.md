# ClimateShield AI — Model Card v2.0
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
