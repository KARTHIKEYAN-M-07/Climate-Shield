# ClimateShield AI — Data Sheet for Datasets
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
