# ClimateShield AI: System Architecture & Data Flow

## 1. System Overview

ClimateShield AI is an academic decision-support prototype for river discharge forecasting across monitored basins in Tamil Nadu, India. The system forecasts modelled daily river discharge ($m^3/s$) for forecast horizons of $h \in \{1, 3, 7\}$ days ahead.

Discharge data is sourced from the Copernicus GloFAS / Open-Meteo modelled hydrological dataset and meteorological precipitation data is sourced from Open-Meteo under CC BY 4.0.

> [!IMPORTANT]
> **Academic Decision-Support Prototype**:
> All discharge figures represent **modelled hydrological runoff**, not physical in-situ river gauge water levels. All risk thresholds are prototype statistical percentiles ($p_{50}, p_{75}, p_{90}$) calculated from historical training sets, not statutory warning stages issued by the Central Water Commission (CWC).

---

## 2. End-to-End Pipeline Architecture

```mermaid
flowchart TD
    subgraph INGEST["1. Data Ingestion & Sanity Gating"]
        A["Open-Meteo REST API<br/>(GloFAS Discharge + Precipitation)"] --> B["data/raw/<br/>Historical CSVs"]
        B --> C{"Station Sanity Check<br/>(Missing, Constant, Zero flow)"}
        C -- "Insufficient Data (e.g. Chennai 100% missing)" --> C_FLAG["Flag Station as Insufficient Data<br/>(Write reports/data_sanity_<id>.json)"]
        C -- "Passes Sanity Check" --> D["src/data/cleaning.py<br/>(Causal forward fill <= 3d, reindex, alignment)"]
    end

    subgraph PREP["2. Feature Engineering & Strict Split"]
        D --> E["data/processed/<br/>Aligned Datasets"]
        E --> F["src/features/temporal_features.py<br/>(Lags, Rolling Means/Stds, Rainfall Intensity,<br/>Days Since Rain, Month Sin/Cos)"]
        F --> G["Chronological Split<br/>(Train 70% | Validation 15% | Test 15%)<br/>Zero Future Leakage"]
    end

    subgraph TRAIN["3. Model Training & Conformal Calibration"]
        G --> H["Target Transform Evaluation<br/>(log1p vs raw discharge on validation)"]
        H --> I["Model Training Suite<br/>(Persistence, Ridge, Decision Tree, Random Forest,<br/>Gradient Boosting, XGBoost, HistGradientBoosting,<br/>ElasticNet, ExtraTrees)"]
        I --> J["Split-Conformal Calibration<br/>(Calibration on Validation Residuals: q_hat)"]
        J --> K["Baseline Benchmark Comparison<br/>(Persistence, Seasonal Naive, Climatology)"]
    end

    subgraph ARTIFACTS["4. Artifacts & Reports"]
        I --> L["models/<station_id>/<br/>*.joblib & *.meta.json"]
        J --> M["models/<station_id>/<br/>*_conformal.json"]
        K --> N["reports/<br/>model_scores_all.csv<br/>training_coverage.csv<br/>feature_ablation.csv<br/>training_log.jsonl"]
    end

    subgraph SERVING["5. Operational Serving & Monitoring"]
        L & M & N --> O["Streamlit Dashboard<br/>(src/dashboard/app.py)"]
        L & M & N --> P["FastAPI Service<br/>(src/api/main.py)"]
        O --> Q["Top KPI Cards<br/>(Decoupled Risk, Flow, Conformal Band, Drift)"]
        O --> R["Operational Retraining Gate<br/>(src/retrain.py with Val RMSE Gate)"]
        O --> S["Feature Drift Engine<br/>(PSI < 0.10, |z| < 2.5)"]
    end
```

---

## 3. Core Architectural Modules

### 3.1 Data Sanity & Quality Assurance
- **Sanity Checks (`src/train.py`, `src/data/`):**
  Validates each station before training. Stations with zero valid readings, all-zero flows, or large gaps (e.g., Chennai with 100% missing discharge) are cleanly flagged as `insufficient_data` and saved to `reports/data_sanity_<station_id>.json`. The pipeline skips downstream model fitting for these stations without halting the multi-station loop.
- **Causal Cleaning:**
  Missing gaps up to 3 days are forward-filled; longer gaps remain marked as NaN with gap tracking flags to eliminate future-to-past data leakage.

### 3.2 Leakage-Free Temporal Feature Engineering
Features are strictly computed using historical observations available at time $t$:
- **Lags:** Discharge ($t-1, t-2, t-3, t-7, t-14$) and Rainfall ($t, t-1, t-2, t-3, t-7$).
- **Rolling Aggregations:** Rolling means ($3d, 7d, 14d$) and rolling standard deviations ($7d, 14d$).
- **Rainfall Intensity:** Maximum daily rainfall in previous 3 and 7 days.
- **Antecedent Indicators:** Days since significant rainfall ($P \ge 5.0\text{ mm}$), Antecedent Precipitation Index (API).
- **Calendar Encodings:** Sin/Cos transformations of day of year and month.

### 3.3 Target Transformation & Cross-Validation
- **Target Formulation:**
  Evaluates raw discharge versus $\log(1 + y)$ target transformation on the validation set. Inverse transformations ($\exp(y) - 1$) are applied before calculating evaluation metrics.
- **Chronological Splitting:**
  Strict chronological split (70% Train, 15% Validation, 15% Test) ensures temporal integrity. Scalers and imputers are fit exclusively on the training partition.

### 3.4 Conformal Prediction Intervals
- **Methodology:** Split-conformal calibration on validation set absolute residuals:
  $$q_{\hat{\alpha}} = \text{Quantile}_{1-\alpha}(|y_{\text{val}} - \hat{y}_{\text{val}}|)$$
- **Interval Bounds:** $[\max(0, \hat{y} - q_{\hat{\alpha}}), \hat{y} + q_{\hat{\alpha}}]$.
- **Measured Coverage:** Empirical coverage is evaluated on the untouched test partition and displayed directly in the user interface (e.g., *Nominal 90%, measured coverage 91.4%*).

### 3.5 Decoupled Risk Classification
- **Statutory Decoupling:** Risk categories (NORMAL, ADVISORY, WATCH, WARNING) are evaluated strictly from the model forecast discharge against station-specific empirical percentiles ($p_{50}, p_{75}, p_{90}$) calibrated on the training partition.
- **Staleness & Absence Protection:** If data is stale ($>48\text{h}$) or if a station lacks a trained model, the risk category strictly defaults to `UNAVAILABLE`.

### 3.6 Automated Retraining with Quality Gate (`src/retrain.py`)
- Retrains models on newly ingested observations.
- Compares newly trained model validation RMSE against the existing deployed model:
  $$\text{RMSE}_{\text{new}} \le \text{RMSE}_{\text{deployed}} \times 1.001$$
- Artifacts on disk are updated **only** if the gate passes; all retraining decisions are audited to `reports/retrain_log.jsonl`.
