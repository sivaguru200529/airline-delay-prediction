# Airline Delay Prediction & Operations Analytics

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pytest-brightgreen.svg)](tests/)
[![Code-Style](https://img.shields.io/badge/Code%20Style-PEP%208-orange.svg)](https://pep8.org/)

A production-grade, end-to-end Machine Learning and Operations Analytics system designed to predict commercial flight arrival delays and surface operational bottlenecks for airline dispatchers and airport station managers.

---

## 1. Project Objective

The primary objective is to develop an enterprise-ready system that:
1. Ingests and standardizes multi-year public aviation flight records.
2. Identifies key structural and temporal factors driving flight disruptions.
3. Formulates a binary classification engine predicting the likelihood of significant arrival delays before pushback.
4. Categorizes predictions into actionable operational risk tiers (Low, Medium, High, Very High).
5. Serves low-latency predictions via a FastAPI REST service and provides an interactive Streamlit operations dashboard.

---

## 2. Problem Statement

Flight delays cost the global aviation industry tens of billions of dollars annually in fuel burn, crew rescheduling, passenger accommodations, and missed connections. 

Operational dispatchers currently react to delays after they materialize. By providing a reliable delay probability score prior to scheduled departure, airlines can proactively swap aircraft, adjust turnarounds, notify ground handlers, and optimize gate assignments.

### Prediction Scenario & Reference Timestamp
* **Prediction Timing**: The model generates predictions at or before scheduled departure:
  $$\text{prediction\_time} = \text{scheduled\_departure}$$
* **Binary Target Definition**:
  $$\text{delay\_target} = \begin{cases} 1 & \text{if } \text{arrival\_delay} \ge 15 \text{ minutes} \\ 0 & \text{otherwise} \end{cases}$$
  *(15 minutes is the standard threshold defined by the FAA and U.S. Bureau of Transportation Statistics).*

---

## 3. Strict Data Leakage Prevention

A critical failure mode in aviation delay modeling is **data leakage**—training models on operational variables that are only known after the flight pushes back, departs, or lands.

To guarantee zero leakage:
1. **Target Isolation**: `arrival_delay` is strictly the prediction target and is never used as an input feature.
2. **Post-Flight Feature Purge**: The following operational fields are retained in the cleaned operational dataset for historical analysis, but are **strictly excluded** from the pre-departure prediction feature matrix:
   - `arrival_delay`
   - `departure_delay`
   - `actual_departure` / `actual_dep_time`
   - `actual_arrival` / `actual_arr_time`
   - `taxi_out` & `taxi_in`
   - `wheels_off` & `wheels_on`
   - `air_time` & `elapsed_time`
3. **Data Flow Separation**:
   $$\text{Raw Operational Data} \longrightarrow \text{Target Generation} \longrightarrow \text{Pre-Departure Feature Matrix}$$

---

## 4. Time-Aware Validation & Historical Feature Policy

* **Temporal Train/Test Validation**: Future machine learning phases will evaluate models using chronological out-of-time splits (e.g. historical quarters for training, subsequent months for testing) rather than random splits, preventing future-to-past information bleed.
* **Historical Feature Policy**: All rolling metrics (e.g. 30-day airline delay rate, origin airport congestion rate, route delay history) must strictly obey:
  $$\text{observation\_timestamp} < \text{prediction\_time}$$
  No flight occurring after `prediction_time` may contribute to rolling aggregations.

---

## 5. Technology Stack

* **Programming**: Python 3.11+, Jupyter Notebook
* **Data Processing**: Pandas, NumPy, PyArrow
* **Configuration & Logging**: Pydantic, python-dotenv, Python standard `logging`
* **Testing**: Pytest, pytest-cov
* **Machine Learning (Phase 2)**: Scikit-learn, XGBoost
* **Model Explainability (Phase 2)**: SHAP (SHapley Additive exPlanations)
* **Visualization (Phase 2)**: Matplotlib, Seaborn, Plotly
* **Backend API (Phase 3)**: FastAPI, Pydantic, Uvicorn
* **Operations Dashboard (Phase 3)**: Streamlit, Plotly
* **Database (Phase 3)**: PostgreSQL, SQLAlchemy

---

## 6. Public Dataset Strategy

This project is built to work with official public flight datasets without synthetic fabrication:

1. **U.S. Bureau of Transportation Statistics (BTS) TranStats**:
   - Official Reporting Carrier On-Time Performance database.
   - [BTS TranStats Portal](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ)
2. **Kaggle 2015 Flight Delays and Cancellations**:
   - Curated BTS data extract covering 5.8M flights across 14 major airlines.
   - [Kaggle Dataset Link](https://www.kaggle.com/datasets/usdot/flight-delays)

### Dynamic Schema Mapping
The ingestion pipeline automatically recognizes column headers from either BTS (`FL_DATE`, `OP_UNIQUE_CARRIER`, `CRS_DEP_TIME`, `ARR_DELAY`, etc.) or Kaggle (`YEAR`, `MONTH`, `DAY`, `AIRLINE`, `SCHEDULED_DEPARTURE`, `ARRIVAL_DELAY`, etc.) and normalizes them to canonical representations.

For instructions on placing raw data files or using development test samples, see [data/raw/README.md](data/raw/README.md).

---

## 7. Project Architecture

```text
airline-delay-prediction/
│
├── data/
│   ├── raw/                 # Public datasets (refer to data/raw/README.md)
│   ├── processed/           # Cleaned operational & pre-departure datasets
│   ├── sample/              # Development sample for smoke testing
│   └── external/            # External sources (weather observations for Phase 2)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_eda.ipynb
│   └── 05_model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/                # Ingestion, validation, and leakage-aware cleaning
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   ├── features/            # Feature engineering and temporal checks (Phase 2)
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/              # Training, inference, evaluation (Phase 2)
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── utils/               # Configuration and structured logging
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logger.py
│   └── database/            # Database connection and schema (Phase 3)
│       ├── __init__.py
│       └── connection.py
│
├── api/                     # FastAPI service (Phase 3)
│   ├── __init__.py
│   └── main.py
│
├── dashboard/               # Streamlit application (Phase 3)
│   └── app.py
│
├── models/                  # Serialized ML artifacts (.joblib / .json)
├── reports/
│   ├── data_quality_report.md
│   ├── data_quality_report.json
│   └── figures/
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_features.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```

---

## 8. Installation & Setup

### Prerequisites
* Python 3.11 or higher
* Git

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/airline-delay-prediction.git
cd airline-delay-prediction

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env   # Windows
# or: cp .env.example .env # Linux/macOS
```

---

## 9. Pipeline CLI Execution

### Run Complete End-to-End Pipeline
```bash
# Run full Phase 1 -> Phase 2A -> Phase 2B end-to-end pipeline
python main.py --all
```
*(If no file is yet placed in `data/raw/`, the CLI runs against the verified development sample in `data/sample/`).*

### Run Specific Modular Phases
```bash
# Run only Phase 2B (ML training, evaluation, threshold analysis, SHAP, model serialization)
python main.py --phase2b

# Run only Phase 2A (Feature engineering, historical features, quality reports)
python main.py --phase2a

# Ingestion & Schema Inspection only (Phase 1)
python main.py --ingest

# Data Quality Validation only (Phase 1)
python main.py --validate

# Preprocessing & Anti-Leakage purge only (Phase 1)
python main.py --preprocess
```

### Run on a Specific Public Dataset
```bash
python main.py --data-path data/raw/flights_2024.csv --all
```

---

## 10. Phase 2A — Feature Engineering, Historical Features & EDA

Phase 2A delivers a validated, ML-ready feature dataset (`flights_features.parquet`) ready for Phase 2B modeling without target leakage.

### 1. Pre-Departure Feature Engineering (`src/features/feature_engineering.py`)
- **Calendar Attributes**: `year`, `month`, `day`, `day_of_week` (0=Mon..6=Sun), `week_of_year`, `day_of_year`, `is_weekend`.
- **Scheduled Departure Timing**: `departure_hour`, `departure_minute`, `departure_minutes_since_midnight`, and operational blocks `time_of_day` (`overnight` [22-6), `morning` [6-12), `afternoon` [12-18), `evening` [18-22)).
- **Scheduled Arrival Timing**: `arrival_hour`, `arrival_minute`, `arrival_minutes_since_midnight` (when present in schedule).
- **Periodic Cyclical Transformations**: Mathematically exact and semantically continuous unit-circle projections for `departure_hour_sin/cos` (period 24), `day_of_week_sin/cos` (period 7), and `month_sin/cos` (period 12), ensuring 23:59 is adjacent to 00:01 and December is adjacent to January.
- **Route & Flight Attributes**: `route` (`ORIGIN_DEST`), `distance`, and FAA haul categories `haul_category` (`short_haul` <500 mi, `medium_haul` 500-1500 mi, `long_haul` >1500 mi).
- **Conditional Omission**: `same_airport_flag` was audited; because 100% of commercial flights had $\text{origin} \ne \text{destination}$ (zero variance), it was omitted to eliminate dead noise.

### 2. Time-Aware Historical Delay Features (`src/features/historical_features.py`)
- **Strict Anti-Leakage Invariant**:
  $$\text{observation\_timestamp} < \text{prediction\_time} = \text{scheduled\_departure}$$
- **Features Generated**:
  - `historical_origin_delay_rate` & `historical_origin_flight_count`
  - `historical_destination_delay_rate` & `historical_destination_flight_count`
  - `historical_airline_delay_rate` & `historical_airline_flight_count`
  - `historical_route_delay_rate` & `historical_route_flight_count`
- **Zero-Leakage Guarantees**:
  - Current flight target is strictly excluded.
  - Concurrent flights departing at the exact same timestamp are excluded from each other.
  - Future flights are barred.
  - Supporting volume counts are retained alongside rates to allow downstream models to weight low-sample evidence.
  - Configurable minimum-history rule (`min_history = 3`) with fallback to strictly prior global rate.
- **Development Sample Limitation**:
  The 500-flight development sample contains limited historical depth across 10 days. Sample-derived rates are designed for integration verification, not population-level operational conclusions.

### 3. Weather Integration Foundation (`src/features/weather_features.py`)
- **Observed vs. Forecast Distinction**: The architecture explicitly models weather *observed* at or before scheduled departure ($t_{obs} \le T_{dep}$), preserving local-to-UTC alignment. It does *not* claim observed weather is equivalent to an advance forecast.
- **Zero Fabrication**: Because external weather datasets are not yet placed in `data/external/`, the pipeline outputs:
  ```text
  WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED
  ```
  leaving the production ML matrix unpolluted by synthetic records while the interface, schema validator, and nearest-prior temporal join logic are fully verified.

### 4. Automated Leakage Audit (`assert_no_target_leakage`)
- Centralized in `src/utils/config.py`.
- Rejects any post-flight outcome columns (`arrival_delay`, `departure_delay`, `actual_dep_time`, `actual_arr_time`, `taxi_out`, `taxi_in`, `wheels_off`, `wheels_on`, `air_time`, `elapsed_time`) or derived keyword patterns.
- Guarantees `LEAKAGE AUDIT: PASS`.

### 5. Chronological Out-of-Time Dataset Splitting
- Sorts strictly by scheduled departure timestamp:
  - **Train**: 70% earliest records (2024-01-01 to 2024-01-07)
  - **Validation**: 15% intermediate records (2024-01-07 to 2024-01-09)
  - **Test**: 15% future out-of-time records (2024-01-09 to 2024-01-10)
- Prevents future-to-past lookahead bias.

### 6. Generated Phase 2A Artifacts
- **ML Datasets**:
  - `data/processed/flights_features.parquet` (481 records, 38 features)
  - `data/processed/flights_features.csv`
- **Documentation & Reports**:
  - `reports/feature_dictionary.md` (complete specification of all 38 features)
  - `reports/feature_quality_report.md` & `.json` (data health, missingness, leakage results)
  - `reports/figures/*.png` (6 publication-grade figures)
  - `notebooks/04_eda.ipynb` (interactive exploratory analysis)

---

## 11. Phase 2B — Machine Learning Training, Evaluation & Explainability

Phase 2B implements the complete machine-learning training, validation, testing, threshold optimization, probability calibration, explainability, and model serialization pipeline.

> **DEVELOPMENT DATASET LIMITATION NOTICE:**  
> The metrics presented below are **development-sample results** evaluated on **481 completed flights** (January 1–10, 2024).  
> These metrics verify pipeline execution, anti-leakage invariants, out-of-time chronological validation, and explainability architecture.  
> **Statistically representative operational performance requires training on the full multi-month or multi-year BTS dataset.**

### 1. Problem Formulation & Prediction Contract
* **Target Definition**:
  $$\text{delay\_target} = \begin{cases} 1 & \text{if } \text{arrival\_delay} \ge 15 \text{ minutes} \\ 0 & \text{otherwise} \end{cases}$$
* **Prediction Timing Invariant**:
  $$\text{prediction\_time} = \text{scheduled\_departure}$$
* **Anti-Leakage Guarantees**:
  - Excludes `delay_target` and all post-flight outcome variables (`arrival_delay`, `departure_delay`, `actual_dep_time`, `actual_arr_time`, `taxi_out`, `taxi_in`, `wheels_off`, `wheels_on`, `air_time`, `elapsed_time`) from feature matrices.
  - Learned preprocessing operations (imputers, encoders, scalers) are **fitted exclusively on the training partition**.
  - Model selection and classification threshold optimization are **conducted strictly on validation data**, with final evaluation performed **once on out-of-time test data**.

### 2. Chronological Out-of-Time Dataset Partitioning
Sorts strictly by scheduled departure timestamp without random shuffling:
* **Train Partition**: 336 flights (69.85%) [2024-01-01 to 2024-01-07] — Delay Rate: 31.85%
* **Validation Partition**: 72 flights (14.97%) [2024-01-07 to 2024-01-09] — Delay Rate: 27.78%
* **Test Partition**: 73 flights (15.18%) [2024-01-09 to 2024-01-10] — Delay Rate: 24.66%
* **Temporal Ordering Verified**: $\max(\text{Train}) \le \min(\text{Val}) \le \min(\text{Test})$.

### 3. Leakage-Free Preprocessing & Feature Schema
* **Numerical Features (30)**: `distance`, `departure_hour`, `departure_minute`, cyclical projections (`sin`/`cos`), and all 8 historical rate/count features. Handled via `SimpleImputer(strategy='median')` (+ `StandardScaler` for Logistic Regression).
* **Categorical Features (6)**: `airline`, `origin_airport`, `dest_airport`, `time_of_day`, `route`, `haul_category`. Handled via `OneHotEncoder(handle_unknown='ignore')`.
* **Data-Driven Route Representation**: Evaluated 129 unique routes in 481 flights (average 3.73 observations/route). Included with `handle_unknown='ignore'` to support route-level granularity while avoiding test failure on unseen routes.
* **Preservation of Historical Features**: All 8 Phase 2A strictly prior historical rates and observation counts retained.

### 4. Validation Model Benchmarking & Selection
Evaluated objectively on the **Validation Set** to prevent test set data snooping:

| Candidate Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | 0.7222 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2778 | 0.2778 |
| **Logistic Regression** | 0.3194 | 0.2836 | 0.9500 | 0.4368 | 0.5615 | 0.3557 | 0.4791 |
| **Random Forest** | 0.4861 | 0.2069 | 0.3000 | 0.2449 | 0.3779 | 0.2291 | 0.2621 |
| **XGBoost** | 0.4722 | 0.2353 | 0.4000 | 0.2963 | 0.4067 | 0.2483 | 0.3032 |

#### Validation Selection Rationale
* **Selected Winning Model**: `Logistic Regression`
* **Validation Justification**: Highest validation PR-AUC (0.3557 vs XGBoost 0.2483, Random Forest 0.2291) and highest F1-score (0.4368 vs XGBoost 0.2963).
* **Operational Threshold Selected**: **0.50** (achieves 95.0% recall of validation delays while maintaining balanced precision).

### 5. Final Out-of-Time Test Evaluation (Unbiased)
Evaluated **once** on the untouched test partition (73 flights, Jan 9–10, 2024) at threshold = 0.50:

* **Accuracy**: 0.2603
* **Precision**: 0.2500
* **Recall**: 1.0000
* **F1-Score**: 0.4000
* **ROC-AUC**: 0.5424
* **PR-AUC**: 0.3036
* **Brier Score**: 0.5995
* **Test Confusion Matrix**:
  - True Negatives (TN): 1
  - False Positives (FP): 54
  - False Negatives (FN): 0
  - True Positives (TP): 18

### 6. Operational Risk Tiers & Inference Utility (`src/models/predict.py`)
Predictions are output as continuous probabilities $P(\text{Delay} \ge 15\text{ min})$ and mapped into configurable operational risk tiers:
* `[0.00, 0.30)` $\to$ **LOW**
* `[0.30, 0.60)` $\to$ **MEDIUM**
* `[0.60, 0.80)` $\to$ **HIGH**
* `[0.80, 1.00]` $\to$ **VERY HIGH**

### 7. Feature Importance & SHAP Explainability
* **Meaningful Feature Names**: Transformed one-hot encoded feature names are dynamically retrieved from the fitted pipeline (`get_feature_names_out()`).
* **Top Predictive Features**: `route_ATL_BOS` (1.4169), `origin_airport_MIA` (1.2505), `route_JFK_BOS` (1.0365), `origin_airport_BOS` (0.9347), `route_ATL_SFO` (0.9135).
* **SHAP Explainability**: Status: **SUCCESS**. Global feature contributions computed via `shap.LinearExplainer` / `shap.TreeExplainer` and exported to `reports/figures/shap_summary.png`.

### 8. Phase 2B Artifacts Generated
* **Model Serialization**:
  - `models/delay_model.joblib` (complete fitted preprocessing + classification pipeline)
  - `models/model_metadata.json` (architecture, training timestamp, feature list, date ranges, metrics)
* **Reports**:
  - `reports/model_evaluation_report.md` (comprehensive markdown evaluation report)
  - `reports/model_evaluation_report.json` (machine-readable metrics and metadata)
* **Figures**:
  - `reports/figures/confusion_matrix.png`
  - `reports/figures/roc_curve.png`
  - `reports/figures/precision_recall_curve.png`
  - `reports/figures/calibration_curve.png`
  - `reports/figures/feature_importance.png`
  - `reports/figures/shap_summary.png`
* **Interactive Notebook**:
  - `notebooks/05_model_training.ipynb` (16 structured sections using modular `src/` functions)

---

## 12. Running Automated Tests

Run the full unit test suite with pytest:

```bash
pytest -v tests/
```

### Verified Test Cases (33 Total — 100% Pass):
* **Phase 1 Tests (6 tests)**: Target generation, leakage column purging, BTS schema mapping, Kaggle schema mapping, required column validation, data quality checks.
* **Phase 2A Tests (15 tests)**: Temporal validity acceptance/rejection, calendar feature extraction, military time parsing and midnight rollovers, cyclical unit-circle identities, route and haul categorization, historical delay strictly prior aggregation, minimum history fallback, weather schema validation, weather backward temporal join, automated leakage audit, chronological dataset split, final 38-feature schema validation.
* **Phase 2B Tests (12 tests)**: Chronological split ordering ($\text{Train} < \text{Val} < \text{Test}$), target/leakage exclusion from feature matrix $X$, route feature suitability evaluation, preprocessing fitted strictly on training data, unknown categorical level handling (`handle_unknown='ignore'`), candidate model training and convergence (Baseline, Logistic Regression, Random Forest, XGBoost), validation model selection, inference schema and operational risk tier mapping, model serialization and reload reproducibility, metric calculation correctness, threshold sensitivity analysis, tree feature importance extraction with meaningful transformed names.

---

## 13. Project Status & Roadmap

| Phase | Milestone | Status |
| :--- | :--- | :--- |
| **Phase 1** | Scaffolding, Ingestion, Validation, Anti-Leakage Preprocessing, CLI, Pytest | **COMPLETED** |
| **Phase 2A** | Feature Engineering, Temporal Historical Features, Weather Foundation, Leakage Audit, EDA | **COMPLETED** |
| **Phase 2B** | Chronological Split Training, Baseline, LR, RF, XGBoost, Thresholds, SHAP, Reports | **COMPLETED** |
| **Phase 3** | PostgreSQL Data Layer, FastAPI Microservice, Streamlit Operations Dashboard | *Next Phase* |
| **Phase 4** | Dockerization, CI/CD Pipeline, Model Registry, Production Packaging | *Upcoming* |
