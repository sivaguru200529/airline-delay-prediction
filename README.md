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

## 9. Running the Phase 1 Pipeline

### Run Complete End-to-End Pipeline
```bash
# Run full Phase 1 + Phase 2A end-to-end pipeline
python main.py --all

# Run only Phase 2A (Feature engineering, historical features, quality reports)
python main.py --phase2a
```
*(If no file is yet placed in `data/raw/`, the CLI runs against the verified development sample in `data/sample/`).*

### Run on a Specific Public Dataset
```bash
python main.py --data-path data/raw/flights_2024.csv --all
```

### Run Specific Modular Steps
```bash
# Ingestion & Schema Inspection only (Phase 1)
python main.py --ingest

# Data Quality Validation only (Phase 1)
python main.py --validate

# Preprocessing & Anti-Leakage purge only (Phase 1)
python main.py --preprocess

# Feature Engineering & Quality Reporting only (Phase 2A)
python main.py --phase2a
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

## 11. Running Automated Tests

Run the full unit test suite with pytest:

```bash
pytest -v tests/
```

### Verified Test Cases (21 Total — 100% Pass):
* **Target Threshold Verification**: Confirms $\text{delay} \ge 15 \to 1$, $\text{delay} < 15 \to 0$.
* **Data Leakage Check**: Asserts post-flight operational columns are purged from the pre-departure feature set.
* **Schema Resolution**: Tests automatic mapping of BTS and Kaggle headers.
* **Data Validation**: Tests detection of duplicates, invalid dates, malformed airport codes, and negative distances.
* **Temporal Integrity**: Tests that historical feature calculation rejects future records.
* **Date Feature Generation**: Tests extraction of year, month, day, day_of_week, week_of_year, day_of_year, is_weekend.
* **Scheduled Timing Parsing**: Tests military time parsing, midnight rollovers, and time-of-day blocks.
* **Cyclical Math & Semantic Periodicity**: Tests $\sin^2(x) + \cos^2(x) = 1$, hour 0 vs 24 equality, hour 23/0 circular adjacency, December/January wrap-around, and Sunday/Monday continuity.
* **Route & Haul Categorization**: Tests route string assembly, FAA distance haul grouping, and conditional zero-variance omission.
* **Historical Strict Anti-Leakage**: Deterministic proof that a flight cannot see itself, concurrent flights are excluded, and only prior flights contribute.
* **Historical Fallback**: Tests minimum history threshold and global prior fallback behavior.
* **Weather Schema Validation**: Validates external weather fields, valid IATA codes, and physical value ranges.
* **Weather Temporal Matching**: Tests backward asof join obeying observation_time $\le$ scheduled departure.
* **Automated Leakage Audit**: Tests loud rejection when any post-flight field is introduced.
* **Chronological Dataset Split**: Tests out-of-time separation with zero temporal overlap.
* **Final Feature Dataset Schema**: Validates all 38 required columns and confirms zero leakage in exported parquet.

---

## 12. Project Status & Roadmap

| Phase | Milestone | Status |
| :--- | :--- | :--- |
| **Phase 1** | Scaffolding, Ingestion, Validation, Anti-Leakage Preprocessing, CLI, Pytest | **COMPLETED** |
| **Phase 2A** | Feature Engineering, Temporal Historical Features, Weather Foundation, Leakage Audit, EDA | **COMPLETED** |
| **Phase 2B** | Chronological Split Training, Baseline & XGBoost, Cost-Sensitive Thresholds, SHAP | *Next Phase* |
| **Phase 3** | PostgreSQL Data Layer, FastAPI Microservice, Streamlit Operations Dashboard | *Upcoming* |
| **Phase 4** | Dockerization, CI/CD Pipeline, Model Registry, Production Packaging | *Upcoming* |
