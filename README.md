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
python main.py --all
```
*(If no file is yet placed in `data/raw/`, the CLI runs against the verified development sample in `data/sample/`).*

### Run on a Specific Public Dataset
```bash
python main.py --data-path data/raw/flights_2024.csv --all
```

### Run Specific Modular Steps
```bash
# Ingestion & Schema Inspection only
python main.py --ingest

# Data Quality Validation only
python main.py --validate

# Preprocessing & Anti-Leakage purge only
python main.py --preprocess
```

---

## 10. Running Automated Tests

Run the full unit test suite with pytest:

```bash
pytest -v tests/
```

### Verified Test Cases:
* **Target Threshold Verification**: Confirms $\text{delay} \ge 15 \to 1$, $\text{delay} < 15 \to 0$.
* **Data Leakage Check**: Asserts that `arrival_delay`, `departure_delay`, `actual_departure`, `actual_arrival`, `taxi_out`, `taxi_in`, `wheels_off`, `wheels_on`, and `air_time` are purged from the pre-departure feature set.
* **Schema Resolution**: Tests automatic mapping of BTS and Kaggle headers.
* **Data Validation**: Tests detection of duplicates, invalid dates, malformed airport codes, and negative distances.
* **Temporal Integrity**: Tests that historical feature calculation rejects future records.

---

## 11. Project Status & Roadmap

| Phase | Milestone | Status |
| :--- | :--- | :--- |
| **Phase 1** | Scaffolding, Ingestion, Validation, Anti-Leakage Preprocessing, CLI, Pytest | **COMPLETED** |
| **Phase 2** | Feature Engineering, Time-Aware Split, Weather Join, Baseline & XGBoost, SHAP | *Next Phase* |
| **Phase 3** | PostgreSQL Data Layer, FastAPI Microservice, Streamlit Operations Dashboard | *Upcoming* |
| **Phase 4** | Dockerization, CI/CD Pipeline, Model Registry, Production Packaging | *Upcoming* |
