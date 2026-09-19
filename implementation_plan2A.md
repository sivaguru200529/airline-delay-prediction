# Phase 2A — Feature Engineering, Historical Features, Weather Foundation & EDA

## Objective & Scope

Extend the existing Phase 1 architecture to implement **Phase 2A**:
1. **Feature Engineering** (`src/features/feature_engineering.py`): Calendar, scheduled departure/arrival time representations, cyclical transformations ($\sin$/$\cos$), route characteristics, and haul categories.
2. **Strict Time-Aware Historical Features** (`src/features/historical_features.py`): Origin airport, destination airport, airline, and route historical delay rates and volume metrics computed strictly over prior events ($t_{observation} < t_{prediction}$), excluding current and concurrent flights, with configurable fallbacks.
3. **Weather Integration Foundation** (`src/features/weather_features.py`): Modular interface, schema validation, and nearest-prior timestamp matching for external weather data; detects absence of real data in `data/external/` and reports `WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED` without fabricating data.
4. **Missing Value & Outlier Analysis**: Column-level missing statistics and feature-specific treatment without blind zero-filling.
5. **Centralized Leakage Audit** (`src/utils/config.py` & audit function): Rigorous verification that no forbidden post-flight outcome or derived field enters the feature matrix.
6. **Temporal Dataset Split Utility**: Chronological splitting tool respecting actual dataset date boundaries (train/validation/test).
7. **Quality Reporting & Documentation**:
   - `reports/feature_quality_report.md` and `.json`
   - `reports/feature_dictionary.md`
   - Updated `reports/figures/`
8. **Exploratory Data Analysis (EDA)** (`notebooks/04_eda.ipynb` & export script): Structured analysis with publication-quality visualizations covering dataset overview, temporal trends, airline performance, airport congestion, and route/distance distributions.
9. **CLI & Unit Testing**:
   - Updated `main.py` supporting `--phase2a` and `--all`
   - Comprehensive test suite in `tests/test_features.py` ensuring temporal validity, non-leakage, schema validation, and cyclical math.

---

## User Review Required

> [!IMPORTANT]
> - Real weather data is absent from `data/external/` (`.gitkeep` only). In strict accordance with the prompt specification, no synthetic official records will be fabricated into the production pipeline. The weather foundation interface, schema, and matching logic will be fully implemented and verified with isolated test fixtures, and the pipeline will report `WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED`.
> - Historical delay rates will be calculated using strictly prior flights where $t < t_{flight}$ (excluding current and concurrent flights). For flights with insufficient history ($count < min\_history$), a configurable prior global baseline (or null/fallback) is applied.

---

## Proposed Changes

### Configuration & Leakage Invariants

#### [MODIFY] [config.py](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/src/utils/config.py)
- Add centralized definitions for:
  - `forbidden_leakage_features` (comprehensive list of raw and derived post-flight columns)
  - Time-of-day categories and threshold definitions (`overnight`: 22:00-06:00, `morning`: 06:00-12:00, `afternoon`: 12:00-18:00, `evening`: 18:00-22:00)
  - Weather integration schema and tolerance parameters
  - Historical feature configuration defaults (`min_history`, fallback strategy)

---

### Feature Engineering Core

#### [MODIFY] [feature_engineering.py](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/src/features/feature_engineering.py)
- Modular feature builders:
  - `build_date_features(df)`: `year`, `month`, `day`, `day_of_week`, `week_of_year`, `day_of_year`, `is_weekend`.
  - `parse_scheduled_time(time_series)`: Robust parsing of HHMM integers/floats handling minute overflow/midnight rollover safely.
  - `build_dep_time_features(df)`: `departure_hour`, `departure_minute`, `departure_minutes_since_midnight`, and categorical `time_of_day`.
  - `build_arr_time_features(df)`: `arrival_hour`, `arrival_minute`, `arrival_minutes_since_midnight` if scheduled arrival time is present.
  - `build_cyclical_features(df)`: Period-exact $\sin$/$\cos$ periodic transformations for `departure_hour` (period 24), `day_of_week` (period 7), and `month` (period 12).
  - `build_route_features(df)`: `route` (`ORIGIN_DEST`), `same_airport_flag`, and distance haul categorization (`short_haul`, `medium_haul`, `long_haul`).
  - `build_all_pre_departure_features(df)`: High-level pipeline aggregating date, time, cyclical, and route features.
  - `assert_no_target_leakage(df)`: Automated leakage audit inspecting columns and derived names against centralized config.

---

### Historical Features (Anti-Leakage Guaranteed)

#### [NEW] [historical_features.py](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/src/features/historical_features.py)
- Construct unified scheduled departure timestamp `scheduled_dep_datetime`.
- Calculate strictly expanding prior metrics over unique timestamps:
  - `historical_origin_delay_rate`, `historical_origin_flight_count`
  - `historical_destination_delay_rate`, `historical_destination_flight_count`
  - `historical_airline_delay_rate`, `historical_airline_flight_count`
  - `historical_route_delay_rate`, `historical_route_flight_count`
- Anti-leakage guarantees:
  - Exclude current flight ($t < T$).
  - Exclude concurrent flights ($t == T$).
  - Exclude future flights ($t > T$).
  - Documented fallback to global prior delay rate or NaN when prior count $< min\_history$.

---

### Weather Integration Foundation

#### [NEW] [weather_features.py](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/src/features/weather_features.py)
- `WeatherSchema`: Schema validator for external weather datasets (airport, timestamp, temperature, humidity, wind_speed, wind_direction, precipitation, visibility, pressure, cloud_cover, weather_condition).
- `WeatherIntegrator`: Nearest-prior temporal join (`observation_time <= scheduled_departure_time` within lookback window, e.g. 2 hours).
- Distinguishes observed weather vs forecast weather.
- Gracefully handles absent external weather data, returning `WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED`.

---

### Pipeline Orchestration & CLI

#### [MODIFY] [main.py](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/main.py)
- Add `--phase2a` command flag.
- Connect:
  1. Phase 1 pre-departure dataset ingestion
  2. Feature engineering & cyclical transformations
  3. Historical delay rate calculation
  4. Weather check (foundation ready status)
  5. Missing value & outlier profiling
  6. Automated leakage audit (`assert_no_target_leakage`)
  7. Dataset export to `data/processed/flights_features.parquet` and `.csv`
  8. Feature quality report export (`reports/feature_quality_report.md` and `.json`)
- Preserve existing `--all`, `--ingest`, `--validate`, and `--preprocess` commands.

---

### EDA & Reporting

#### [MODIFY] [04_eda.ipynb](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/notebooks/04_eda.ipynb)
- Detailed, fully structured exploratory data analysis cells:
  - Executive dataset overview (delay rate, cancellations, diversions).
  - Temporal patterns (by hour, day of week, time of day).
  - Carrier delay comparisons (with explicit correlation $\ne$ causation caveats).
  - Origin and destination airport volume and delay rate distributions.
  - Route analysis with minimum volume thresholds.
  - Haul/distance analysis and correlation with delay duration.
- Automated generation of EDA plots saved to `reports/figures/`.

#### [NEW] [feature_dictionary.md](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/reports/feature_dictionary.md)
- Complete dictionary for every feature:
  - Name, data type, description, source column(s), calculation logic, prediction-time availability, leakage risk, missing-value handling.
  - Explicit documentation for historical features (`Uses only observations strictly before prediction_time`).

#### [NEW] [feature_quality_report.md](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/reports/feature_quality_report.md) & [feature_quality_report.json](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/reports/feature_quality_report.json)
- Summary of input vs final records, feature count, numerical vs categorical columns, missingness report, outlier notes, leakage audit outcome, and chronological split summary.

---

### Tests

#### [MODIFY] [test_features.py](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/tests/test_features.py)
- Preserve existing 2 tests (`test_temporal_validity_accepts_strictly_prior_observations`, `test_temporal_validity_rejects_future_or_concurrent_observations`).
- Add tests for:
  - `test_date_feature_generation`: verifies `year`, `month`, `day`, `day_of_week`, `is_weekend`, etc.
  - `test_dep_time_parsing_and_categories`: verifies HHMM parsing, minute rollover, and time-of-day categories.
  - `test_cyclical_features_periodicity`: verifies $\sin^2 + \cos^2 = 1$ and correct boundary handling.
  - `test_route_feature_generation`: verifies route concatenation and distance hauls.
  - `test_historical_delay_rates_anti_leakage`: deterministic fixture proving flight cannot see itself, future flights, or concurrent flights.
  - `test_historical_delay_rates_progression`: verifies historical delay rate changes properly over time.
  - `test_historical_fallback_handling`: verifies behavior when prior count is below threshold.
  - `test_weather_schema_validation`: verifies weather schema rules.
  - `test_weather_temporal_matching`: verifies weather join strictly matches observations $\le$ prediction time.
  - `test_leakage_audit_fails_on_post_flight_columns`: verifies `assert_no_target_leakage` raises error when post-flight fields enter.
  - `test_final_feature_dataset_schema`: verifies final exported feature matrix structure.

---

### Documentation

#### [MODIFY] [README.md](file:///c:/Users/91934/.gemini/antigravity-ide/scratch/airline-delay-prediction/README.md)
- Add comprehensive Phase 2A section explaining feature engineering, temporal anti-leakage invariants, weather integration design, artifacts generated, and CLI usage.

---

## Verification Plan

### Automated Tests
- Run full test suite with:
  `.venv\Scripts\python.exe -m pytest -v tests/`
  Target: 100% pass across all existing and new tests.

### Pipeline Execution
- Execute Phase 2A pipeline via:
  `.venv\Scripts\python.exe main.py --phase2a`
- Verify outputs:
  - `data/processed/flights_features.parquet`
  - `data/processed/flights_features.csv`
  - `reports/feature_quality_report.md`
  - `reports/feature_quality_report.json`
  - `reports/feature_dictionary.md`
  - `reports/figures/*.png`
- Execute full pipeline:
  `.venv\Scripts\python.exe main.py --all`
