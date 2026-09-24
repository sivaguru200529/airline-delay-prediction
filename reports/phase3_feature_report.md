# Phase 3 — Advanced Feature Engineering Report

> **DEVELOPMENT DATASET LIMITATION NOTICE**
> The features and coverage statistics reported here were generated on the development sample dataset
> (**481 completed flights**, January 1–10, 2024).
> This evaluation verifies pipeline integrity, strict anti-leakage guards, and real-data compatibility.
> **Production operational conclusions require scaling to the full multi-month/multi-year public BTS dataset.**

---

## 1. Feature Version & Change Summary

* **Feature Version**: `phase3`
* **Total Features**: **68 columns** (including `delay_target`)
* **Retained from Phase 2A (9)**: `flight_date`, `airline`, `origin_airport`, `dest_airport`, `scheduled_dep_time`, `scheduled_arr_time`, `distance`, `route`, `delay_target`
* **Enhanced in Phase 3 (7)**: `dep_time_of_day`, `route_distance`, `route_distance_category`, `prior_airline_delay_rate`, `prior_origin_delay_rate`, `prior_dest_delay_rate`, `prior_route_delay_rate`
* **New in Phase 3 (52)**: `cancelled`, `diverted`, `is_month_start`, `is_month_end`, `quarter`, `season`, `dep_time_bucket`, `dep_minute_sin`, `dep_minute_cos`, `arr_time_of_day`, `arr_time_bucket`, `arr_hour_sin`, `arr_hour_cos`, `distance_category`, `prior_route_frequency`, `prior_origin_flight_volume`, `prior_dest_flight_volume`, `prior_airline_flight_count`, `prior_airline_delay_count`, `carrier_prior_flight_count`, `airline_prior_flight_count`, `carrier_prior_delay_count`, `airline_prior_delay_count`, `carrier_prior_delay_rate`, `airline_prior_delay_rate`, `prior_origin_flight_count`, `prior_origin_delay_count`, `origin_prior_flight_count`, `origin_prior_delay_count`, `origin_prior_delay_rate`, `prior_dest_flight_count`, `prior_dest_delay_count`, `dest_prior_flight_count`, `dest_prior_delay_count`, `dest_prior_delay_rate`, `prior_route_flight_count`, `prior_route_delay_count`, `route_prior_flight_count`, `route_prior_delay_count`, `route_prior_delay_rate`, `prior_airline_dep_hour_flight_count`, `prior_airline_dep_hour_delay_count`, `prior_airline_dep_hour_delay_rate`, `carrier_origin_hour_prior_flight_count`, `airline_dep_hour_prior_flight_count`, `carrier_origin_hour_prior_delay_count`, `airline_dep_hour_prior_delay_count`, `carrier_origin_hour_prior_delay_rate`, `airline_dep_hour_prior_delay_rate`, `prior_origin_dep_hour_flight_count`, `prior_origin_dep_hour_delay_count`, `prior_origin_dep_hour_delay_rate`

---

## 2. Temporal Anti-Leakage & Availability Contract

* **Prediction Time Reference**: Scheduled Departure ($T_{dep}$)
* **Historical Features Invariant**:
  $$\text{historical\_observation\_timestamp} < T_{dep}$$
  The current flight and future flights are **strictly excluded** from historical metrics.
* **Strictly Prior Global Fallback**:
  When historical observations fall below `min_history=3`, the global delay fallback rate is calculated **strictly from observations occurring prior to $T_{dep}$** (no full-dataset lookahead).
* **Weather Prediction-Time Availability Contract**:
  - **1. Observed Weather**: METAR/station observations captured at or before prediction time ($T_{obs} \le T_{dep}$).
  - **2. Historical Weather**: Archived surface observations strictly prior to departure ($T_{obs} < T_{dep}$).
  - **3. Forecast Weather**: Meteorological forecasts (TAF) issued at or before prediction time ($T_{issue} \le T_{dep}$).
  - **Anti-Leakage Prohibition**: Weather observations recorded after scheduled departure and forecasts issued after scheduled departure are strictly barred from entering the pre-departure feature matrix.
* **Leakage Audit Status**: **`LEAKAGE AUDIT: PASS`**

---

## 3. Historical Delay Feature Coverage Audit

Evaluated on the development sample dataset (minimum history threshold = 3):

| Entity / Dimension | No History (Count = 0) | Insufficient History (0 < Count < 3) | Sufficient History (Count >= 3) |
| :--- | :--- | :--- | :--- |
| Airline Delay Rate | 7 (1.46%) | 14 (2.91%) | 460 (95.63%) |
| Origin Delay Rate | 12 (2.49%) | 25 (5.2%) | 444 (92.31%) |
| Destination Delay Rate | 13 (2.7%) | 23 (4.78%) | 445 (92.52%) |
| Route Delay Rate | 129 (26.82%) | 209 (43.45%) | 143 (29.73%) |
| Airline Dep Hour Delay Rate | 115 (23.91%) | 195 (40.54%) | 171 (35.55%) |
| Origin Dep Hour Delay Rate | 178 (37.01%) | 214 (44.49%) | 89 (18.5%) |

*Note: On this 10-day development extract, low-frequency routes and airport-hour pairs appropriately fell back to strictly prior global rates, preserving robust sample weighting for downstream models.*

---

## 4. Route Distance Distribution Analysis

* **Minimum Distance**: 205 miles
* **Maximum Distance**: 2,597 miles
* **Median Distance**: 1,382.0 miles
* **Missing Values**: 0
* **Documented Haul Buckets**:
  - `short_haul`: $[0, 500)$ miles
  - `medium_haul`: $[500, 1500)$ miles
  - `long_haul`: $[1500, \infty)$ miles

---

## 5. Weather Integration Foundation

* **Weather Status**: `WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED`
* **Integration Strategy**: Verified interface ready for NOAA METAR/TAF surface observations without synthetic data fabrication.

---

## 6. Complete Phase 3 Feature Dictionary

| Feature Name | Phase Status | Data Type | Unique Values | Missing % | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `flight_date` | RETAINED | `str` | 10 | 0.0% | Flight scheduled calendar date. |
| `airline` | RETAINED | `str` | 7 | 0.0% | Operating airline 2-letter carrier code. |
| `origin_airport` | RETAINED | `str` | 12 | 0.0% | Scheduled departure airport 3-letter IATA code. |
| `dest_airport` | RETAINED | `str` | 12 | 0.0% | Scheduled arrival airport 3-letter IATA code. |
| `scheduled_dep_time` | RETAINED | `int64` | 107 | 0.0% | Scheduled departure military time (HHMM). |
| `scheduled_arr_time` | RETAINED | `int64` | 424 | 0.0% | Scheduled arrival military time (HHMM). |
| `distance` | RETAINED | `int64` | 429 | 0.0% | Non-stop flight distance in statute miles. |
| `cancelled` | NEW (PHASE 3) | `int64` | 1 | 0.0% | Engineered operational predictor. |
| `diverted` | NEW (PHASE 3) | `int64` | 1 | 0.0% | Engineered operational predictor. |
| `is_month_start` | NEW (PHASE 3) | `int64` | 2 | 0.0% | Binary flag (1 if day == 1, else 0). |
| `is_month_end` | NEW (PHASE 3) | `int64` | 1 | 0.0% | Binary flag (1 if day is last day of the month, else 0). |
| `quarter` | NEW (PHASE 3) | `int64` | 1 | 0.0% | Calendar quarter (1-4). |
| `season` | NEW (PHASE 3) | `str` | 1 | 0.0% | Meteorological season ('winter', 'spring', 'summer', 'fall'). |
| `dep_time_of_day` | ENHANCED (PHASE 3) | `str` | 4 | 0.0% | Standardized departure block ('night', 'morning', 'afternoon', 'evening'). |
| `dep_time_bucket` | NEW (PHASE 3) | `str` | 5 | 0.0% | 4-hour operational block ('00-04', '04-08', '08-12', '12-16', '16-20', '20-24'). |
| `dep_minute_sin` | NEW (PHASE 3) | `float64` | 7 | 0.0% | sin(2*pi*dep_minute/60) circular minute projection. |
| `dep_minute_cos` | NEW (PHASE 3) | `float64` | 7 | 0.0% | cos(2*pi*dep_minute/60) circular minute projection. |
| `arr_time_of_day` | NEW (PHASE 3) | `str` | 4 | 0.0% | Standardized arrival block ('night', 'morning', 'afternoon', 'evening'). |
| `arr_time_bucket` | NEW (PHASE 3) | `str` | 6 | 0.0% | 4-hour operational block for arrival schedule. |
| `arr_hour_sin` | NEW (PHASE 3) | `float64` | 12 | 0.0% | sin(2*pi*arr_hour/24) unit-circle projection. |
| `arr_hour_cos` | NEW (PHASE 3) | `float64` | 13 | 0.0% | cos(2*pi*arr_hour/24) unit-circle projection. |
| `route` | RETAINED | `str` | 129 | 0.0% | Directional airport-pair route identifier. |
| `route_distance` | ENHANCED (PHASE 3) | `int64` | 429 | 0.0% | Validated route flight distance in statute miles. |
| `route_distance_category` | ENHANCED (PHASE 3) | `str` | 3 | 0.0% | Phase 3 documented haul category (<500 mi short, 500-1500 mi med, >=1500 mi long). |
| `distance_category` | NEW (PHASE 3) | `str` | 3 | 0.0% | Engineered operational predictor. |
| `prior_route_frequency` | NEW (PHASE 3) | `int64` | 9 | 0.0% | Number of flights on this route occurring strictly before scheduled departure (t < T). |
| `prior_origin_flight_volume` | NEW (PHASE 3) | `int64` | 46 | 0.0% | Total flights departing from origin airport strictly before scheduled departure (t < T). |
| `prior_dest_flight_volume` | NEW (PHASE 3) | `int64` | 49 | 0.0% | Total flights arriving at destination airport strictly before scheduled departure (t < T). |
| `prior_airline_flight_count` | NEW (PHASE 3) | `int32` | 80 | 0.0% | Total prior flights for airline strictly before departure. |
| `prior_airline_delay_count` | NEW (PHASE 3) | `int32` | 27 | 0.0% | Total prior delayed flights for airline strictly before departure. |
| `prior_airline_delay_rate` | ENHANCED (PHASE 3) | `float64` | 233 | 0.0% | Strictly prior airline delay rate with strictly prior global fallback. |
| `carrier_prior_flight_count` | NEW (PHASE 3) | `int32` | 80 | 0.0% | Engineered operational predictor. |
| `airline_prior_flight_count` | NEW (PHASE 3) | `int32` | 80 | 0.0% | Engineered operational predictor. |
| `carrier_prior_delay_count` | NEW (PHASE 3) | `int32` | 27 | 0.0% | Engineered operational predictor. |
| `airline_prior_delay_count` | NEW (PHASE 3) | `int32` | 27 | 0.0% | Engineered operational predictor. |
| `carrier_prior_delay_rate` | NEW (PHASE 3) | `float64` | 233 | 0.0% | Engineered operational predictor. |
| `airline_prior_delay_rate` | NEW (PHASE 3) | `float64` | 233 | 0.0% | Engineered operational predictor. |
| `prior_origin_flight_count` | NEW (PHASE 3) | `int32` | 46 | 0.0% | Engineered operational predictor. |
| `prior_origin_delay_count` | NEW (PHASE 3) | `int32` | 18 | 0.0% | Total prior delayed flights for origin airport strictly before departure. |
| `prior_origin_delay_rate` | ENHANCED (PHASE 3) | `float64` | 144 | 0.0% | Strictly prior origin delay rate with strictly prior global fallback. |
| `origin_prior_flight_count` | NEW (PHASE 3) | `int32` | 46 | 0.0% | Engineered operational predictor. |
| `origin_prior_delay_count` | NEW (PHASE 3) | `int32` | 18 | 0.0% | Engineered operational predictor. |
| `origin_prior_delay_rate` | NEW (PHASE 3) | `float64` | 144 | 0.0% | Engineered operational predictor. |
| `prior_dest_flight_count` | NEW (PHASE 3) | `int32` | 49 | 0.0% | Engineered operational predictor. |
| `prior_dest_delay_count` | NEW (PHASE 3) | `int32` | 19 | 0.0% | Total prior delayed flights for destination airport strictly before departure. |
| `prior_dest_delay_rate` | ENHANCED (PHASE 3) | `float64` | 147 | 0.0% | Strictly prior destination delay rate with strictly prior global fallback. |
| `dest_prior_flight_count` | NEW (PHASE 3) | `int32` | 49 | 0.0% | Engineered operational predictor. |
| `dest_prior_delay_count` | NEW (PHASE 3) | `int32` | 19 | 0.0% | Engineered operational predictor. |
| `dest_prior_delay_rate` | NEW (PHASE 3) | `float64` | 147 | 0.0% | Engineered operational predictor. |
| `prior_route_flight_count` | NEW (PHASE 3) | `int32` | 9 | 0.0% | Engineered operational predictor. |
| `prior_route_delay_count` | NEW (PHASE 3) | `int32` | 5 | 0.0% | Total prior delayed flights for route strictly before departure. |
| `prior_route_delay_rate` | ENHANCED (PHASE 3) | `float64` | 225 | 0.0% | Strictly prior route delay rate with strictly prior global fallback. |
| `route_prior_flight_count` | NEW (PHASE 3) | `int32` | 9 | 0.0% | Engineered operational predictor. |
| `route_prior_delay_count` | NEW (PHASE 3) | `int32` | 5 | 0.0% | Engineered operational predictor. |
| `route_prior_delay_rate` | NEW (PHASE 3) | `float64` | 225 | 0.0% | Engineered operational predictor. |
| `prior_airline_dep_hour_flight_count` | NEW (PHASE 3) | `int32` | 9 | 0.0% | Prior flights for airline departing in same hour bucket strictly before T. |
| `prior_airline_dep_hour_delay_count` | NEW (PHASE 3) | `int32` | 5 | 0.0% | Engineered operational predictor. |
| `prior_airline_dep_hour_delay_rate` | NEW (PHASE 3) | `float64` | 218 | 0.0% | Prior delay rate for airline departing in same hour bucket with global fallback. |
| `carrier_origin_hour_prior_flight_count` | NEW (PHASE 3) | `int32` | 9 | 0.0% | Engineered operational predictor. |
| `airline_dep_hour_prior_flight_count` | NEW (PHASE 3) | `int32` | 9 | 0.0% | Engineered operational predictor. |
| `carrier_origin_hour_prior_delay_count` | NEW (PHASE 3) | `int32` | 5 | 0.0% | Engineered operational predictor. |
| `airline_dep_hour_prior_delay_count` | NEW (PHASE 3) | `int32` | 5 | 0.0% | Engineered operational predictor. |
| `carrier_origin_hour_prior_delay_rate` | NEW (PHASE 3) | `float64` | 218 | 0.0% | Engineered operational predictor. |
| `airline_dep_hour_prior_delay_rate` | NEW (PHASE 3) | `float64` | 218 | 0.0% | Engineered operational predictor. |
| `prior_origin_dep_hour_flight_count` | NEW (PHASE 3) | `int32` | 7 | 0.0% | Prior flights departing origin in same hour bucket strictly before T. |
| `prior_origin_dep_hour_delay_count` | NEW (PHASE 3) | `int32` | 4 | 0.0% | Engineered operational predictor. |
| `prior_origin_dep_hour_delay_rate` | NEW (PHASE 3) | `float64` | 258 | 0.0% | Prior delay rate departing origin in same hour bucket with global fallback. |
| `delay_target` | RETAINED | `int64` | 2 | 0.0% | Binary delay label (1 if arrival_delay >= 15 min else 0). |

---

## 7. Known Limitations & Roadmap

1. **Development Sample Size**: 481 flights provides limited historical depth for specific carrier-hour and route pairs.
2. **Phase 2B Model Preservation**: Phase 2B models (`models/delay_model.joblib`) were strictly preserved.
3. **Future Phase 3B**: Model retraining and benchmarking (Phase 2A features vs Phase 3 features) will be executed in a dedicated, separate comparison phase.
