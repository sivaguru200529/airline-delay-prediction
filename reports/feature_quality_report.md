# Airline Delay Prediction — Feature Quality Report (Phase 2A)

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| **Input Records** | 481 |
| **Final ML Features Records** | 481 |
| **Total Feature Count** | 38 |
| **Numerical Features** | 31 |
| **Categorical Features** | 7 |
| **Leakage Audit Status** | **PASS** |
| **Temporal Validity** | **PASS ($t_{obs} < T_{pred}$)** |
| **Weather Integration Status** | **WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED** |

---

## 2. Leakage Audit & Anti-Leakage Invariants

The automated audit verified that 0 post-flight operational outcome columns exist in the feature set.
The following fields were strictly barred from the ML pre-departure view:

- `arrival_delay` (Ground-truth prediction target isolated)
- `departure_delay` (Post-pushback outcome)
- `actual_dep_time` & `actual_arr_time`
- `taxi_out` & `taxi_in`
- `wheels_off` & `wheels_on`
- `air_time` & `elapsed_time`

---

## 3. Historical Delay Features & Strict Temporal Boundaries

- **Historical Rates Created**: `historical_origin_delay_rate`, `historical_destination_delay_rate`, `historical_airline_delay_rate`, `historical_route_delay_rate`
- **Historical Observation Counts Retained**: `historical_origin_flight_count`, `historical_destination_flight_count`, `historical_airline_flight_count`, `historical_route_flight_count`
- **Minimum History Rule**: `min_history = 3` (configurable)
- **Fallback Strategy**: `global_prior` applied when observations < 3

> [!IMPORTANT]
> Every historical calculation satisfies $t_{observation} < T_{scheduled\_departure}$.
> Current flights cannot see their own outcomes, concurrent flights at the same timestamp are excluded, and future flights are completely barred.

---

## 4. Conditional Features & Omissions

- **`same_airport_flag`**: OMITTED: Zero variance detected (100% flights have origin != dest). Feature omitted to avoid dead zero-variance noise.

---

## 5. Missing Values Breakdown

| Feature | Missing Count | Missing Rate | Recommended Handling |
| :--- | :--- | :--- | :--- |
| `historical_origin_delay_rate` | 2 | 0.42% | Group/Global prior fallback applied |
| `historical_destination_delay_rate` | 2 | 0.42% | Group/Global prior fallback applied |
| `historical_airline_delay_rate` | 2 | 0.42% | Group/Global prior fallback applied |
| `historical_route_delay_rate` | 2 | 0.42% | Group/Global prior fallback applied |

---

## 6. Chronological Dataset Split (Out-of-Time)

| Split Cohort | Record Count | Percentage | Date Range |
| :--- | :--- | :--- | :--- |
| **Training (Past)** | 336 | 69.85% | 2024-01-01 to 2024-01-07 |
| **Validation (Intermediate)** | 72 | 14.97% | 2024-01-07 to 2024-01-09 |
| **Test (Future Out-of-Time)** | 73 | 15.18% | 2024-01-09 to 2024-01-10 |

---

## 7. Operational Limitations & Notes

- Historical Feature Limitation: The current development dataset contains only a limited number of historical observations (481 flights across 10 days). Historical group statistics should not be interpreted as representative population-level estimates. For production model training, ingest multi-month BTS extracts as per data/raw/README.md.
- Weather Foundation Status: WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED. External weather records were not found in data/external/. No synthetic weather records were fabricated into the production feature matrix.
