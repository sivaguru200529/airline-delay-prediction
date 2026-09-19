# Airline Operations - Data Quality Audit Report

## Executive Summary

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Raw Records** | 500 | 100.00% |
| **Duplicate Records** | 0 | 0.00% |
| **Cancelled Flights** | 13 | 2.60% |
| **Diverted Flights** | 6 | 1.20% |
| **Invalid / Corrupt Records** | 0 | 0.00% |
| **Valid Completed Flights** | 481 | 96.20% |

## Target Distribution (Valid Completed Flights)

| Target Class | Definition | Flight Count | Proportion |
| :--- | :--- | :--- | :--- |
| **Delayed (1)** | Arrival Delay $\ge 15$ min | 145 | 30.15% |
| **On-Time (0)** | Arrival Delay $< 15$ min | 336 | 69.85% |

## Data Leakage Audit (Post-Flight Columns Detected)

The following columns were detected in the raw operational dataset and classified as **post-flight leakage**.
These fields are retained in the full cleaned dataset for operational retrospectives, but are strictly excluded from the pre-departure prediction feature matrix:

- ` arrival_delay ` (Post-flight / Outcome)
- ` departure_delay ` (Post-flight / Outcome)
- ` actual_dep_time ` (Post-flight / Outcome)
- ` actual_arr_time ` (Post-flight / Outcome)
- ` taxi_out ` (Post-flight / Outcome)
- ` taxi_in ` (Post-flight / Outcome)
- ` air_time ` (Post-flight / Outcome)

## Schema Mapping Audit

| Canonical Conceptual Field | Source Column |
| :--- | :--- |
| `flight_date` | `FL_DATE` |
| `airline` | `OP_UNIQUE_CARRIER` |
| `origin_airport` | `ORIGIN` |
| `dest_airport` | `DEST` |
| `scheduled_dep_time` | `CRS_DEP_TIME` |
| `actual_dep_time` | `DEP_TIME` |
| `departure_delay` | `DEP_DELAY` |
| `scheduled_arr_time` | `CRS_ARR_TIME` |
| `actual_arr_time` | `ARR_TIME` |
| `arrival_delay` | `ARR_DELAY` |
| `distance` | `DISTANCE` |
| `cancelled` | `CANCELLED` |
| `diverted` | `DIVERTED` |
| `air_time` | `AIR_TIME` |
| `taxi_out` | `TAXI_OUT` |
| `taxi_in` | `TAXI_IN` |

## Missing Values Breakdown

| Column | Missing Count | Missing Rate |
| :--- | :--- | :--- |
| `actual_arr_time` | 19 | 3.80% |
| `arrival_delay` | 19 | 3.80% |
| `air_time` | 19 | 3.80% |
| `taxi_in` | 19 | 3.80% |
| `actual_dep_time` | 13 | 2.60% |
| `departure_delay` | 13 | 2.60% |
| `taxi_out` | 13 | 2.60% |

## Cleaning & Segregation Decisions

- Detected 0 exact duplicate rows flagged for deduplication.
- Identified 13 cancelled flights; segregated into operational cancellation analysis.
- Identified 6 diverted flights; segregated into operational diversion analysis.
- Flagged 0 records with unparseable flight dates.
- Flagged 0 records with invalid IATA airport codes (non-3-letter).
- Flagged 0 records with non-positive or extreme distances (>10,000 miles).
- Flagged 0 completed flights with missing or extreme arrival delay values.
- Arrival delay >= 15 min used as delayed ground-truth target.
- All post-flight operational metrics (actual departure/arrival times, taxi durations, air time) strictly excluded from pre-departure model feature matrix to prevent data leakage.
