# Airline Delay Prediction — Feature Dictionary (Phase 2A)

## Overview

This feature dictionary defines all variables present in the ML-ready dataset (`data/processed/flights_features.parquet`).
Every feature adheres to the strict anti-leakage condition:
$$\text{observation\_timestamp} < \text{prediction\_time} = \text{scheduled\_departure}$$

## Anti-Leakage Compliance Summary

- **Zero Post-Flight Leakage**: `arrival_delay`, `departure_delay`, `actual_dep_time`, `actual_arr_time`, `taxi_out`, `taxi_in`, `wheels_off`, `wheels_on`, `air_time`, and `elapsed_time` are strictly purged.
- **Historical Statistics**: Historical rates use strictly prior events ($t < T$). Current, concurrent, and future flight outcomes are excluded.
- **Conditional Omission**: `same_airport_flag` is omitted due to zero variance across commercial flights.
- **Weather Foundation**: Observed weather interface is ready; no synthetic weather is fabricated.

---

## Feature Catalog

### `flight_date`
- **Data Type**: string (YYYY-MM-DD)
- **Description**: Flight scheduled calendar date.
- **Source Field(s)**: Raw flight schedule
- **Calculation Logic**: Standardized ISO-8601 date string.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Required conceptual field; records with unparseable dates filtered during validation.

### `airline`
- **Data Type**: string (categorical)
- **Description**: Two-letter IATA carrier code (e.g. AA, DL, UA).
- **Source Field(s)**: Raw OP_UNIQUE_CARRIER / AIRLINE
- **Calculation Logic**: Normalized uppercase carrier code.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Required identifier; missing values rejected.

### `origin_airport`
- **Data Type**: string (categorical)
- **Description**: Three-letter IATA origin departure airport code.
- **Source Field(s)**: Raw ORIGIN / ORIGIN_AIRPORT
- **Calculation Logic**: Trimmed uppercase 3-letter IATA code.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Required field; invalid format rejected.

### `dest_airport`
- **Data Type**: string (categorical)
- **Description**: Three-letter IATA destination arrival airport code.
- **Source Field(s)**: Raw DEST / DESTINATION_AIRPORT
- **Calculation Logic**: Trimmed uppercase 3-letter IATA code.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Required field; invalid format rejected.

### `route`
- **Data Type**: string (categorical)
- **Description**: Directional airport-pair route identifier.
- **Source Field(s)**: origin_airport, dest_airport
- **Calculation Logic**: origin_airport + '_' + dest_airport (e.g., 'JFK_LAX').
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Constructed from validated non-null airport codes.

### `scheduled_dep_time`
- **Data Type**: int32
- **Description**: Scheduled departure time in military HHMM format.
- **Source Field(s)**: Raw CRS_DEP_TIME / SCHEDULED_DEPARTURE
- **Calculation Logic**: Direct representation from schedule timetable.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Required schedule attribute.

### `departure_hour`
- **Data Type**: int32
- **Description**: Scheduled departure hour of the day (0-23).
- **Source Field(s)**: scheduled_dep_time
- **Calculation Logic**: (scheduled_dep_time // 100 + (scheduled_dep_time % 100) // 60) % 24
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Computed safely with minute rollover.

### `departure_minute`
- **Data Type**: int32
- **Description**: Scheduled departure minute of the hour (0-59).
- **Source Field(s)**: scheduled_dep_time
- **Calculation Logic**: (scheduled_dep_time % 100) % 60
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Computed safely with modulo 60.

### `departure_minutes_since_midnight`
- **Data Type**: int32
- **Description**: Total elapsed minutes from midnight to scheduled departure (0-1439).
- **Source Field(s)**: departure_hour, departure_minute
- **Calculation Logic**: departure_hour * 60 + departure_minute
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Complete deterministic calculation.

### `time_of_day`
- **Data Type**: string (categorical)
- **Description**: Operational time block: overnight [22:00, 06:00), morning [06:00, 12:00), afternoon [12:00, 18:00), evening [18:00, 22:00).
- **Source Field(s)**: departure_hour
- **Calculation Logic**: Binned according to FAA operational shift definitions.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Default 'morning' fallback for invalid values.

### `scheduled_arr_time`
- **Data Type**: int32
- **Description**: Scheduled arrival time in military HHMM format.
- **Source Field(s)**: Raw CRS_ARR_TIME / SCHEDULED_ARRIVAL
- **Calculation Logic**: Direct representation from schedule timetable.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Optional feature; retained if present in raw extract.

### `arrival_hour`
- **Data Type**: int32
- **Description**: Scheduled arrival hour of the day (0-23).
- **Source Field(s)**: scheduled_arr_time
- **Calculation Logic**: (scheduled_arr_time // 100 + (scheduled_arr_time % 100) // 60) % 24
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Computed safely with minute rollover if column present.

### `arrival_minute`
- **Data Type**: int32
- **Description**: Scheduled arrival minute of the hour (0-59).
- **Source Field(s)**: scheduled_arr_time
- **Calculation Logic**: (scheduled_arr_time % 100) % 60
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Computed safely with modulo 60 if column present.

### `arrival_minutes_since_midnight`
- **Data Type**: int32
- **Description**: Total elapsed minutes from midnight to scheduled arrival (0-1439).
- **Source Field(s)**: arrival_hour, arrival_minute
- **Calculation Logic**: arrival_hour * 60 + arrival_minute
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Complete deterministic calculation if column present.

### `year`
- **Data Type**: int32
- **Description**: Calendar year of scheduled flight.
- **Source Field(s)**: flight_date
- **Calculation Logic**: flight_date.dt.year
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Parsed from validated flight_date.

### `month`
- **Data Type**: int32
- **Description**: Calendar month of scheduled flight (1-12).
- **Source Field(s)**: flight_date
- **Calculation Logic**: flight_date.dt.month
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Parsed from validated flight_date.

### `day`
- **Data Type**: int32
- **Description**: Calendar day of month (1-31).
- **Source Field(s)**: flight_date
- **Calculation Logic**: flight_date.dt.day
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Parsed from validated flight_date.

### `day_of_week`
- **Data Type**: int32
- **Description**: Day of the week (0=Monday, 1=Tuesday, ..., 6=Sunday).
- **Source Field(s)**: flight_date
- **Calculation Logic**: flight_date.dt.dayofweek
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Parsed from validated flight_date.

### `week_of_year`
- **Data Type**: int32
- **Description**: ISO calendar week number (1-53).
- **Source Field(s)**: flight_date
- **Calculation Logic**: flight_date.dt.isocalendar().week
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Parsed from validated flight_date.

### `day_of_year`
- **Data Type**: int32
- **Description**: Day of the calendar year (1-366).
- **Source Field(s)**: flight_date
- **Calculation Logic**: flight_date.dt.dayofyear
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Parsed from validated flight_date.

### `is_weekend`
- **Data Type**: int32
- **Description**: Binary indicator for weekend flight (1 if Saturday or Sunday, else 0).
- **Source Field(s)**: day_of_week
- **Calculation Logic**: 1 if day_of_week in [5, 6] else 0
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Complete deterministic calculation.

### `departure_hour_sin`
- **Data Type**: float64
- **Description**: Sine projection of scheduled departure hour on unit circle (period 24).
- **Source Field(s)**: departure_hour
- **Calculation Logic**: sin(2 * pi * departure_hour / 24)
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Mathematically bounded in [-1.0, 1.0].

### `departure_hour_cos`
- **Data Type**: float64
- **Description**: Cosine projection of scheduled departure hour on unit circle (period 24).
- **Source Field(s)**: departure_hour
- **Calculation Logic**: cos(2 * pi * departure_hour / 24)
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Mathematically bounded in [-1.0, 1.0].

### `day_of_week_sin`
- **Data Type**: float64
- **Description**: Sine projection of day of week on unit circle (period 7).
- **Source Field(s)**: day_of_week
- **Calculation Logic**: sin(2 * pi * day_of_week / 7)
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Mathematically bounded in [-1.0, 1.0].

### `day_of_week_cos`
- **Data Type**: float64
- **Description**: Cosine projection of day of week on unit circle (period 7).
- **Source Field(s)**: day_of_week
- **Calculation Logic**: cos(2 * pi * day_of_week / 7)
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Mathematically bounded in [-1.0, 1.0].

### `month_sin`
- **Data Type**: float64
- **Description**: Sine projection of calendar month on unit circle (period 12).
- **Source Field(s)**: month
- **Calculation Logic**: sin(2 * pi * (month - 1) / 12)
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Mathematically bounded in [-1.0, 1.0].

### `month_cos`
- **Data Type**: float64
- **Description**: Cosine projection of calendar month on unit circle (period 12).
- **Source Field(s)**: month
- **Calculation Logic**: cos(2 * pi * (month - 1) / 12)
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Mathematically bounded in [-1.0, 1.0].

### `distance`
- **Data Type**: float64
- **Description**: Great-circle statute distance between origin and destination airports in miles.
- **Source Field(s)**: Raw DISTANCE
- **Calculation Logic**: Validated non-negative distance in miles.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Validated in Phase 1 (> 0 and <= 10,000 miles).

### `haul_category`
- **Data Type**: string (categorical)
- **Description**: Flight distance haul category: short_haul (<500 mi), medium_haul (500-1500 mi), long_haul (>1500 mi).
- **Source Field(s)**: distance
- **Calculation Logic**: Standard FAA distance binning.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: Binned from validated distance.

### `historical_origin_delay_rate`
- **Data Type**: float64
- **Description**: Historical delay rate of the origin airport strictly prior to prediction time.
- **Source Field(s)**: Prior flights from origin_airport
- **Calculation Logic**: Uses only observations strictly before prediction_time. If count >= min_history, sum(target)/count; else global_prior fallback.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE (Strict temporal filter: observation_timestamp < scheduled_departure)
- **Missing Value Handling**: Fallback to global prior rate if prior observations < min_history.

### `historical_origin_flight_count`
- **Data Type**: int32
- **Description**: Number of prior historical flights observed from origin airport strictly before prediction time.
- **Source Field(s)**: Prior flights from origin_airport
- **Calculation Logic**: Uses only observations strictly before prediction_time.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: 0 if no prior flights recorded.

### `historical_destination_delay_rate`
- **Data Type**: float64
- **Description**: Historical delay rate of the destination airport strictly prior to prediction time.
- **Source Field(s)**: Prior flights to dest_airport
- **Calculation Logic**: Uses only observations strictly before prediction_time. If count >= min_history, sum(target)/count; else global_prior fallback.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE (Strict temporal filter: observation_timestamp < scheduled_departure)
- **Missing Value Handling**: Fallback to global prior rate if prior observations < min_history.

### `historical_destination_flight_count`
- **Data Type**: int32
- **Description**: Number of prior historical flights observed to destination airport strictly before prediction time.
- **Source Field(s)**: Prior flights to dest_airport
- **Calculation Logic**: Uses only observations strictly before prediction_time.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: 0 if no prior flights recorded.

### `historical_airline_delay_rate`
- **Data Type**: float64
- **Description**: Historical delay rate of the operating carrier strictly prior to prediction time.
- **Source Field(s)**: Prior flights by airline
- **Calculation Logic**: Uses only observations strictly before prediction_time. If count >= min_history, sum(target)/count; else global_prior fallback.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE (Strict temporal filter: observation_timestamp < scheduled_departure)
- **Missing Value Handling**: Fallback to global prior rate if prior observations < min_history.

### `historical_airline_flight_count`
- **Data Type**: int32
- **Description**: Number of prior historical flights observed for airline strictly before prediction time.
- **Source Field(s)**: Prior flights by airline
- **Calculation Logic**: Uses only observations strictly before prediction_time.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: 0 if no prior flights recorded.

### `historical_route_delay_rate`
- **Data Type**: float64
- **Description**: Historical delay rate for specific origin-destination route strictly prior to prediction time.
- **Source Field(s)**: Prior flights on route
- **Calculation Logic**: Uses only observations strictly before prediction_time. If count >= min_history, sum(target)/count; else global_prior fallback.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE (Strict temporal filter: observation_timestamp < scheduled_departure)
- **Missing Value Handling**: Fallback to global prior rate if prior observations < min_history.

### `historical_route_flight_count`
- **Data Type**: int32
- **Description**: Number of prior historical flights observed on route strictly before prediction time.
- **Source Field(s)**: Prior flights on route
- **Calculation Logic**: Uses only observations strictly before prediction_time.
- **Available at Prediction Time**: YES
- **Leakage Risk**: NONE
- **Missing Value Handling**: 0 if no prior flights recorded.

### `delay_target`
- **Data Type**: int32 (Binary label)
- **Description**: Official FAA/BTS arrival delay classification ground-truth target: 1 if arrival_delay >= 15 min, else 0.
- **Source Field(s)**: Raw arrival_delay (isolated during Phase 1 preprocessing)
- **Calculation Logic**: 1 if arrival_delay >= 15 else 0
- **Available at Prediction Time**: GROUND TRUTH LABEL (NOT an input feature for inference)
- **Leakage Risk**: Ground-truth target; isolated as supervised training label.
- **Missing Value Handling**: Missing or corrupted arrival delays segregated during Phase 1 validation.
