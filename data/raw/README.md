# Raw Flight Datasets Guide

This directory (`data/raw/`) is designated for storing external raw flight datasets used for the Airline Delay Prediction & Operations Analytics pipeline.

Raw data files (`.csv`, `.parquet`, `.zip`, `.gz`) are excluded from Git tracking via `.gitignore` to prevent committing large binary artifacts to version control.

---

## 1. Supported Public Dataset Sources

The ingestion pipeline is designed to be **schema-adaptive** and works natively with historical flight datasets from major public aviation authorities:

### Primary Source: US DOT Bureau of Transportation Statistics (BTS) TranStats
* **Name**: Airline On-Time Performance Data (Reporting Carrier On-Time Performance)
* **Provider**: U.S. Department of Transportation, Bureau of Transportation Statistics (BTS)
* **URL**: [https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ)
* **License**: U.S. Government Public Domain
* **Description**: Official monthly on-time performance records filed by domestic air carriers reporting departure/arrival delays, flight cancellations, diversions, flight durations, and airport pairs.

### Secondary Source: Kaggle 2015 Flight Delays and Cancellations
* **Name**: 2015 Flight Delays and Cancellations (US DOT / BTS extract)
* **Provider**: Kaggle / US Department of Transportation
* **URL**: [https://www.kaggle.com/datasets/usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays)
* **Files**: `flights.csv`, `airlines.csv`, `airports.csv`
* **License**: CC0: Public Domain

---

## 2. Expected Conceptual Schema & Column Mappings

The ingestion engine (`src/data/ingestion.py`) dynamically maps raw columns to standard canonical names. You can supply datasets matching any of the following conventions:

| Conceptual Field | Canonical Name | BTS TranStats Variant | Kaggle 2015 Variant | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Flight Date** | `flight_date` | `FL_DATE`, `FlightDate` | `YEAR` + `MONTH` + `DAY` or `FLIGHT_DATE` | Date of the flight (YYYY-MM-DD) |
| **Airline** | `airline` | `OP_UNIQUE_CARRIER`, `OP_CARRIER`, `CARRIER` | `AIRLINE` | 2-letter IATA carrier code (e.g. AA, DL, UA) |
| **Origin Airport** | `origin_airport` | `ORIGIN`, `ORIGIN_AIRPORT_ID` | `ORIGIN_AIRPORT` | 3-letter IATA origin airport code (e.g. ATL, ORD) |
| **Destination Airport**| `dest_airport` | `DEST`, `DEST_AIRPORT_ID` | `DESTINATION_AIRPORT` | 3-letter IATA destination airport code |
| **Scheduled Departure**| `scheduled_dep_time`| `CRS_DEP_TIME` | `SCHEDULED_DEPARTURE` | Scheduled local departure time (HHMM / 24-hr) |
| **Actual Departure** | `actual_dep_time` | `DEP_TIME` | `DEPARTURE_TIME` | *Post-flight operational field (Leakage)* |
| **Departure Delay** | `departure_delay` | `DEP_DELAY` | `DEPARTURE_DELAY` | *Post-flight operational field (Leakage)* |
| **Scheduled Arrival** | `scheduled_arr_time`| `CRS_ARR_TIME` | `SCHEDULED_ARRIVAL` | Scheduled local arrival time (HHMM / 24-hr) |
| **Actual Arrival** | `actual_arr_time` | `ARR_TIME` | `ARRIVAL_TIME` | *Post-flight operational field (Leakage)* |
| **Arrival Delay** | `arrival_delay` | `ARR_DELAY` | `ARRIVAL_DELAY` | **Prediction Target Source** ($\ge 15$ min) |
| **Distance** | `distance` | `DISTANCE` | `DISTANCE` | Flight distance in miles |
| **Cancelled Flag** | `cancelled` | `CANCELLED` | `CANCELLED` | 1 if cancelled, 0 otherwise |
| **Diverted Flag** | `diverted` | `DIVERTED` | `DIVERTED` | 1 if diverted, 0 otherwise |
| **Air Time** | `air_time` | `AIR_TIME` | `AIR_TIME` | *Post-flight operational field (Leakage)* |
| **Taxi Out / In** | `taxi_out`, `taxi_in`| `TAXI_OUT`, `TAXI_IN` | `TAXI_OUT`, `TAXI_IN` | *Post-flight operational fields (Leakage)* |

---

## 3. Minimum Required Conceptual Fields

For any dataset to be valid, it must contain at least:
1. `flight_date` (or date components: `year`, `month`, `day`)
2. `airline` (carrier identifier)
3. `origin_airport` (origin identifier)
4. `dest_airport` (destination identifier)
5. `scheduled_dep_time` (scheduled departure time)
6. `arrival_delay` (ground truth arrival delay in minutes)

---

## 4. How to Download and Place the Dataset

1. Download the historical dataset from BTS TranStats or Kaggle.
2. If downloaded as a ZIP file, extract it locally.
3. Move or copy the dataset file into this directory:
   ```bash
   # Place your raw CSV or Parquet file here:
   data/raw/flights_raw.csv
   # OR
   data/raw/flights.csv
   ```
4. Verify the file exists:
   ```powershell
   dir data\raw
   ```

---

## 5. Development Sample vs. Full Dataset Policy

* **Full Dataset**: BTS annual data typically contains 6M+ flights per year. Full files are ideal for high-performance servers and deep historical backtesting.
* **Development Sample**: For rapid local experimentation, unit testing, and CI/CD pipelines, a representative slice (e.g. 5,000 to 50,000 records) extracted from the public dataset can be placed in `data/raw/flights_dev_sample.csv`.
* **Important Attribution Rule**: Any development sample created or used in this directory must be explicitly documented as a sampled subset of the official BTS/Kaggle dataset, including the extraction date, sample size, and filtering criteria. No synthetic or fabricated data may be presented as genuine historical records.
