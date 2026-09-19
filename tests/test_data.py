"""Unit tests for data ingestion, schema mapping, validation, and leakage prevention."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.utils.config import get_config
from src.data.ingestion import FlightDataIngestor
from src.data.validation import DataQualityValidator
from src.data.preprocessing import FlightDataPreprocessor


@pytest.fixture
def sample_bts_dataframe():
    """Fixture providing sample DataFrame using standard BTS TranStats schema."""
    return pd.DataFrame({
        "FL_DATE": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
        "OP_UNIQUE_CARRIER": ["AA", "DL", "UA", "WN", "B6"],
        "ORIGIN": ["JFK", "ATL", "ORD", "DFW", "DEN"],
        "DEST": ["LAX", "MIA", "SFO", "SEA", "BOS"],
        "CRS_DEP_TIME": [800, 930, 1200, 1415, 1800],
        "DEP_TIME": [802, 928, 1225, 1420, 1845],
        "DEP_DELAY": [2.0, -2.0, 25.0, 5.0, 45.0],
        "CRS_ARR_TIME": [1130, 1145, 1430, 1630, 1930],
        "ARR_TIME": [1144, 1140, 1500, 1640, 2030],
        "ARR_DELAY": [14.0, -5.0, 30.0, 10.0, 60.0],
        "DISTANCE": [2475, 594, 1846, 1660, 1754],
        "CANCELLED": [0, 0, 0, 0, 0],
        "DIVERTED": [0, 0, 0, 0, 0],
        "AIR_TIME": [320, 110, 250, 230, 240],
        "TAXI_OUT": [15, 12, 18, 14, 20],
        "TAXI_IN": [7, 6, 8, 6, 10],
    })


@pytest.fixture
def sample_kaggle_dataframe():
    """Fixture providing sample DataFrame using Kaggle 2015 schema."""
    return pd.DataFrame({
        "YEAR": [2015, 2015, 2015],
        "MONTH": [1, 1, 1],
        "DAY": [15, 15, 15],
        "AIRLINE": ["AA", "UA", "DL"],
        "ORIGIN_AIRPORT": ["DFW", "ORD", "ATL"],
        "DESTINATION_AIRPORT": ["ORD", "LAX", "MCO"],
        "SCHEDULED_DEPARTURE": [700, 1030, 1500],
        "DEPARTURE_TIME": [705, 1032, 1545],
        "DEPARTURE_DELAY": [5.0, 2.0, 45.0],
        "SCHEDULED_ARRIVAL": [930, 1300, 1645],
        "ARRIVAL_TIME": [944, 1315, 1750],
        "ARRIVAL_DELAY": [14.0, 15.0, 65.0],
        "DISTANCE": [802, 1744, 404],
        "CANCELLED": [0, 0, 0],
        "DIVERTED": [0, 0, 0],
    })


# ==============================================================================
# 1. TARGET GENERATION TESTS
# ==============================================================================

def test_target_generation_thresholds():
    """Verify target definition:
        arrival_delay = 14 -> 0
        arrival_delay = 15 -> 1
        arrival_delay = 30 -> 1
        arrival_delay = -5 -> 0
    """
    preprocessor = FlightDataPreprocessor()
    test_df = pd.DataFrame({
        "arrival_delay": [14.0, 15.0, 30.0, -5.0, 0.0, 14.9, 15.1]
    })
    targets = preprocessor.create_target_variable(test_df, delay_col="arrival_delay")

    assert targets.iloc[0] == 0, "14 min delay must be classified as on-time (0)"
    assert targets.iloc[1] == 1, "15 min delay must be classified as delayed (1)"
    assert targets.iloc[2] == 1, "30 min delay must be classified as delayed (1)"
    assert targets.iloc[3] == 0, "Negative delay must be classified as on-time (0)"
    assert targets.iloc[4] == 0, "0 min delay must be classified as on-time (0)"
    assert targets.iloc[5] == 0, "14.9 min delay must be classified as on-time (0)"
    assert targets.iloc[6] == 1, "15.1 min delay must be classified as delayed (1)"


# ==============================================================================
# 2. DATA LEAKAGE PREVENTION TESTS
# ==============================================================================

def test_leakage_prevention_columns_excluded():
    """Verify that all post-flight leakage fields are purged from pre-departure features:
        - arrival_delay
        - departure_delay
        - actual_departure / actual_dep_time
        - actual_arrival / actual_arr_time
        - taxi_out
        - taxi_in
        - wheels_off
        - wheels_on
        - air_time
    """
    preprocessor = FlightDataPreprocessor()
    operational_df = pd.DataFrame({
        "flight_date": ["2024-01-01"],
        "airline": ["AA"],
        "origin_airport": ["JFK"],
        "dest_airport": ["LAX"],
        "scheduled_dep_time": [800],
        "distance": [2475],
        "delay_target": [1],
        # Post-flight leakage fields:
        "arrival_delay": [45.0],
        "departure_delay": [30.0],
        "actual_dep_time": [830],
        "actual_arr_time": [1215],
        "taxi_out": [20],
        "taxi_in": [10],
        "wheels_off": [850],
        "wheels_on": [1205],
        "air_time": [315],
    })

    pre_dep_df = preprocessor.extract_pre_departure_features(operational_df, include_target=True)

    forbidden_columns = [
        "arrival_delay",
        "departure_delay",
        "actual_departure",
        "actual_arrival",
        "actual_dep_time",
        "actual_arr_time",
        "taxi_out",
        "taxi_in",
        "wheels_off",
        "wheels_on",
        "air_time",
    ]

    for col in forbidden_columns:
        assert col not in pre_dep_df.columns, (
            f"Leakage violation! Post-flight column '{col}' found in pre-departure features."
        )

    # Ensure pre-departure features and target remain
    assert "airline" in pre_dep_df.columns
    assert "origin_airport" in pre_dep_df.columns
    assert "dest_airport" in pre_dep_df.columns
    assert "scheduled_dep_time" in pre_dep_df.columns
    assert "distance" in pre_dep_df.columns
    assert "delay_target" in pre_dep_df.columns


# ==============================================================================
# 3. SCHEMA MAPPING TESTS
# ==============================================================================

def test_bts_schema_mapping(sample_bts_dataframe):
    """Verify dynamic schema mapping for BTS TranStats columns."""
    ingestor = FlightDataIngestor()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample_bts_dataframe.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        result = ingestor.load_dataset(tmp_path)
        mapped_df = result.mapped_df

        assert "flight_date" in mapped_df.columns
        assert "airline" in mapped_df.columns
        assert "origin_airport" in mapped_df.columns
        assert "dest_airport" in mapped_df.columns
        assert "scheduled_dep_time" in mapped_df.columns
        assert "arrival_delay" in mapped_df.columns
        assert len(result.missing_required_columns) == 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_kaggle_schema_mapping(sample_kaggle_dataframe):
    """Verify dynamic schema mapping and date synthesis for Kaggle 2015 format."""
    ingestor = FlightDataIngestor()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample_kaggle_dataframe.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        result = ingestor.load_dataset(tmp_path)
        mapped_df = result.mapped_df

        assert "flight_date" in mapped_df.columns
        assert mapped_df["flight_date"].iloc[0] == "2015-01-15"
        assert "airline" in mapped_df.columns
        assert "origin_airport" in mapped_df.columns
        assert "dest_airport" in mapped_df.columns
        assert "arrival_delay" in mapped_df.columns
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_missing_required_columns_raises_error():
    """Verify that omitting required conceptual columns raises a clear ValueError."""
    ingestor = FlightDataIngestor()
    incomplete_df = pd.DataFrame({
        "carrier": ["AA"],
        "origin": ["JFK"],
        # Missing destination, flight_date, scheduled_dep_time, arrival_delay
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        incomplete_df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError) as excinfo:
            ingestor.load_dataset(tmp_path)
        assert "Missing required conceptual fields" in str(excinfo.value)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ==============================================================================
# 4. DATA VALIDATION TESTS
# ==============================================================================

def test_data_quality_validation_checks():
    """Verify validator detects duplicates, bad airport codes, negative distances, invalid dates."""
    validator = DataQualityValidator()

    # Construct test data with intentional anomalies
    dirty_df = pd.DataFrame({
        "flight_date": [
            "2024-01-01",  # Valid
            "2024-01-01",  # Duplicate row
            "bad-date",    # Invalid date
            "2024-01-02",  # Invalid origin airport (length 4)
            "2024-01-02",  # Invalid negative distance
            "2024-01-03",  # Cancelled flight
            "2024-01-03",  # Diverted flight
        ],
        "airline": ["AA", "AA", "DL", "UA", "WN", "AA", "DL"],
        "origin_airport": ["JFK", "JFK", "ATL", "XXXX", "ORD", "DFW", "DEN"],
        "dest_airport": ["LAX", "LAX", "MIA", "SFO", "SEA", "BOS", "LAX"],
        "scheduled_dep_time": [800, 800, 900, 1000, 1100, 1200, 1300],
        "arrival_delay": [10.0, 10.0, 5.0, 20.0, 15.0, np.nan, np.nan],
        "distance": [2475, 2475, 594, 1846, -50, 1660, 1754],
        "cancelled": [0, 0, 0, 0, 0, 1, 0],
        "diverted": [0, 0, 0, 0, 0, 0, 1],
    })

    summary, annotated_df = validator.validate(dirty_df)

    assert summary.total_records == 7
    assert summary.duplicate_records == 1
    assert summary.invalid_date_records == 1
    assert summary.invalid_airport_records == 1
    assert summary.invalid_distance_records == 1
    assert summary.cancelled_flights == 1
    assert summary.diverted_flights == 1
    # Only row 0 is completely clean, valid, non-cancelled, non-diverted
    assert summary.total_valid_records == 1
    assert summary.ontime_flights == 1
    assert summary.delayed_flights == 0
