"""Unit tests for feature engineering temporal validity and interface constraints."""

import pandas as pd
import pytest

from src.features.feature_engineering import verify_temporal_validity


def test_temporal_validity_accepts_strictly_prior_observations():
    """Verify that historical timestamps occurring strictly before prediction_time pass."""
    prediction_time = pd.Timestamp("2024-06-15 10:00:00")
    historical_timestamps = pd.Series([
        pd.Timestamp("2024-06-01 08:00:00"),
        pd.Timestamp("2024-06-14 23:59:59"),
        pd.Timestamp("2024-06-15 09:59:00"),
    ])

    # Should succeed without error
    is_valid = verify_temporal_validity(prediction_time, historical_timestamps)
    assert is_valid is True


def test_temporal_validity_rejects_future_or_concurrent_observations():
    """Verify that any observation timestamp >= prediction_time raises a ValueError."""
    prediction_time = pd.Timestamp("2024-06-15 10:00:00")

    # Includes concurrent timestamp (10:00:00) and future timestamp (10:05:00)
    invalid_timestamps = pd.Series([
        pd.Timestamp("2024-06-15 08:00:00"),
        pd.Timestamp("2024-06-15 10:00:00"),  # Concurrent violation
        pd.Timestamp("2024-06-15 10:05:00"),  # Future violation
    ])

    with pytest.raises(ValueError) as excinfo:
        verify_temporal_validity(prediction_time, invalid_timestamps)

    assert "Temporal Leakage Violation" in str(excinfo.value)
    assert "observation_timestamp < prediction_time" in str(excinfo.value)
