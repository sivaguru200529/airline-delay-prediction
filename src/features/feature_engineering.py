"""Feature engineering module and temporal leakage validation.

================================================================================
CRITICAL DESIGN SPECIFICATION & TEMPORAL CONSTRAINTS:
================================================================================

1. Prediction Scenario:
   The system predicts whether a flight will experience an arrival delay of >= 15 min
   using ONLY information available strictly at or before scheduled departure:
       prediction_time = scheduled_departure

2. Strict Temporal Integrity Policy:
   All future historical and rolling features (such as airline historical delay rate,
   origin airport delay rate, route delay rate, and recent inbound aircraft delay)
   MUST obey:
       observation_timestamp < prediction_time

   NO FUTURE FLIGHT RECORDS MAY CONTRIBUTE TO ANY FEATURE CALCULATION.
   In Phase 2, rolling windows and historical statistics must be computed using
   expanding or sliding time windows evaluated strictly up to prediction_time.

3. Weather Integration Strategy (Phase 2):
   Weather features (temperature, precipitation, wind speed, visibility, etc.)
   will be joined using an exact or nearest-prior match strategy:
       (origin_airport, weather_observation_timestamp)
   with explicit local-to-UTC timezone reconciliation. Forecast or METAR weather
   must be dated strictly BEFORE scheduled departure.
================================================================================
"""

from typing import Dict, List, Optional
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("feature_engineering")


def verify_temporal_validity(
    prediction_time: pd.Timestamp,
    historical_timestamps: pd.Series,
) -> bool:
    """Verify that all historical records strictly precede the prediction timestamp.

    This helper enforces the anti-leakage invariant:
        historical_timestamp < prediction_time

    Args:
        prediction_time: Scheduled departure timestamp for the flight being predicted.
        historical_timestamps: Series of observation timestamps used in historical aggregation.

    Returns:
        True if all historical timestamps are strictly before prediction_time.

    Raises:
        ValueError: If any historical timestamp >= prediction_time is detected.
    """
    ts_series = pd.to_datetime(historical_timestamps)
    pred_ts = pd.to_datetime(prediction_time)

    future_records = ts_series >= pred_ts
    if future_records.any():
        num_future = int(future_records.sum())
        max_future_ts = ts_series[future_records].max()
        error_msg = (
            f"Temporal Leakage Violation! Detected {num_future} historical observations "
            f"occurring at or after prediction_time ({pred_ts}). Max future timestamp: {max_future_ts}. "
            f"All features must satisfy: observation_timestamp < prediction_time."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return True


# ==============================================================================
# PHASE 2 INTERFACE PLACEHOLDERS
# ==============================================================================

def build_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar and schedule attributes available at booking/dispatch.
    
    Planned Phase 2 Features:
        - departure_hour (0-23)
        - departure_minute (0-59)
        - day_of_week (1-7)
        - month (1-12)
        - week_of_year (1-52)
        - is_weekend (0 or 1)
        - is_holiday (US federal holiday indicator)
    """
    raise NotImplementedError(
        "Schedule feature engineering is planned for Phase 2."
    )


def build_historical_delay_rates(
    df: pd.DataFrame,
    window_days: int = 30,
) -> pd.DataFrame:
    """Compute time-aware historical delay rates for airlines, airports, and routes.
    
    Planned Phase 2 Features:
        - airline_hist_delay_rate (Trailing rolling delay rate)
        - origin_hist_delay_rate (Trailing airport congestion/delay rate)
        - route_hist_delay_rate (Origin -> Dest historical performance)
    
    Enforces:
        observation_time < scheduled_departure
    """
    raise NotImplementedError(
        "Historical delay rate calculation is planned for Phase 2."
    )


def integrate_weather_features(
    flights_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join pre-departure weather observations on (origin_airport, timestamp).
    
    Planned Phase 2 Features:
        - temperature, precipitation, wind_speed, visibility, weather_condition
    """
    raise NotImplementedError(
        "Weather feature integration is planned for Phase 2."
    )
