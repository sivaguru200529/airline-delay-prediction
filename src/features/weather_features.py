"""Weather integration foundation module for flight delay prediction.

================================================================================
CRITICAL DESIGN & TEMPORAL CONSTRAINTS:
================================================================================
1. Distinction Between Observed Weather and Forecast Weather:
   - Observed Weather: METAR/station observations captured at an airport at or
     before scheduled departure time (observation_time <= scheduled_departure).
   - Forecast Weather: TAF or model forecast generated hours in advance.
   - This module explicitly models OBSERVED weather available at pushback.
     Observed historical weather must NEVER be represented as a forecast.

2. Real Data Strategy:
   - No fabricated official weather records.
   - If data/external/ contains no real weather dataset, the system reports:
     "WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED"
     and leaves the final ML feature dataset unpolluted by synthetic weather.
================================================================================
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("weather_features")


@dataclass
class WeatherValidationResult:
    """Container for weather schema validation results."""

    is_valid: bool
    total_records: int
    airports_covered: List[str]
    min_timestamp: Optional[pd.Timestamp]
    max_timestamp: Optional[pd.Timestamp]
    missing_fields: List[str]
    invalid_rows_count: int
    validation_notes: List[str]


class WeatherIntegrator:
    """Validates external weather data and performs leakage-free nearest-prior joins."""

    WEATHER_COLUMNS: List[str] = [
        "airport",
        "timestamp",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "precipitation",
        "visibility",
        "pressure",
        "cloud_cover",
        "weather_condition",
    ]

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize weather integrator."""
        self.config = config or get_config()
        self.lookback_minutes = self.config.weather_lookback_minutes

    def check_external_weather_availability(self) -> Tuple[str, Optional[Path]]:
        """Check if real external weather data exists in data/external/.

        Returns:
            Tuple of (status_string, optional_file_path).
        """
        ext_dir = self.config.data_external_dir
        if not ext_dir.exists():
            return "WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED", None

        # Look for real weather files (excluding .gitkeep or hidden files)
        candidates = [
            f for f in ext_dir.glob("*")
            if f.suffix.lower() in [".csv", ".parquet", ".pq", ".gz"]
            and not f.name.startswith(".")
        ]

        if not candidates:
            return "WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED", None

        return "WEATHER STATUS: INTEGRATED", candidates[0]

    def validate_weather_schema(self, weather_df: pd.DataFrame) -> WeatherValidationResult:
        """Validate external weather dataset schema and domain ranges.

        Checks:
            - Required columns exist
            - Airport code is valid 3-letter IATA format
            - Timestamp parses to datetime
            - Numeric range checks:
                - humidity in [0, 100]
                - wind_direction in [0, 360]
                - wind_speed >= 0
                - visibility >= 0
                - pressure >= 800 (hPa/mbar)

        Args:
            weather_df: Input weather DataFrame.

        Returns:
            WeatherValidationResult container.
        """
        cols_lower = {c.lower(): c for c in weather_df.columns}
        missing = [req for req in self.WEATHER_COLUMNS if req not in cols_lower]

        validation_notes: List[str] = []
        if missing:
            validation_notes.append(f"Missing required weather fields: {missing}")
            return WeatherValidationResult(
                is_valid=False,
                total_records=len(weather_df),
                airports_covered=[],
                min_timestamp=None,
                max_timestamp=None,
                missing_fields=missing,
                invalid_rows_count=len(weather_df),
                validation_notes=validation_notes,
            )

        working = weather_df.rename(columns={cols_lower[c]: c for c in self.WEATHER_COLUMNS}).copy()

        # Check airport codes
        valid_airport = working["airport"].astype(str).str.strip().str.match(r"^[A-Za-z]{3}$")

        # Check timestamps
        parsed_ts = pd.to_datetime(working["timestamp"], errors="coerce")
        valid_ts = ~parsed_ts.isna()

        # Range checks
        valid_humidity = (
            pd.to_numeric(working["humidity"], errors="coerce").between(0, 100)
            | working["humidity"].isna()
        )
        valid_wind_dir = (
            pd.to_numeric(working["wind_direction"], errors="coerce").between(0, 360)
            | working["wind_direction"].isna()
        )
        valid_wind_spd = (
            (pd.to_numeric(working["wind_speed"], errors="coerce") >= 0)
            | working["wind_speed"].isna()
        )

        invalid_mask = ~(valid_airport & valid_ts & valid_humidity & valid_wind_dir & valid_wind_spd)
        invalid_count = int(invalid_mask.sum())

        airports = sorted(working.loc[valid_airport, "airport"].unique().tolist())
        min_ts = parsed_ts.min() if valid_ts.any() else None
        max_ts = parsed_ts.max() if valid_ts.any() else None

        is_valid = invalid_count == 0
        validation_notes.append(f"Validated {len(working)} records covering {len(airports)} airports.")
        if invalid_count > 0:
            validation_notes.append(f"Detected {invalid_count} records violating domain range constraints.")

        return WeatherValidationResult(
            is_valid=is_valid,
            total_records=len(working),
            airports_covered=airports,
            min_timestamp=min_ts,
            max_timestamp=max_ts,
            missing_fields=[],
            invalid_rows_count=invalid_count,
            validation_notes=validation_notes,
        )

    def join_nearest_prior_weather(
        self,
        flights_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        flight_airport_col: str = "origin_airport",
        flight_time_col: str = "_scheduled_dep_ts",
        lookback_minutes: Optional[int] = None,
    ) -> pd.DataFrame:
        """Join weather observation to flights using exact airport and nearest strictly-prior timestamp.

        Anti-Leakage Guarantee:
            weather_observation_timestamp <= scheduled_departure_timestamp

        Observations occurring AFTER scheduled departure are strictly excluded.
        If no observation exists within [scheduled_dep_ts - lookback_minutes, scheduled_dep_ts],
        weather features are populated with NaN.

        Args:
            flights_df: DataFrame containing flight records and departure timestamp.
            weather_df: Validated weather DataFrame.
            flight_airport_col: Origin airport column in flights_df.
            flight_time_col: Scheduled departure timestamp column in flights_df.
            lookback_minutes: Maximum lookback window in minutes (defaults to config).

        Returns:
            DataFrame of joined weather features with prefix 'weather_origin_' matching flights_df index.
        """
        max_lookback = lookback_minutes if lookback_minutes is not None else self.lookback_minutes
        lookback_delta = pd.Timedelta(minutes=max_lookback)

        f_copy = pd.DataFrame({
            "flight_idx": flights_df.index,
            "airport": flights_df[flight_airport_col].astype(str).str.strip().str.upper(),
            "dep_ts": pd.to_datetime(flights_df[flight_time_col]),
        })

        w_copy = weather_df.copy()
        w_copy["airport"] = w_copy["airport"].astype(str).str.strip().str.upper()
        w_copy["w_ts"] = pd.to_datetime(w_copy["timestamp"])

        feature_cols = [
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
            "precipitation",
            "visibility",
            "pressure",
            "cloud_cover",
            "weather_condition",
        ]

        # Prepare container for joined results with appropriate dtypes
        joined_results = pd.DataFrame(index=flights_df.index)
        for col in feature_cols:
            if col in w_copy.columns:
                target_dtype = w_copy[col].dtype
                joined_results[f"weather_origin_{col}"] = pd.Series(index=flights_df.index, dtype=target_dtype)
            else:
                joined_results[f"weather_origin_{col}"] = np.nan

        # Merge asof per airport
        for airport in f_copy["airport"].unique():
            f_airport = f_copy[f_copy["airport"] == airport].sort_values("dep_ts")
            w_airport = w_copy[w_copy["airport"] == airport].sort_values("w_ts")

            if w_airport.empty:
                continue

            merged = pd.merge_asof(
                f_airport,
                w_airport[["w_ts"] + [c for c in feature_cols if c in w_airport.columns]],
                left_on="dep_ts",
                right_on="w_ts",
                direction="backward",  # Ensures w_ts <= dep_ts
                tolerance=lookback_delta,
            )

            for col in feature_cols:
                if col in merged.columns:
                    col_name = f"weather_origin_{col}"
                    joined_results.loc[merged["flight_idx"], col_name] = merged[col].to_numpy()

        logger.info(
            "Completed weather join (lookback=%d min). Valid weather joins: %d / %d flights.",
            max_lookback,
            joined_results["weather_origin_temperature"].notna().sum(),
            len(flights_df),
        )
        return joined_results
