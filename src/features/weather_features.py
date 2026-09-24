"""Weather integration foundation module for flight delay prediction (Phase 2A & Phase 3).

================================================================================
CRITICAL DESIGN & TEMPORAL CONSTRAINTS:
================================================================================
1. Prediction-Time Availability Contract:
   - Observed Weather: METAR / surface station observations captured at or
     before scheduled departure time (observation_time <= scheduled_departure).
   - Forecast Weather: TAF or meteorological model forecasts issued at or
     before scheduled departure time (forecast_issue_time <= scheduled_departure).
   - Historical Weather: Strictly prior station records.

   NEVER use:
     - Weather observations recorded after scheduled departure.
     - Forecasts issued after scheduled departure.
     - En-route or arrival weather observed after the departure prediction point.

2. Real Data Strategy (Zero-Fabrication):
   - No fabricated official weather records.
   - If data/external/ contains no real weather dataset, the system reports:
     "WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED"
     and leaves the feature matrix unpolluted by synthetic weather.
================================================================================
"""

from dataclasses import dataclass
from pathlib import Path
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

    OPTIONAL_EXTENDED_COLUMNS: List[str] = [
        "feels_like",
        "wind_gust",
        "precipitation_probability",
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
        """Validate external weather dataset schema and domain ranges."""
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

        # Check airport codes (3-letter IATA)
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

    def derive_severe_weather_indicators(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Extract deterministic severe weather boolean indicator flags.

        Indicators derived:
            - is_rain / rain_flag: 1 if precipitation > 0 or condition mentions rain/drizzle
            - is_snow / snow_flag: 1 if condition mentions snow/blizzard/wintry or temp <= 0 with precip
            - is_fog / fog_flag: 1 if visibility < 1.0 or condition mentions fog/mist
            - is_storm / storm_flag: 1 if condition mentions storm/thunder/tstm or wind_speed > 35 knots
            - is_low_visibility / low_visibility_flag: 1 if visibility < 3.0 statute miles (aviation IFR minimum)
            - is_severe_weather / severe_weather_flag: 1 if storm, snow, low visibility, or high wind
        """
        out = pd.DataFrame(index=weather_df.index)

        def _find_series(keywords: List[str], default_val: float) -> pd.Series:
            for kw in keywords:
                for c in weather_df.columns:
                    if kw in c.lower():
                        return pd.to_numeric(weather_df[c], errors="coerce").fillna(default_val)
            return pd.Series(default_val, index=weather_df.index)

        def _find_str_series(keywords: List[str]) -> pd.Series:
            for kw in keywords:
                for c in weather_df.columns:
                    if kw in c.lower():
                        return weather_df[c].astype(str).str.lower()
            return pd.Series("", index=weather_df.index)

        cond_str = _find_str_series(["weather_condition", "condition"])
        precip = _find_series(["precipitation", "precip"], 0.0)
        temp = _find_series(["temperature", "temp"], 20.0)
        vis = _find_series(["visibility"], 10.0)
        wind = _find_series(["wind_speed", "wind"], 0.0)
        snow = _find_series(["snow"], 0.0)

        rain_flag = ((precip > 0.0) | cond_str.str.contains("rain|drizzle|shwr", regex=True)).astype(int)
        snow_flag = (
            (snow > 0.0)
            | cond_str.str.contains("snow|blizzard|sleet|wintry", regex=True)
            | ((temp <= 0.0) & (precip > 0.0))
        ).astype(int)
        fog_flag = ((vis < 1.0) | cond_str.str.contains("fog|mist|haze", regex=True)).astype(int)
        storm_flag = (
            cond_str.str.contains("storm|thunder|tstm|squall", regex=True)
            | (wind > 35.0)
        ).astype(int)
        low_vis_flag = (vis < 3.0).astype(int)
        severe_flag = (
            (storm_flag == 1)
            | (snow_flag == 1)
            | (low_vis_flag == 1)
            | (wind > 30.0)
        ).astype(int)

        out["is_rain"] = out["rain_flag"] = rain_flag
        out["is_snow"] = out["snow_flag"] = snow_flag
        out["is_fog"] = out["fog_flag"] = fog_flag
        out["is_storm"] = out["storm_flag"] = storm_flag
        out["is_low_visibility"] = out["low_visibility_flag"] = low_vis_flag
        out["is_severe_weather"] = out["severe_weather_flag"] = severe_flag

        for pfx in ["origin_", "dest_", "weather_origin_", "weather_dest_"]:
            if any(c.startswith(pfx) for c in weather_df.columns):
                out[f"{pfx}is_rain"] = out[f"{pfx}rain_flag"] = rain_flag
                out[f"{pfx}is_snow"] = out[f"{pfx}snow_flag"] = snow_flag
                out[f"{pfx}is_fog"] = out[f"{pfx}fog_flag"] = fog_flag
                out[f"{pfx}is_storm"] = out[f"{pfx}storm_flag"] = storm_flag
                out[f"{pfx}is_low_visibility"] = out[f"{pfx}low_visibility_flag"] = low_vis_flag
                out[f"{pfx}is_severe_weather"] = out[f"{pfx}severe_weather_flag"] = severe_flag

        return out

    def join_nearest_prior_weather(
        self,
        flights_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        flight_airport_col: str = "origin_airport",
        flight_time_col: str = "_scheduled_dep_ts",
        prefix: str = "weather_origin_",
        lookback_minutes: Optional[int] = None,
    ) -> pd.DataFrame:
        """Join weather observation using exact airport and nearest strictly-prior timestamp.

        Prediction-Time Availability Contract:
            weather_observation_timestamp <= scheduled_departure_timestamp

        Observations occurring AFTER scheduled departure are strictly excluded.
        """
        max_lookback = lookback_minutes if lookback_minutes is not None else self.lookback_minutes
        lookback_delta = pd.Timedelta(minutes=max_lookback)

        resolved_airport_col = flight_airport_col
        if resolved_airport_col not in flights_df.columns:
            if "origin" in flights_df.columns and "origin" in flight_airport_col:
                resolved_airport_col = "origin"
            elif "dest" in flights_df.columns and "dest" in flight_airport_col:
                resolved_airport_col = "dest"
            elif "origin_airport" in flights_df.columns:
                resolved_airport_col = "origin_airport"

        if flight_time_col in flights_df.columns:
            dep_ts_series = pd.to_datetime(flights_df[flight_time_col])
        else:
            dep_col = "scheduled_dep_time" if "scheduled_dep_time" in flights_df.columns else "crs_dep_time"
            date_col = "flight_date" if "flight_date" in flights_df.columns else "fl_date"
            from src.features.historical_features import construct_departure_timestamp
            dep_ts_series = construct_departure_timestamp(flights_df, date_col=date_col, dep_time_col=dep_col)

        f_copy = pd.DataFrame({
            "flight_idx": flights_df.index,
            "airport": flights_df[resolved_airport_col].astype(str).str.strip().str.upper(),
            "dep_ts": dep_ts_series,
        })

        w_copy = weather_df.copy()
        w_airport_col = "airport" if "airport" in w_copy.columns else ("station_id" if "station_id" in w_copy.columns else "origin")
        w_ts_col = "timestamp" if "timestamp" in w_copy.columns else ("weather_timestamp" if "weather_timestamp" in w_copy.columns else next((c for c in w_copy.columns if "time" in c.lower()), "timestamp"))

        w_copy["airport"] = w_copy[w_airport_col].astype(str).str.strip().str.upper()
        w_copy["w_ts"] = pd.to_datetime(w_copy[w_ts_col])

        # Base weather metrics
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
        # Include optional fields if present
        for opt in self.OPTIONAL_EXTENDED_COLUMNS:
            if opt in w_copy.columns:
                feature_cols.append(opt)

        # Include any custom columns in weather_df (e.g. temperature_c, etc.)
        for c in w_copy.columns:
            if c not in ["airport", "station_id", "timestamp", "weather_timestamp", "w_ts", "flight_idx"] and c not in feature_cols:
                feature_cols.append(c)

        # Derive severe indicators
        indicators = self.derive_severe_weather_indicators(w_copy)
        for col in indicators.columns:
            w_copy[col] = indicators[col]
            if col not in feature_cols:
                feature_cols.append(col)

        joined_results = pd.DataFrame(index=flights_df.index)
        for col in feature_cols:
            if col in w_copy.columns:
                target_dtype = w_copy[col].dtype
                joined_results[f"{prefix}{col}"] = pd.Series(index=flights_df.index, dtype=target_dtype)
            else:
                joined_results[f"{prefix}{col}"] = np.nan

        # Missing indicator
        joined_results[f"{prefix}missing_flag"] = 1

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
                direction="backward",  # Strict invariant: w_ts <= dep_ts
                tolerance=lookback_delta,
            )

            for col in feature_cols:
                if col in merged.columns:
                    col_name = f"{prefix}{col}"
                    joined_results.loc[merged["flight_idx"], col_name] = merged[col].to_numpy()

            # Where join succeeded, clear missing flag
            matched_indices = merged.loc[merged["w_ts"].notna(), "flight_idx"]
            joined_results.loc[matched_indices, f"{prefix}missing_flag"] = 0

        # Provide origin_ aliases if prefix is weather_origin_
        if prefix == "weather_origin_":
            for col in feature_cols:
                if f"weather_origin_{col}" in joined_results.columns:
                    joined_results[f"origin_{col}"] = joined_results[f"weather_origin_{col}"]
            joined_results["origin_weather_observed_missing"] = joined_results["weather_origin_missing_flag"]

        logger.info(
            "Completed weather join for %s (lookback=%d min). Valid matches: %d / %d flights.",
            prefix,
            max_lookback,
            (joined_results[f"{prefix}missing_flag"] == 0).sum(),
            len(flights_df),
        )
        return joined_results
