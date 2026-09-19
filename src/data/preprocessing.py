"""Data preprocessing and leakage prevention module.

Handles data cleaning, standardization, binary target creation:
    delay_target = 1 if arrival_delay >= 15 else 0

Enforces strict separation between:
    1. Raw Operational Data (All recorded fields)
    2. Cleaned Operational Dataset (Standardized operational records for retrospective analysis)
    3. Pre-Departure Prediction View (STRICTLY excludes all post-flight leakage fields)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("data_preprocessing")


@dataclass
class PreprocessingResult:
    """Encapsulates results of data cleaning and leakage-free dataset generation."""

    cleaned_operational_df: pd.DataFrame
    pre_departure_df: pd.DataFrame
    cancelled_df: pd.DataFrame
    diverted_df: pd.DataFrame
    invalid_df: pd.DataFrame
    total_processed: int
    valid_completed_count: int
    delayed_count: int
    ontime_count: int
    delay_rate: float
    output_files: Dict[str, str]


class FlightDataPreprocessor:
    """Preprocesses flight data and enforces strict leakage prevention boundaries."""

    def __init__(self, config=None) -> None:
        """Initialize preprocessor with configuration."""
        self.config = config or get_config()

    def create_target_variable(
        self,
        df: pd.DataFrame,
        delay_col: str = "arrival_delay",
    ) -> pd.Series:
        """Construct the official binary classification target:
        
            delay_target = 1 if arrival_delay >= 15 minutes
            delay_target = 0 otherwise

        NOTE:
            arrival_delay is strictly the target variable ground truth.
            It MUST NEVER be included as an input feature for pre-departure prediction.

        Args:
            df: Input DataFrame containing delay_col.
            delay_col: Column name containing arrival delay in minutes.

        Returns:
            pd.Series of integers (0 or 1).
        """
        if delay_col not in df.columns:
            raise KeyError(f"Target creation failed: column '{delay_col}' not found in DataFrame.")

        numeric_delays = pd.to_numeric(df[delay_col], errors="coerce")
        target = (numeric_delays >= self.config.arrival_delay_threshold).astype(int)
        return target

    def extract_pre_departure_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Isolate pre-departure prediction features and strictly purge leakage columns.

        DATA LEAKAGE PREVENTION RULE:
        Prediction occurs at:
            prediction_time = scheduled_departure

        Therefore, any field known only after departure pushback, takeoff, airborne flight,
        or landing is classified as post-flight leakage and is strictly dropped from this view.

        Purged Leakage Columns:
            - arrival_delay (Ground truth prediction target)
            - departure_delay (Known only after actual pushback)
            - actual_departure / actual_dep_time
            - actual_arrival / actual_arr_time
            - taxi_out / taxi_in
            - wheels_off / wheels_on
            - air_time / elapsed_time

        Args:
            df: Cleaned operational DataFrame containing delay_target.
            include_target: Whether to retain 'delay_target' in the resulting feature DataFrame.

        Returns:
            DataFrame containing only pre-departure attributes (and optionally delay_target).
        """
        leakage_cols_to_drop = [
            col for col in self.config.post_flight_leakage_columns if col in df.columns
        ]
        # Also drop internal audit flags if present
        audit_cols = [c for c in df.columns if c.startswith("_is_")]

        cols_to_exclude = set(leakage_cols_to_drop + audit_cols)

        # Build clean pre-departure feature set
        feature_cols = [c for c in df.columns if c not in cols_to_exclude]

        pre_departure_df = df[feature_cols].copy()

        if include_target and "delay_target" in df.columns:
            pre_departure_df["delay_target"] = df["delay_target"].copy()

        logger.info(
            "Purged %d leakage-prone columns from pre-departure prediction view: %s",
            len(leakage_cols_to_drop),
            leakage_cols_to_drop,
        )
        return pre_departure_df

    def process(
        self,
        annotated_df: pd.DataFrame,
        save_output: bool = True,
    ) -> PreprocessingResult:
        """Run full preprocessing pipeline on validated DataFrame.

        Steps:
            1. Segregate cancelled, diverted, and invalid records into separate tracking sets.
            2. Extract valid completed flights.
            3. Standardize categorical attributes (airport codes uppercase/trimmed, airlines).
            4. Parse flight_date to standard datetime format.
            5. Generate binary delay_target.
            6. Construct leakage-free pre-departure view.
            7. Persist cleaned datasets to data/processed/.

        Args:
            annotated_df: DataFrame output by DataQualityValidator with audit flags.
            save_output: Whether to persist processed files to disk.

        Returns:
            PreprocessingResult container.
        """
        logger.info("Processing %d records through cleaning pipeline", len(annotated_df))

        # Separate cohorts based on validation audit flags
        is_invalid = annotated_df.get("_is_invalid", pd.Series(False, index=annotated_df.index))
        is_cancelled = annotated_df.get("_is_cancelled", pd.Series(False, index=annotated_df.index))
        is_diverted = annotated_df.get("_is_diverted", pd.Series(False, index=annotated_df.index))

        invalid_df = annotated_df[is_invalid].copy()
        cancelled_df = annotated_df[~is_invalid & is_cancelled].copy()
        diverted_df = annotated_df[~is_invalid & ~is_cancelled & is_diverted].copy()

        # Valid completed flights eligible for delay classification
        valid_mask = ~is_invalid & ~is_cancelled & ~is_diverted
        valid_df = annotated_df[valid_mask].copy()

        # Clean string formats
        if "origin_airport" in valid_df.columns:
            valid_df["origin_airport"] = valid_df["origin_airport"].astype(str).str.strip().str.upper()
        if "dest_airport" in valid_df.columns:
            valid_df["dest_airport"] = valid_df["dest_airport"].astype(str).str.strip().str.upper()
        if "airline" in valid_df.columns:
            valid_df["airline"] = valid_df["airline"].astype(str).str.strip().str.upper()

        # Parse dates
        if "flight_date" in valid_df.columns:
            valid_df["flight_date"] = pd.to_datetime(valid_df["flight_date"]).dt.strftime("%Y-%m-%d")

        # Convert distance and delays to numeric
        if "distance" in valid_df.columns:
            valid_df["distance"] = pd.to_numeric(valid_df["distance"], errors="coerce")
        if "arrival_delay" in valid_df.columns:
            valid_df["arrival_delay"] = pd.to_numeric(valid_df["arrival_delay"], errors="coerce")

        # Create binary delay target
        valid_df["delay_target"] = self.create_target_variable(valid_df, "arrival_delay")

        delayed_count = int((valid_df["delay_target"] == 1).sum())
        ontime_count = int((valid_df["delay_target"] == 0).sum())
        delay_rate = (delayed_count / max(1, len(valid_df))) * 100.0

        # Construct strict pre-departure view (zero leakage)
        pre_departure_df = self.extract_pre_departure_features(valid_df, include_target=True)

        output_files: Dict[str, str] = {}
        if save_output:
            self.config.data_processed_dir.mkdir(parents=True, exist_ok=True)

            # 1. Full cleaned operational dataset (with operational outcomes for retrospectives)
            clean_parquet = self.config.data_processed_dir / "flights_cleaned_operational.parquet"
            clean_csv = self.config.data_processed_dir / "flights_cleaned_operational.csv"
            valid_df.to_parquet(clean_parquet, index=False)
            valid_df.to_csv(clean_csv, index=False)
            output_files["cleaned_operational_parquet"] = str(clean_parquet)
            output_files["cleaned_operational_csv"] = str(clean_csv)

            # 2. Pre-departure prediction-ready dataset (zero leakage, only pre-flight features + target)
            pred_parquet = self.config.data_processed_dir / "flights_pre_departure.parquet"
            pred_csv = self.config.data_processed_dir / "flights_pre_departure.csv"
            pre_departure_df.to_parquet(pred_parquet, index=False)
            pre_departure_df.to_csv(pred_csv, index=False)
            output_files["pre_departure_parquet"] = str(pred_parquet)
            output_files["pre_departure_csv"] = str(pred_csv)

            logger.info("Saved processed datasets to %s", self.config.data_processed_dir)

        return PreprocessingResult(
            cleaned_operational_df=valid_df,
            pre_departure_df=pre_departure_df,
            cancelled_df=cancelled_df,
            diverted_df=diverted_df,
            invalid_df=invalid_df,
            total_processed=len(annotated_df),
            valid_completed_count=len(valid_df),
            delayed_count=delayed_count,
            ontime_count=ontime_count,
            delay_rate=round(delay_rate, 2),
            output_files=output_files,
        )
