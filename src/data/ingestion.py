"""Generic, schema-adaptive flight data ingestion module.

Loads CSV or Parquet historical flight datasets, dynamically resolves column schemas
(supporting BTS TranStats, Kaggle 2015, and custom formats), verifies required fields,
and flags post-flight leakage columns.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("data_ingestion")


@dataclass
class IngestionResult:
    """Container holding results of raw dataset ingestion and schema mapping."""

    raw_df: pd.DataFrame
    mapped_df: pd.DataFrame
    schema_mapping: Dict[str, str]  # canonical_name -> raw_column_name
    unmapped_columns: List[str]
    required_columns_found: List[str]
    missing_required_columns: List[str]
    detected_leakage_columns: List[str]
    total_records: int
    total_columns: int
    file_path: str


class FlightDataIngestor:
    """Adaptive flight data ingestor supporting multiple public dataset schemas."""

    # Dictionary of known aliases for canonical conceptual fields
    # Canonical name -> list of recognized alternative column headers
    SCHEMA_ALIASES: Dict[str, List[str]] = {
        "flight_date": [
            "flight_date", "fl_date", "flightdate", "date", "flight_day", "fl_date_str"
        ],
        "airline": [
            "airline", "op_unique_carrier", "op_carrier", "carrier", "reporting_airline",
            "airline_code", "unique_carrier"
        ],
        "origin_airport": [
            "origin_airport", "origin", "origin_airport_id", "origin_code", "orig"
        ],
        "dest_airport": [
            "dest_airport", "dest", "destination_airport", "dest_airport_id", "destination"
        ],
        "scheduled_dep_time": [
            "scheduled_dep_time", "crs_dep_time", "scheduled_departure", "sched_dep_time",
            "crs_departure_time"
        ],
        "actual_dep_time": [
            "actual_dep_time", "dep_time", "departure_time", "actual_departure"
        ],
        "departure_delay": [
            "departure_delay", "dep_delay", "dep_delay_new", "dep_delay_minutes"
        ],
        "scheduled_arr_time": [
            "scheduled_arr_time", "crs_arr_time", "scheduled_arrival", "sched_arr_time",
            "crs_arrival_time"
        ],
        "actual_arr_time": [
            "actual_arr_time", "arr_time", "arrival_time", "actual_arrival"
        ],
        "arrival_delay": [
            "arrival_delay", "arr_delay", "arr_delay_new", "arr_delay_minutes"
        ],
        "distance": [
            "distance", "dist", "flight_distance", "distance_miles"
        ],
        "cancelled": [
            "cancelled", "is_cancelled", "cancellation_code", "cancellation"
        ],
        "diverted": [
            "diverted", "is_diverted"
        ],
        "air_time": [
            "air_time", "actual_elapsed_time", "flight_time"
        ],
        "taxi_out": [
            "taxi_out", "taxi_out_time"
        ],
        "taxi_in": [
            "taxi_in", "taxi_in_time"
        ],
        "wheels_off": [
            "wheels_off"
        ],
        "wheels_on": [
            "wheels_on"
        ],
    }

    def __init__(self, config=None) -> None:
        """Initialize ingestor with application configuration."""
        self.config = config or get_config()

    def resolve_schema(self, columns: List[str]) -> Tuple[Dict[str, str], List[str]]:
        """Map raw column headers to canonical conceptual names.
        
        Args:
            columns: List of raw dataset column names.
            
        Returns:
            Tuple of (canonical_to_raw_map, unmapped_columns).
        """
        raw_cols_lower = {col.strip().lower(): col for col in columns}
        canonical_map: Dict[str, str] = {}

        for canonical_name, aliases in self.SCHEMA_ALIASES.items():
            for alias in aliases:
                alias_lower = alias.strip().lower()
                if alias_lower in raw_cols_lower:
                    canonical_map[canonical_name] = raw_cols_lower[alias_lower]
                    break

        mapped_raw_names = set(canonical_map.values())
        unmapped_columns = [col for col in columns if col not in mapped_raw_names]
        return canonical_map, unmapped_columns

    def load_dataset(
        self,
        file_path: Union[str, Path],
        nrows: Optional[int] = None,
    ) -> IngestionResult:
        """Load and normalize flight data from CSV or Parquet.

        Args:
            file_path: Path to raw data file (.csv, .csv.gz, or .parquet).
            nrows: Optional integer row limit for fast development sampling.

        Returns:
            IngestionResult dataclass with raw and mapped DataFrames and metadata.
            
        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If file type is unsupported or missing essential required fields.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Raw flight data file not found at: {path}")

        logger.info("Ingesting flight data from %s (nrows=%s)", path, nrows)

        # Read file based on extension
        suffix = path.suffix.lower()
        if suffix in [".parquet", ".pq"]:
            raw_df = pd.read_parquet(path)
            if nrows is not None:
                raw_df = raw_df.head(nrows)
        elif suffix in [".csv", ".gz", ".zip"]:
            raw_df = pd.read_csv(path, nrows=nrows, low_memory=False)
        else:
            raise ValueError(f"Unsupported dataset format: '{suffix}'. Expected CSV or Parquet.")

        total_records = len(raw_df)
        total_columns = len(raw_df.columns)
        logger.info("Loaded %d records and %d columns", total_records, total_columns)

        # Handle Kaggle date components: YEAR, MONTH, DAY -> flight_date
        cols_lower = {c.lower(): c for c in raw_df.columns}
        if "flight_date" not in cols_lower and "fl_date" not in cols_lower:
            if "year" in cols_lower and "month" in cols_lower and "day" in cols_lower:
                logger.info("Synthesizing 'flight_date' from YEAR, MONTH, DAY components")
                raw_df["flight_date"] = pd.to_datetime(
                    raw_df[cols_lower["year"]].astype(str)
                    + "-"
                    + raw_df[cols_lower["month"]].astype(str).str.zfill(2)
                    + "-"
                    + raw_df[cols_lower["day"]].astype(str).str.zfill(2),
                    errors="coerce",
                ).dt.strftime("%Y-%m-%d")

        # Resolve column mappings
        canonical_map, unmapped_columns = self.resolve_schema(list(raw_df.columns))

        # Check required fields
        required_fields = self.config.required_conceptual_fields
        found_required = [req for req in required_fields if req in canonical_map]
        missing_required = [req for req in required_fields if req not in canonical_map]

        if missing_required:
            available_cols_sample = list(raw_df.columns[:15])
            error_msg = (
                f"Dataset ingestion failed! Missing required conceptual fields: {missing_required}. "
                f"Found required fields: {found_required}. "
                f"Available dataset columns ({total_columns} total): {available_cols_sample}... "
                f"Please verify that the dataset matches BTS or Kaggle schema."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Build normalized mapped DataFrame
        # Invert map to: raw_col -> canonical_name
        rename_dict = {raw_col: canonical for canonical, raw_col in canonical_map.items()}
        mapped_df = raw_df.rename(columns=rename_dict).copy()

        # Identify detected post-flight leakage columns
        detected_leakage = [
            col for col in self.config.post_flight_leakage_columns if col in mapped_df.columns
        ]
        logger.info(
            "Schema mapping successfully resolved %d conceptual fields. "
            "Detected %d post-flight leakage columns (will be isolated).",
            len(canonical_map),
            len(detected_leakage),
        )

        return IngestionResult(
            raw_df=raw_df,
            mapped_df=mapped_df,
            schema_mapping=canonical_map,
            unmapped_columns=unmapped_columns,
            required_columns_found=found_required,
            missing_required_columns=missing_required,
            detected_leakage_columns=detected_leakage,
            total_records=total_records,
            total_columns=total_columns,
            file_path=str(path),
        )
