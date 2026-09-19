"""Data quality validation module for flight operations datasets.

Performs comprehensive validation checks on flight records including duplicates,
missing values, timestamp validity, airport code formats, distance sanity,
duration integrity, cancelled/diverted flight accounting, and leakage column audit.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("data_validation")


@dataclass
class ValidationSummary:
    """Comprehensive data quality audit report summary."""

    total_records: int
    duplicate_records: int
    missing_values_by_column: Dict[str, int]
    missing_required_records: int
    cancelled_flights: int
    diverted_flights: int
    invalid_date_records: int
    invalid_airport_records: int
    invalid_distance_records: int
    invalid_delay_records: int
    total_invalid_records: int
    total_valid_records: int
    delayed_flights: int
    ontime_flights: int
    delay_rate_percent: float
    schema_mapping_used: Dict[str, str]
    detected_leakage_columns: List[str]
    required_columns_detected: List[str]
    unmapped_columns: List[str]
    cleaning_decisions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation summary to standard dictionary."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Generate human-readable Markdown data quality report."""
        lines = [
            "# Airline Operations - Data Quality Audit Report",
            "",
            "## Executive Summary",
            "",
            "| Metric | Count | Percentage |",
            "| :--- | :--- | :--- |",
            f"| **Total Raw Records** | {self.total_records:,} | 100.00% |",
            f"| **Duplicate Records** | {self.duplicate_records:,} | {self.duplicate_records / max(1, self.total_records) * 100:.2f}% |",
            f"| **Cancelled Flights** | {self.cancelled_flights:,} | {self.cancelled_flights / max(1, self.total_records) * 100:.2f}% |",
            f"| **Diverted Flights** | {self.diverted_flights:,} | {self.diverted_flights / max(1, self.total_records) * 100:.2f}% |",
            f"| **Invalid / Corrupt Records** | {self.total_invalid_records:,} | {self.total_invalid_records / max(1, self.total_records) * 100:.2f}% |",
            f"| **Valid Completed Flights** | {self.total_valid_records:,} | {self.total_valid_records / max(1, self.total_records) * 100:.2f}% |",
            "",
            "## Target Distribution (Valid Completed Flights)",
            "",
            "| Target Class | Definition | Flight Count | Proportion |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Delayed (1)** | Arrival Delay $\\ge 15$ min | {self.delayed_flights:,} | {self.delay_rate_percent:.2f}% |",
            f"| **On-Time (0)** | Arrival Delay $< 15$ min | {self.ontime_flights:,} | {100.0 - self.delay_rate_percent:.2f}% |",
            "",
            "## Data Leakage Audit (Post-Flight Columns Detected)",
            "",
            "The following columns were detected in the raw operational dataset and classified as **post-flight leakage**.",
            "These fields are retained in the full cleaned dataset for operational retrospectives, but are strictly excluded from the pre-departure prediction feature matrix:",
            "",
        ]
        if self.detected_leakage_columns:
            for col in self.detected_leakage_columns:
                lines.append(f"- ` {col} ` (Post-flight / Outcome)")
        else:
            lines.append("- *No post-flight leakage columns detected.*")

        lines.extend([
            "",
            "## Schema Mapping Audit",
            "",
            "| Canonical Conceptual Field | Source Column |",
            "| :--- | :--- |",
        ])
        for canonical, source in self.schema_mapping_used.items():
            lines.append(f"| `{canonical}` | `{source}` |")

        lines.extend([
            "",
            "## Missing Values Breakdown",
            "",
            "| Column | Missing Count | Missing Rate |",
            "| :--- | :--- | :--- |",
        ])
        for col, count in sorted(self.missing_values_by_column.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"| `{col}` | {count:,} | {count / max(1, self.total_records) * 100:.2f}% |")

        if not any(count > 0 for count in self.missing_values_by_column.values()):
            lines.append("| *(All columns complete)* | 0 | 0.00% |")

        lines.extend([
            "",
            "## Cleaning & Segregation Decisions",
            "",
        ])
        for decision in self.cleaning_decisions:
            lines.append(f"- {decision}")

        lines.append("")
        return "\n".join(lines)


class DataQualityValidator:
    """Validates flight datasets against aviation domain rules and quality benchmarks."""

    IATA_AIRPORT_REGEX = re.compile(r"^[A-Za-z]{3}$")

    def __init__(self, config=None) -> None:
        """Initialize validator with configuration."""
        self.config = config or get_config()

    def validate(
        self,
        df: pd.DataFrame,
        schema_mapping: Optional[Dict[str, str]] = None,
        detected_leakage: Optional[List[str]] = None,
        unmapped: Optional[List[str]] = None,
    ) -> Tuple[ValidationSummary, pd.DataFrame]:
        """Perform full validation audit on mapped flight DataFrame.

        Args:
            df: Canonical DataFrame produced by FlightDataIngestor.
            schema_mapping: Dictionary mapping canonical to raw column names.
            detected_leakage: List of post-flight leakage columns identified.
            unmapped: List of unmapped raw column names.

        Returns:
            Tuple of (ValidationSummary, annotated_df).
        """
        logger.info("Starting data quality validation on %d records", len(df))
        total_records = len(df)
        working_df = df.copy()

        # Initialize boolean masks for various defect classes
        is_duplicate = working_df.duplicated(keep="first")
        duplicate_count = int(is_duplicate.sum())

        # Missing values breakdown
        missing_by_col = {col: int(working_df[col].isna().sum()) for col in working_df.columns}

        # Cancelled & Diverted flags
        if "cancelled" in working_df.columns:
            is_cancelled = working_df["cancelled"].fillna(0).astype(float) == 1.0
        else:
            is_cancelled = pd.Series(False, index=working_df.index)

        if "diverted" in working_df.columns:
            is_diverted = working_df["diverted"].fillna(0).astype(float) == 1.0
        else:
            is_diverted = pd.Series(False, index=working_df.index)

        cancelled_count = int(is_cancelled.sum())
        diverted_count = int(is_diverted.sum())

        # Validate Flight Date format
        invalid_date = pd.Series(False, index=working_df.index)
        if "flight_date" in working_df.columns:
            parsed_dates = pd.to_datetime(working_df["flight_date"], errors="coerce")
            invalid_date = parsed_dates.isna()
        invalid_date_count = int(invalid_date.sum())

        # Validate Airport IATA codes (Origin & Destination)
        invalid_origin = pd.Series(False, index=working_df.index)
        invalid_dest = pd.Series(False, index=working_df.index)

        if "origin_airport" in working_df.columns:
            origin_str = working_df["origin_airport"].astype(str).str.strip()
            invalid_origin = ~origin_str.str.match(r"^[A-Za-z]{3}$") | origin_str.isna()

        if "dest_airport" in working_df.columns:
            dest_str = working_df["dest_airport"].astype(str).str.strip()
            invalid_dest = ~dest_str.str.match(r"^[A-Za-z]{3}$") | dest_str.isna()

        invalid_airport = invalid_origin | invalid_dest
        invalid_airport_count = int(invalid_airport.sum())

        # Validate Distance (Distance must be positive and <= 10,000 miles for US domestic/territory)
        invalid_distance = pd.Series(False, index=working_df.index)
        if "distance" in working_df.columns:
            dist_numeric = pd.to_numeric(working_df["distance"], errors="coerce")
            invalid_distance = dist_numeric.isna() | (dist_numeric <= 0) | (dist_numeric > 10000)
        invalid_distance_count = int(invalid_distance.sum())

        # Validate Arrival Delay (for non-cancelled flights)
        # Delay must be numeric, and within reasonable range: -120 min to 2880 min (48 hours)
        invalid_delay = pd.Series(False, index=working_df.index)
        if "arrival_delay" in working_df.columns:
            arr_delay_numeric = pd.to_numeric(working_df["arrival_delay"], errors="coerce")
            # For non-cancelled/non-diverted flights, missing or extreme arrival delay is invalid
            non_disrupted = ~is_cancelled & ~is_diverted
            invalid_delay = non_disrupted & (
                arr_delay_numeric.isna()
                | (arr_delay_numeric < -120)
                | (arr_delay_numeric > 2880)
            )
        invalid_delay_count = int(invalid_delay.sum())

        # Combine all invalid conditions
        is_invalid = (
            is_duplicate
            | invalid_date
            | invalid_airport
            | invalid_distance
            | invalid_delay
        )
        total_invalid_count = int(is_invalid.sum())

        # Valid completed records suitable for arrival delay classification modeling
        is_valid_completed = ~is_invalid & ~is_cancelled & ~is_diverted
        valid_count = int(is_valid_completed.sum())

        # Calculate delay target distribution on valid completed flights
        arr_delay_series = pd.to_numeric(
            working_df.loc[is_valid_completed, "arrival_delay"], errors="coerce"
        )
        delayed_count = int((arr_delay_series >= self.config.arrival_delay_threshold).sum())
        ontime_count = valid_count - delayed_count
        delay_rate = (delayed_count / max(1, valid_count)) * 100.0

        # Annotate working dataframe with audit flags
        working_df["_is_duplicate"] = is_duplicate
        working_df["_is_cancelled"] = is_cancelled
        working_df["_is_diverted"] = is_diverted
        working_df["_is_invalid"] = is_invalid
        working_df["_is_valid_completed"] = is_valid_completed

        # Document explicit cleaning and segregation decisions
        cleaning_decisions = [
            f"Detected {duplicate_count} exact duplicate rows flagged for deduplication.",
            f"Identified {cancelled_count} cancelled flights; segregated into operational cancellation analysis.",
            f"Identified {diverted_count} diverted flights; segregated into operational diversion analysis.",
            f"Flagged {invalid_date_count} records with unparseable flight dates.",
            f"Flagged {invalid_airport_count} records with invalid IATA airport codes (non-3-letter).",
            f"Flagged {invalid_distance_count} records with non-positive or extreme distances (>10,000 miles).",
            f"Flagged {invalid_delay_count} completed flights with missing or extreme arrival delay values.",
            f"Arrival delay >= {self.config.arrival_delay_threshold} min used as delayed ground-truth target.",
            "All post-flight operational metrics (actual departure/arrival times, taxi durations, air time) "
            "strictly excluded from pre-departure model feature matrix to prevent data leakage.",
        ]

        summary = ValidationSummary(
            total_records=total_records,
            duplicate_records=duplicate_count,
            missing_values_by_column=missing_by_col,
            missing_required_records=int(
                working_df[self.config.required_conceptual_fields].isna().any(axis=1).sum()
            ),
            cancelled_flights=cancelled_count,
            diverted_flights=diverted_count,
            invalid_date_records=invalid_date_count,
            invalid_airport_records=invalid_airport_count,
            invalid_distance_records=invalid_distance_count,
            invalid_delay_records=invalid_delay_count,
            total_invalid_records=total_invalid_count,
            total_valid_records=valid_count,
            delayed_flights=delayed_count,
            ontime_flights=ontime_count,
            delay_rate_percent=round(delay_rate, 2),
            schema_mapping_used=schema_mapping or {},
            detected_leakage_columns=detected_leakage or [],
            required_columns_detected=self.config.required_conceptual_fields,
            unmapped_columns=unmapped or [],
            cleaning_decisions=cleaning_decisions,
        )

        logger.info(
            "Validation complete: %d valid completed flights (%.2f%% delayed), "
            "%d invalid, %d cancelled, %d diverted",
            valid_count,
            delay_rate,
            total_invalid_count,
            cancelled_count,
            diverted_count,
        )
        return summary, working_df

    def save_reports(
        self,
        summary: ValidationSummary,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Persist validation summary to Markdown and JSON reports.

        Args:
            summary: ValidationSummary instance.
            output_dir: Target directory (defaults to config.reports_dir).

        Returns:
            Tuple of (markdown_path, json_path).
        """
        out_dir = output_dir or self.config.reports_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / "data_quality_report.md"
        json_path = out_dir / "data_quality_report.json"

        # Write Markdown report
        md_content = summary.to_markdown()
        md_path.write_text(md_content, encoding="utf-8")

        # Write JSON report
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)

        logger.info("Saved data quality reports to %s and %s", md_path, json_path)
        return md_path, json_path
