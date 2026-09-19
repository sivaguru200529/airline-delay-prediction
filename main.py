"""CLI Entrypoint for the Airline Delay Prediction & Operations Analytics pipeline.

Usage:
    python main.py --all
    python main.py --data-path data/sample/bts_2024_dev_sample.csv --all
    python main.py --ingest
    python main.py --validate
    python main.py --preprocess
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.data.ingestion import FlightDataIngestor
from src.data.validation import DataQualityValidator
from src.data.preprocessing import FlightDataPreprocessor

logger = get_logger("airline_delay_cli")


def find_default_dataset(config) -> Optional[Path]:
    """Find dataset in data/raw/ or fall back to development sample."""
    raw_dir = config.data_raw_dir

    # Search for files in data/raw/
    raw_candidates = [
        f for f in raw_dir.glob("*")
        if f.suffix.lower() in [".csv", ".parquet", ".pq", ".gz"]
        and not f.name.startswith(".")
    ]
    if raw_candidates:
        return raw_candidates[0]

    # Fallback to data/sample/bts_2024_dev_sample.csv
    sample_file = config.project_root / "data" / "sample" / "bts_2024_dev_sample.csv"
    if sample_file.exists():
        return sample_file

    return None


def run_pipeline(
    data_path: Optional[str] = None,
    do_ingest: bool = False,
    do_validate: bool = False,
    do_preprocess: bool = False,
    do_all: bool = False,
    nrows: Optional[int] = None,
) -> int:
    """Execute selected steps or complete Phase 1 data pipeline."""
    config = get_config()

    # Determine input dataset
    if data_path:
        target_file = Path(data_path).resolve()
    else:
        target_file = find_default_dataset(config)

    if not target_file or not target_file.exists():
        logger.error(
            "No flight dataset found! Please place a historical CSV/Parquet dataset in "
            "'%s' (refer to data/raw/README.md) or specify with --data-path.",
            config.data_raw_dir,
        )
        return 1

    is_sample = "sample" in str(target_file).lower()
    if is_sample:
        print("\n" + "=" * 80)
        print("NOTICE: Running pipeline on verified DEVELOPMENT SAMPLE:")
        print(f"  File: {target_file}")
        print("  (For production models, place full BTS dataset into data/raw/ as per data/raw/README.md)")
        print("=" * 80 + "\n")
    else:
        logger.info("Using raw flight dataset: %s", target_file)

    # If no specific action specified or --all, default to complete pipeline
    if do_all or (not do_ingest and not do_validate and not do_preprocess):
        do_ingest = True
        do_validate = True
        do_preprocess = True

    # 1. DATA INGESTION
    ingestor = FlightDataIngestor(config=config)
    ingestion_result = ingestor.load_dataset(target_file, nrows=nrows)

    print("\n" + "-" * 80)
    print("STEP 1: DATA INGESTION & SCHEMA RESOLUTION")
    print("-" * 80)
    print(f"Total Raw Records Ingested : {ingestion_result.total_records:,}")
    print(f"Total Raw Columns Detected  : {ingestion_result.total_columns}")
    print(f"Mapped Conceptual Fields    : {len(ingestion_result.schema_mapping)}")
    print(f"Detected Leakage Columns    : {ingestion_result.detected_leakage_columns}")
    print(f"Unmapped Columns            : {len(ingestion_result.unmapped_columns)}")

    if not do_validate and not do_preprocess:
        return 0

    # 2. DATA QUALITY VALIDATION
    validator = DataQualityValidator(config=config)
    summary, annotated_df = validator.validate(
        df=ingestion_result.mapped_df,
        schema_mapping=ingestion_result.schema_mapping,
        detected_leakage=ingestion_result.detected_leakage_columns,
        unmapped=ingestion_result.unmapped_columns,
    )
    md_path, json_path = validator.save_reports(summary)

    print("\n" + "-" * 80)
    print("STEP 2: DATA QUALITY & LEAKAGE AUDIT SUMMARY")
    print("-" * 80)
    print(f"Total Records Evaluated     : {summary.total_records:,}")
    print(f"Duplicate Records           : {summary.duplicate_records:,}")
    print(f"Cancelled Flights           : {summary.cancelled_flights:,}")
    print(f"Diverted Flights            : {summary.diverted_flights:,}")
    print(f"Invalid Records             : {summary.total_invalid_records:,}")
    print(f"Valid Completed Flights     : {summary.total_valid_records:,}")
    print(f"Delayed Flights (>=15 min)  : {summary.delayed_flights:,} ({summary.delay_rate_percent:.2f}%)")
    print(f"On-Time Flights (<15 min)   : {summary.ontime_flights:,} ({100.0 - summary.delay_rate_percent:.2f}%)")
    print(f"Reports Written To          : {md_path} and {json_path}")

    if not do_preprocess:
        return 0

    # 3. LEAKAGE-FREE PREPROCESSING & DATASET EXPORT
    preprocessor = FlightDataPreprocessor(config=config)
    prep_result = preprocessor.process(annotated_df=annotated_df, save_output=True)

    print("\n" + "-" * 80)
    print("STEP 3: PREPROCESSING & ANTI-LEAKAGE PURGE")
    print("-" * 80)
    print(f"Clean Operational Records   : {len(prep_result.cleaned_operational_df):,}")
    print(f"Pre-Departure Features      : {list(prep_result.pre_departure_df.columns)}")
    print(f"Cancelled Flights Segregated: {len(prep_result.cancelled_df):,}")
    print(f"Diverted Flights Segregated : {len(prep_result.diverted_df):,}")
    print("Exported Files:")
    for key, path in prep_result.output_files.items():
        print(f"  - {key}: {path}")

    print("\n" + "=" * 80)
    print("PHASE 1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")
    return 0


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Airline Delay Prediction & Operations Analytics - Phase 1 CLI"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw flight CSV/Parquet dataset.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run only data ingestion and schema resolution.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data validation and produce quality report.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run cleaning, target generation, and anti-leakage purge.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute complete Phase 1 pipeline.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for fast sampling.",
    )

    args = parser.parse_args()
    exit_code = run_pipeline(
        data_path=args.data_path,
        do_ingest=args.ingest,
        do_validate=args.validate,
        do_preprocess=args.preprocess,
        do_all=args.all,
        nrows=args.nrows,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
