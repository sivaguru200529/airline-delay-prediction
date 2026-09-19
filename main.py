"""CLI Entrypoint for the Airline Delay Prediction & Operations Analytics pipeline.

Usage:
    # Run full Phase 1 + Phase 2A pipeline
    python main.py --all

    # Run only Phase 2A (Feature engineering, historical features, quality reports)
    python main.py --phase2a

    # Run on a specific dataset
    python main.py --data-path data/sample/bts_2024_dev_sample.csv --all

    # Run specific Phase 1 modular steps
    python main.py --ingest
    python main.py --validate
    python main.py --preprocess
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.data.ingestion import FlightDataIngestor
from src.data.validation import DataQualityValidator
from src.data.preprocessing import FlightDataPreprocessor
from src.features.feature_engineering import (
    assert_no_target_leakage,
    build_pre_departure_feature_pipeline,
    split_dataset_chronologically,
)
from src.features.historical_features import HistoricalFeatureCalculator
from src.features.weather_features import WeatherIntegrator
from src.features.feature_reporter import (
    build_feature_dictionary_markdown,
    generate_feature_quality_report,
)

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


def run_phase1(
    target_file: Path,
    config,
    do_ingest: bool = False,
    do_validate: bool = False,
    do_preprocess: bool = False,
    nrows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Execute Phase 1 data pipeline (ingestion, validation, cleaning, anti-leakage purge)."""
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
        return None

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
        return None

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

    return prep_result.pre_departure_df


def run_phase2a(config, pre_departure_df: Optional[pd.DataFrame] = None) -> int:
    """Execute complete Phase 2A feature pipeline.

    Workflow:
        Load Phase 1 pre-departure data
                ↓
        Feature engineering (calendar, timing, cyclical, route)
                ↓
        Historical feature generation (strictly prior, min-history fallback)
                ↓
        Optional weather integration (observed vs forecast verified)
                ↓
        Feature validation & Outlier profiling
                ↓
        Leakage audit (assert_no_target_leakage)
                ↓
        Feature dataset export (parquet & csv)
                ↓
        Feature quality report & dictionary
    """
    print("\n" + "=" * 80)
    print("EXECUTING PHASE 2A: FEATURE ENGINEERING & DATASET PREPARATION")
    print("=" * 80)

    # 1. Load Pre-Departure Dataset if not supplied in memory
    if pre_departure_df is None:
        parquet_file = config.data_processed_dir / "flights_pre_departure.parquet"
        csv_file = config.data_processed_dir / "flights_pre_departure.csv"

        if parquet_file.exists():
            logger.info("Loading pre-departure data from %s", parquet_file)
            input_df = pd.read_parquet(parquet_file)
        elif csv_file.exists():
            logger.info("Loading pre-departure data from %s", csv_file)
            input_df = pd.read_csv(csv_file)
        else:
            logger.warning("No pre-departure dataset found in data/processed/. Running Phase 1 first...")
            sample_target = find_default_dataset(config)
            if not sample_target:
                logger.error("Cannot run Phase 2A: No dataset available.")
                return 1
            input_df = run_phase1(sample_target, config, do_preprocess=True)
            if input_df is None:
                logger.error("Phase 1 preprocessing failed to produce pre-departure dataset.")
                return 1
    else:
        input_df = pre_departure_df.copy()

    logger.info("Phase 2A input dataset loaded: %d records, %d columns", len(input_df), len(input_df.columns))

    print("\n" + "-" * 80)
    print("PHASE 2A - STEP 1: MODULAR FEATURE ENGINEERING")
    print("-" * 80)
    print(f"Input Pre-Departure Records: {len(input_df):,}")

    # 2. Build Date, Departure Timing, Cyclical, and Route Features
    base_features, feat_notes = build_pre_departure_feature_pipeline(input_df, config=config)
    print(f"Base + Cyclical Features Generated: {len(base_features.columns)} columns")
    for k, note in feat_notes.items():
        print(f"  - {k}: {note}")

    # 3. Calculate Strict Anti-Leakage Historical Delay Features
    print("\n" + "-" * 80)
    print("PHASE 2A - STEP 2: TIME-AWARE HISTORICAL DELAY FEATURES")
    print("-" * 80)
    hist_calc = HistoricalFeatureCalculator(config=config)
    hist_features, hist_notes = hist_calc.compute_historical_features(
        base_features,
        target_col="delay_target",
        min_history=config.historical_min_history,
        fallback_strategy=config.historical_fallback_strategy,
    )
    print(f"Historical Features Generated: {len(hist_features.columns)} columns (rates + counts)")
    for col_name, desc in hist_notes.items():
        print(f"  - {col_name}: {desc}")

    # 4. Optional Weather Integration Check
    print("\n" + "-" * 80)
    print("PHASE 2A - STEP 3: WEATHER INTEGRATION FOUNDATION")
    print("-" * 80)
    weather_integrator = WeatherIntegrator(config=config)
    weather_status, weather_file = weather_integrator.check_external_weather_availability()
    print(f"Weather Status: {weather_status}")
    if weather_file:
        print(f"Weather Dataset: {weather_file}")
        # Validate schema and join
        weather_df = pd.read_csv(weather_file) if weather_file.suffix == ".csv" else pd.read_parquet(weather_file)
        w_val = weather_integrator.validate_weather_schema(weather_df)
        if w_val.is_valid:
            joined_weather = weather_integrator.join_nearest_prior_weather(base_features, weather_df)
            weather_features = joined_weather
            print("Successfully merged observed weather records strictly prior to departure.")
        else:
            print(f"External weather validation failed: {w_val.validation_notes}")
            weather_features = None
    else:
        print("Real weather data not supplied in data/external/. Interface verified with zero data fabrication.")
        weather_features = None

    # 5. Assemble Final Feature Matrix
    # Remove temporary target to avoid duplicate column before merging
    target_series = base_features["delay_target"].copy() if "delay_target" in base_features.columns else None
    features_without_target = base_features.drop(columns=["delay_target"], errors="ignore")

    combined_df = pd.concat([features_without_target, hist_features], axis=1)
    if weather_features is not None:
        combined_df = pd.concat([combined_df, weather_features], axis=1)

    # Reattach ground-truth label as the final column
    if target_series is not None:
        combined_df["delay_target"] = target_series

    # 6. Automated Leakage Audit
    print("\n" + "-" * 80)
    print("PHASE 2A - STEP 4: AUTOMATED LEAKAGE AUDIT")
    print("-" * 80)
    leakage_passed = False
    try:
        assert_no_target_leakage(combined_df, config=config)
        leakage_passed = True
        print("LEAKAGE AUDIT: PASS")
        print("  - Zero post-flight outcome columns detected.")
        print("  - Zero derived post-flight keywords detected.")
        print("  - Ground-truth delay_target isolated as training label.")
    except ValueError as e:
        print(f"LEAKAGE AUDIT: FAILED\n{e}")
        return 1

    # 7. Chronological Split Evaluation
    print("\n" + "-" * 80)
    print("PHASE 2A - STEP 5: CHRONOLOGICAL DATASET SPLIT PREPARATION")
    print("-" * 80)
    split_res = split_dataset_chronologically(combined_df, date_col="flight_date")
    split_meta = split_res["split_summary"]
    print(f"Train Set : {split_meta['train_count']:,} ({split_meta['train_pct']}%) [{split_meta['train_date_range'][0]} to {split_meta['train_date_range'][1]}]")
    print(f"Val Set   : {split_meta['val_count']:,} ({split_meta['val_pct']}%) [{split_meta['val_date_range'][0]} to {split_meta['val_date_range'][1]}]")
    print(f"Test Set  : {split_meta['test_count']:,} ({split_meta['test_pct']}%) [{split_meta['test_date_range'][0]} to {split_meta['test_date_range'][1]}]")
    print(f"Note      : {split_meta['sample_size_limitation']}")

    # 8. Export Final Feature Dataset
    config.data_processed_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = config.data_processed_dir / "flights_features.parquet"
    out_csv = config.data_processed_dir / "flights_features.csv"

    combined_df.to_parquet(out_parquet, index=False)
    combined_df.to_csv(out_csv, index=False)

    # 9. Generate Reports & Feature Dictionary
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    dict_md = build_feature_dictionary_markdown()
    dict_file = config.reports_dir / "feature_dictionary.md"
    dict_file.write_text(dict_md, encoding="utf-8")

    all_notes = {**feat_notes, **hist_notes}
    report_dict, report_md = generate_feature_quality_report(
        input_df=input_df,
        features_df=combined_df,
        feature_notes=all_notes,
        leakage_passed=leakage_passed,
        weather_status=weather_status,
        split_summary=split_meta,
        config=config,
    )

    quality_md_file = config.reports_dir / "feature_quality_report.md"
    quality_json_file = config.reports_dir / "feature_quality_report.json"
    quality_md_file.write_text(report_md, encoding="utf-8")
    with open(quality_json_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print("\n" + "-" * 80)
    print("PHASE 2A - STEP 6: ARTIFACT EXPORT & REPORT SUMMARY")
    print("-" * 80)
    print(f"Final ML Feature Dataset : {len(combined_df):,} records, {len(combined_df.columns)} features")
    print(f"  - Parquet Export : {out_parquet}")
    print(f"  - CSV Export     : {out_csv}")
    print(f"Feature Quality Reports:")
    print(f"  - Markdown Report: {quality_md_file}")
    print(f"  - JSON Report    : {quality_json_file}")
    print(f"  - Dictionary     : {dict_file}")

    print("\n" + "=" * 80)
    print("PHASE 2A EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")
    return 0


def run_pipeline(
    data_path: Optional[str] = None,
    do_ingest: bool = False,
    do_validate: bool = False,
    do_preprocess: bool = False,
    do_phase2a: bool = False,
    do_all: bool = False,
    nrows: Optional[int] = None,
) -> int:
    """Execute selected steps or complete Phase 1 and Phase 2A pipeline."""
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

    # When --all is specified, run Phase 1 and Phase 2A end-to-end
    if do_all:
        pre_departure_df = run_phase1(
            target_file=target_file,
            config=config,
            do_ingest=True,
            do_validate=True,
            do_preprocess=True,
            nrows=nrows,
        )
        return run_phase2a(config=config, pre_departure_df=pre_departure_df)

    # When --phase2a is specified alone
    if do_phase2a:
        return run_phase2a(config=config)

    # Otherwise execute requested Phase 1 modular steps
    if do_ingest or do_validate or do_preprocess:
        run_phase1(
            target_file=target_file,
            config=config,
            do_ingest=do_ingest,
            do_validate=do_validate,
            do_preprocess=do_preprocess,
            nrows=nrows,
        )
        return 0

    # Default fallback if no flags provided: run full pipeline
    pre_departure_df = run_phase1(
        target_file=target_file,
        config=config,
        do_ingest=True,
        do_validate=True,
        do_preprocess=True,
        nrows=nrows,
    )
    return run_phase2a(config=config, pre_departure_df=pre_departure_df)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Airline Delay Prediction & Operations Analytics - Pipeline CLI"
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
        help="Run only data ingestion and schema resolution (Phase 1).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data validation and produce quality report (Phase 1).",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run cleaning, target generation, and anti-leakage purge (Phase 1).",
    )
    parser.add_argument(
        "--phase2a",
        action="store_true",
        help="Execute Phase 2A feature engineering, historical features, and quality reports.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute complete end-to-end pipeline (Phase 1 + Phase 2A).",
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
        do_phase2a=args.phase2a,
        do_all=args.all,
        nrows=args.nrows,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
