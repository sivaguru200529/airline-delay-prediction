import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def create_phase3_notebook():
    """Construct nbformat 4 JSON structure for notebooks/06_phase3_feature_engineering.ipynb."""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 3 — Advanced Feature Engineering & Operational Analytics\n",
                    "\n",
                    "**Project:** Airline Delay Prediction & Operations Analytics  \n",
                    "**Scope:** Modular, leakage-safe pre-departure feature engineering layer expanding upon Phase 2A. Covers calendar date features, operational time blocks, route distance analysis and categorization, strictly prior route frequency, airport volume and delay rates, multi-granular historical delay features with prior global fallback, and weather interface validation.  \n",
                    "**Invariant:** Strictly satisfies prediction time reference $T_{dep}$ (scheduled departure).  \n",
                    "\n",
                    "> **DEVELOPMENT DATASET LIMITATION NOTICE:**  \n",
                    "> This notebook runs on the development dataset sample (481 completed flights from January 1–10, 2024). Results verify pipeline execution, anti-leakage invariants, and feature distributions. Production deployment requires the full public BTS dataset."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 1: Load Phase 2A Dataset\n",
                    "\n",
                    "Import required modules, initialize project configuration, and load the Phase 2A feature dataset (`flights_features.parquet`) and pre-departure baseline (`flights_pre_departure.parquet`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Set project root\n",
                    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "if str(project_root) not in sys.path:\n",
                    "    sys.path.insert(0, str(project_root))\n",
                    "\n",
                    "from src.utils.config import get_config\n",
                    "from src.features.advanced_features import (\n",
                    "    build_phase3_date_features,\n",
                    "    build_phase3_time_features,\n",
                    "    build_phase3_route_and_airport_features,\n",
                    "    build_phase3_feature_pipeline,\n",
                    ")\n",
                    "from src.features.historical_features import HistoricalFeatureCalculator\n",
                    "from src.features.weather_features import WeatherIntegrator\n",
                    "from src.features.feature_engineering import assert_no_target_leakage\n",
                    "from src.features.phase3_reporter import generate_phase3_reports\n",
                    "\n",
                    "config = get_config()\n",
                    "sns.set_theme(style='whitegrid', palette='muted')\n",
                    "plt.rcParams['figure.figsize'] = (10, 5)\n",
                    "plt.rcParams['font.size'] = 11\n",
                    "\n",
                    "# Load pre-departure baseline data\n",
                    "input_path = config.data_processed_dir / 'flights_pre_departure.parquet'\n",
                    "df_raw = pd.read_parquet(input_path)\n",
                    "print(f'Loaded Pre-Departure Data: {df_raw.shape[0]} records, {df_raw.shape[1]} columns')\n",
                    "df_raw.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 2: Inspect Existing Features\n",
                    "\n",
                    "Inspect the existing Phase 2A feature catalog, data types, null rates, and distribution of the target variable (`delay_target`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "p2a_path = config.data_processed_dir / 'flights_features.parquet'\n",
                    "if p2a_path.exists():\n",
                    "    df_p2a = pd.read_parquet(p2a_path)\n",
                    "    print(f'Phase 2A Feature Matrix: {df_p2a.shape[0]} records, {df_p2a.shape[1]} columns')\n",
                    "    print('\\nPhase 2A Columns:')\n",
                    "    print(list(df_p2a.columns))\n",
                    "else:\n",
                    "    print('Phase 2A parquet file not found; generating from raw input.')\n",
                    "    df_p2a = df_raw.copy()\n",
                    "\n",
                    "# Delay Target Class Balance\n",
                    "target_counts = df_raw['delay_target'].value_counts(dropna=False)\n",
                    "target_pct = df_raw['delay_target'].value_counts(normalize=True) * 100\n",
                    "target_summary = pd.DataFrame({'Count': target_counts, 'Percentage (%)': target_pct.round(2)})\n",
                    "print('\\nTarget Class Balance (delay_target):')\n",
                    "display(target_summary)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 3: Generate Phase 3 Features\n",
                    "\n",
                    "Execute the Phase 3 feature pipeline across date, time, route, airport, and historical delay dimensions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Base Phase 3 Features (Date, Time, Route, Airport Volume)\n",
                    "base_p3_df, base_notes = build_phase3_feature_pipeline(df_raw, config=config)\n",
                    "print(f'Phase 3 Base Features Generated: {base_p3_df.shape[1]} columns')\n",
                    "\n",
                    "# 2. Multi-Granular Historical Delay Features\n",
                    "hist_calc = HistoricalFeatureCalculator(config=config)\n",
                    "hist_p3_df, hist_notes, coverage_audit = hist_calc.compute_phase3_historical_features(\n",
                    "    base_p3_df,\n",
                    "    target_col='delay_target',\n",
                    "    min_history=config.historical_min_history_p3,\n",
                    ")\n",
                    "print(f'Phase 3 Historical Delay Features: {hist_p3_df.shape[1]} columns')\n",
                    "\n",
                    "# 3. Combine into Unified Phase 3 Feature Dataset\n",
                    "features_without_target = base_p3_df.drop(columns=['delay_target'], errors='ignore')\n",
                    "overlapping = [c for c in hist_p3_df.columns if c in features_without_target.columns]\n",
                    "if overlapping:\n",
                    "    features_without_target = features_without_target.drop(columns=overlapping)\n",
                    "\n",
                    "df_p3 = pd.concat([features_without_target, hist_p3_df], axis=1)\n",
                    "df_p3['delay_target'] = df_raw['delay_target'].values\n",
                    "print(f'\\nTotal Unified Phase 3 Matrix: {df_p3.shape[0]} records, {df_p3.shape[1]} columns')\n",
                    "df_p3.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 4: Analyze Date & Time Features\n",
                    "\n",
                    "Inspect calendar features (`is_month_start`, `is_month_end`, `quarter`, `season`) and operational time buckets (`dep_time_of_day`, `arr_time_of_day`, `dep_time_bucket`, `arr_time_bucket`, cyclical sin/cos)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "# Time of Day Distribution\n",
                    "tod_order = ['night', 'morning', 'afternoon', 'evening']\n",
                    "tod_counts = df_p3['dep_time_of_day'].value_counts().reindex(tod_order).dropna()\n",
                    "sns.barplot(x=tod_counts.index, y=tod_counts.values, ax=axes[0], palette='Blues_d')\n",
                    "axes[0].set_title('Departure Time of Day Distribution')\n",
                    "axes[0].set_xlabel('Operational Block')\n",
                    "axes[0].set_ylabel('Flight Count')\n",
                    "\n",
                    "# 4-Hour Buckets Distribution\n",
                    "bucket_order = ['00-04', '04-08', '08-12', '12-16', '16-20', '20-24']\n",
                    "bucket_counts = df_p3['dep_time_bucket'].value_counts().reindex(bucket_order).fillna(0)\n",
                    "sns.barplot(x=bucket_counts.index, y=bucket_counts.values, ax=axes[1], palette='crest')\n",
                    "axes[1].set_title('4-Hour Departure Time Buckets')\n",
                    "axes[1].set_xlabel('Time Bucket')\n",
                    "axes[1].set_ylabel('Flight Count')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "# Cyclical Encoding Continuity Verification\n",
                    "cyclical_check = (\n",
                    "    df_p3['arr_hour_sin']**2 + df_p3['arr_hour_cos']**2\n",
                    ").round(5)\n",
                    "print(f'Cyclical Identity (sin^2 + cos^2 == 1.0) Min: {cyclical_check.min()}, Max: {cyclical_check.max()}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 5: Analyze Route Features & Distance Categories\n",
                    "\n",
                    "Inspect flight distance distribution and verify FAA-aligned distance categorizations (`short_haul`, `medium_haul`, `long_haul`), along with strictly prior route frequency (`prior_route_frequency`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [],
                "source": [
                    "dist_col = 'distance' if 'distance' in df_p3.columns else 'route_distance'\n",
                    "print('Distance Distribution Summary:')\n",
                    "print(df_p3[dist_col].describe())\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "# Distance Histogram with KDE\n",
                    "sns.histplot(df_p3[dist_col], kde=True, ax=axes[0], color='teal', bins=20)\n",
                    "axes[0].set_title('Flight Distance Distribution (Miles)')\n",
                    "axes[0].set_xlabel('Distance (miles)')\n",
                    "axes[0].axvline(500, color='orange', linestyle='--', label='Short/Medium Cutoff (500 mi)')\n",
                    "axes[0].axvline(1500, color='red', linestyle='--', label='Medium/Long Cutoff (1500 mi)')\n",
                    "axes[0].legend()\n",
                    "\n",
                    "# Haul Category Breakdown\n",
                    "cat_counts = df_p3['distance_category'].value_counts()\n",
                    "axes[1].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', colors=['#4a90e2', '#50e3c2', '#f5a623'])\n",
                    "axes[1].set_title('Haul Category Breakdown')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "# Prior Route Frequency Distribution\n",
                    "print('\\nPrior Route Frequency Summary (strictly prior t < T):')\n",
                    "print(df_p3['prior_route_frequency'].value_counts().head(10))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 6: Analyze Airport Operational Features\n",
                    "\n",
                    "Inspect prior airport operational volume and delay rates (`prior_origin_flight_volume`, `prior_dest_flight_volume`, `prior_origin_delay_rate`, `prior_dest_delay_rate`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "metadata": {},
                "outputs": [],
                "source": [
                    "airport_cols = [\n",
                    "    'prior_origin_flight_volume', 'prior_origin_delay_rate',\n",
                    "    'prior_dest_flight_volume', 'prior_dest_delay_rate'\n",
                    "]\n",
                    "print('Airport Operational Features Summary:')\n",
                    "display(df_p3[airport_cols].describe())\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "sns.boxplot(data=df_p3, x='delay_target', y='prior_origin_flight_volume', ax=axes[0], palette='Set2')\n",
                    "axes[0].set_title('Origin Prior Flight Volume by Delay Outcome')\n",
                    "axes[0].set_xlabel('Delay Target (0=On-time, 1=Delayed)')\n",
                    "axes[0].set_ylabel('Prior Origin Volume')\n",
                    "\n",
                    "sns.boxplot(data=df_p3, x='delay_target', y='prior_origin_delay_rate', ax=axes[1], palette='Set2')\n",
                    "axes[1].set_title('Origin Prior Delay Rate by Delay Outcome')\n",
                    "axes[1].set_xlabel('Delay Target (0=On-time, 1=Delayed)')\n",
                    "axes[1].set_ylabel('Prior Origin Delay Rate')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 7: Analyze Historical Delay Features & Prior Global Fallback\n",
                    "\n",
                    "Analyze multi-granular historical delay features across airline, origin, destination, route, and hour interactions. Evaluate coverage across minimum history threshold (`min_history=3`) and strictly prior global rate fallback."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Historical Delay Features Coverage Audit:')\n",
                    "cov_data = []\n",
                    "for dim, info in coverage_audit.items():\n",
                    "    cov_data.append({\n",
                    "        'Dimension': dim.replace('prior_', '').replace('_', ' ').title(),\n",
                    "        'No History (0)': f\"{info['no_history_count']} ({info['no_history_pct']}%)\",\n",
                    "        'Insufficient (1-2)': f\"{info['insufficient_history_count']} ({info['insufficient_history_pct']}%)\",\n",
                    "        'Sufficient (>=3)': f\"{info['sufficient_history_count']} ({info['sufficient_history_pct']}%)\",\n",
                    "    })\n",
                    "display(pd.DataFrame(cov_data))\n",
                    "\n",
                    "# Correlation of prior delay rates with target\n",
                    "rate_cols = [c for c in df_p3.columns if c.endswith('_delay_rate')]\n",
                    "if rate_cols:\n",
                    "    corrs = df_p3[rate_cols + ['delay_target']].corr()['delay_target'].drop('delay_target').sort_values(ascending=False)\n",
                    "    plt.figure(figsize=(10, 4))\n",
                    "    sns.barplot(x=corrs.values, y=corrs.index, palette='viridis')\n",
                    "    plt.title('Correlation with delay_target across Prior Delay Rates')\n",
                    "    plt.xlabel('Pearson Correlation')\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 8: Analyze Weather Availability & Severe Weather Indicators\n",
                    "\n",
                    "Verify the zero-fabrication weather contract: check for presence of external NOAA METAR/TAF weather observations, verify temporal contract ($T_{obs} \\le T_{dep}$), and demonstrate severe weather indicator derivation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 8,
                "metadata": {},
                "outputs": [],
                "source": [
                    "weather_integrator = WeatherIntegrator(config=config)\n",
                    "status, weather_file = weather_integrator.check_external_weather_availability()\n",
                    "print(f'External Weather Status: {status}')\n",
                    "if weather_file:\n",
                    "    print(f'External Weather File Found: {weather_file}')\n",
                    "else:\n",
                    "    print('Real external weather data not provided in data/external/.')\n",
                    "    print('Contract Verification: zero data fabrication policy strictly maintained.')\n",
                    "\n",
                    "# Demonstrate severe weather indicator schema\n",
                    "demo_weather = pd.DataFrame({\n",
                    "    'origin_precipitation_inches': [0.15, 0.0, 0.0],\n",
                    "    'origin_snow_inches': [0.0, 3.0, 0.0],\n",
                    "    'origin_visibility_miles': [10.0, 4.0, 1.5],\n",
                    "    'origin_weather_condition': ['rain', 'snow', 'fog'],\n",
                    "    'origin_wind_speed_knots': [12.0, 18.0, 42.0],\n",
                    "})\n",
                    "demo_derived = weather_integrator.derive_severe_weather_indicators(demo_weather)\n",
                    "print('\\nDemonstration of Weather Indicator Derivation Schema:')\n",
                    "display(demo_derived)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 9: Missing-Value Analysis\n",
                    "\n",
                    "Audit missingness across all features to guarantee zero unhandled nulls before feeding to downstream machine learning models."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 9,
                "metadata": {},
                "outputs": [],
                "source": [
                    "missing_counts = df_p3.isna().sum()\n",
                    "missing_pcts = (df_p3.isna().mean() * 100).round(2)\n",
                    "missing_df = pd.DataFrame({\n",
                    "    'Missing_Count': missing_counts,\n",
                    "    'Missing_Pct': missing_pcts,\n",
                    "})\n",
                    "missing_features = missing_df[missing_df['Missing_Count'] > 0]\n",
                    "\n",
                    "if len(missing_features) == 0:\n",
                    "    print('MISSING VALUE AUDIT: PASS - Exactly 0 unhandled null values across all features!')\n",
                    "else:\n",
                    "    print(f'Features with missing values ({len(missing_features)}):')\n",
                    "    display(missing_features)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 10: Feature Distributions\n",
                    "\n",
                    "Examine distributions of key continuous and cyclical feature transformations."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 10,
                "metadata": {},
                "outputs": [],
                "source": [
                    "num_cols = ['carrier_prior_flight_count', 'origin_prior_flight_count', 'prior_route_frequency']\n",
                    "existing_num = [c for c in num_cols if c in df_p3.columns]\n",
                    "\n",
                    "fig, axes = plt.subplots(1, len(existing_num), figsize=(5 * len(existing_num), 4))\n",
                    "for i, col in enumerate(existing_num):\n",
                    "    sns.histplot(df_p3[col], ax=axes[i], color='navy', kde=False, discrete=True)\n",
                    "    axes[i].set_title(f'{col} Distribution')\n",
                    "    axes[i].set_xlabel('Count')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 11: Direct & Derived Leakage Audit\n",
                    "\n",
                    "Run the centralized anti-leakage audit (`assert_no_target_leakage`). Verify that zero post-flight operational columns, forbidden keywords, or derived temporal aggregations exist in the feature matrix."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 11,
                "metadata": {},
                "outputs": [],
                "source": [
                    "leakage_passed = False\n",
                    "try:\n",
                    "    assert_no_target_leakage(df_p3, config=config)\n",
                    "    leakage_passed = True\n",
                    "    print('LEAKAGE AUDIT: PASS')\n",
                    "    print('  - Zero post-flight outcome columns detected.')\n",
                    "    print('  - Zero derived post-flight keywords detected.')\n",
                    "    print('  - Historical/prior features verified strictly prior to departure (t < T).')\n",
                    "    print('  - Ground-truth delay_target isolated as training label.')\n",
                    "except ValueError as e:\n",
                    "    print(f'LEAKAGE AUDIT: FAILED\\n{e}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 12: Final Feature Summary & Comparison with Phase 2A\n",
                    "\n",
                    "Generate the comprehensive Phase 3 feature quality reports (`reports/phase3_feature_report.md` and `.json`), summarize feature retention, enhancement, and additions, and document next steps for Phase 3B."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 12,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generate reports\n",
                    "p2a_cols = list(df_p2a.columns) if 'df_p2a' in locals() else []\n",
                    "report_dict, report_md = generate_phase3_reports(\n",
                    "    p3_df=df_p3,\n",
                    "    phase2a_cols=p2a_cols,\n",
                    "    dist_meta=base_notes.get('distance_metadata', {}),\n",
                    "    coverage_audit=coverage_audit,\n",
                    "    leakage_passed=leakage_passed,\n",
                    "    weather_status=status,\n",
                    "    config=config,\n",
                    ")\n",
                    "\n",
                    "print(f'Phase 3 Total Features: {report_dict[\"total_features\"]}')\n",
                    "print(f'  - Retained from Phase 2A : {report_dict[\"feature_counts\"][\"retained_count\"]}')\n",
                    "print(f'  - Enhanced in Phase 3    : {report_dict[\"feature_counts\"][\"enhanced_count\"]}')\n",
                    "print(f'  - New in Phase 3         : {report_dict[\"feature_counts\"][\"new_count\"]}')\n",
                    "print('\\nReports written:')\n",
                    "print(f'  - {config.reports_dir / \"phase3_feature_report.md\"}')\n",
                    "print(f'  - {config.reports_dir / \"phase3_feature_report.json\"}')\n",
                    "\n",
                    "print('\\nNext Phase: Phase 3B \u2014 Model Retraining & Phase 2B vs Phase 3 Evaluation.')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.13.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_file = project_root / "notebooks" / "06_phase3_feature_engineering.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully generated {out_file}")


if __name__ == "__main__":
    create_phase3_notebook()
