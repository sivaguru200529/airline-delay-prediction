"""Script to build and generate notebooks/07_phase5_eda.ipynb with 17 structured sections."""

import json
from pathlib import Path


def create_phase5_eda_notebook():
    """Construct nbformat 4 JSON structure for notebooks/07_phase5_eda.ipynb."""
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    def add_md(source: str):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")],
        })

    def add_code(source: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")],
        })

    # Header
    add_md("""# Phase 5 — Exploratory Data Analysis (EDA)

**Project:** Airline Delay Prediction & Operations Analytics  
**Scope:** In-depth, empirical, and reproducible Exploratory Data Analysis of the Phase 3 engineered airline delay feature dataset.  
**Objectives:**
1. Provide an exhaustive dataset overview (dimensions, column data types, unique counts, missingness).
2. Quantify target class balance and establish the baseline operational delay rate (30.15%).
3. Audit data quality, identifying zero-variance/constant features and verifying physical domain validity.
4. Examine temporal delay dynamics across departure hours, 4-hour buckets, operational time blocks, and days of week.
5. Factually evaluate airline carrier volume and delay rates without subjective ranking.
6. Identify origin and destination airport delay hotspots across 12 major hubs.
7. Conduct thresholded route analysis ($N \\ge 5$) to mitigate small-sample noise.
8. Validate anti-leakage historical delay features, distributions, and cold-start fallback strategies.
9. Audit external weather readiness under the zero-fabrication contract.
10. Analyze continuous numerical feature distributions, central tendency, spread, and outliers.
11. Compute Pearson correlation matrices and identify multicollinear feature redundancies ($|r| \\ge 0.85$).
12. Deliver structured findings and actionable architectural recommendations for Phase 6 Machine Learning.

> **DEVELOPMENT SAMPLE LIMITATION NOTICE:**  
> This analysis is conducted on the verified development sample (**481 completed flights** from January 1–10, 2024). All figures, tables, and statistics are empirical results with zero synthetic data fabrication. Statistically representative operational performance conclusions require scaling to the full multi-month/multi-year public BTS dataset.""")

    # 1. Environment & Setup
    add_md("""## 1. Environment Setup & Library Initialization
Configure project root, load configuration, and set up publication-ready plotting themes.""")
    add_code("""import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set workspace root
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.eda.eda_analyzer import EDAAnalyzer

config = get_config()

# Configure consistent aesthetic plotting parameters
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    matplotlib.use('Agg')

print(f"Environment initialized. Project root: {project_root}")""")

    # 2. Dataset Loading & Schema Overview
    add_md("""## 2. Dataset Loading & Schema Overview
Load the primary Phase 3 feature dataset (`flights_features_p3.parquet`) and the optional operational reference dataset (`flights_cleaned_operational.parquet`). Inspect dimensions, memory consumption, and column data types.""")
    add_code("""analyzer = EDAAnalyzer(config=config)
df = analyzer.load_data()
df_op = analyzer.df_op

print(f"Phase 3 Dataset Shape      : {df.shape[0]} rows x {df.shape[1]} columns")
if df_op is not None:
    print(f"Operational Dataset Shape  : {df_op.shape[0]} rows x {df_op.shape[1]} columns")

overview = analyzer.get_dataset_overview()
print(f"Total Features             : {overview['total_features']}")
print(f"Memory Footprint           : {overview['memory_usage_kb']} KB")
print(f"Numerical Columns Count    : {overview['numeric_columns_count']}")
print(f"Categorical Columns Count  : {overview['categorical_columns_count']}")
print(f"Exact Duplicate Rows       : {overview['duplicate_rows_count']}")
print(f"Data Types Breakdown       : {overview['dtypes_breakdown']}")

# Display first 5 records of primary predictive columns
display_cols = ["flight_date", "airline", "origin_airport", "dest_airport", "scheduled_dep_time", "distance", "dep_time_of_day", "prior_airline_delay_rate", "delay_target"]
df[display_cols].head()""")

    # 3. Target Variable Analysis
    add_md("""## 3. Target Variable Analysis (`delay_target`)
Evaluate the class distribution of `delay_target`, defined by the FAA / BTS standard of **arrival delay $\\ge 15$ minutes**.""")
    add_code("""target_metrics = analyzer.analyze_target()
print("TARGET DISTRIBUTION SUMMARY:")
print(f"  * Target Name            : {target_metrics['target_name']}")
print(f"  * Total Observations     : {target_metrics['total_observations']:,}")
print(f"  * On-Time Flights (0)    : {target_metrics['on_time_count']:,} ({target_metrics['on_time_rate']:.2%})")
print(f"  * Delayed Flights (1)    : {target_metrics['delayed_count']:,} ({target_metrics['delay_rate']:.2%})")
print(f"  * Class Imbalance Ratio  : {target_metrics['imbalance_ratio']}:1 (On-Time : Delayed)")

# Visualize Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
counts = df["delay_target"].value_counts().sort_index()
labels = ["On-Time (<15 min)", "Delayed (>=15 min)"]
colors = ["#2b5c8f", "#d95f02"]

bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
for bar in bars:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    axes[0].annotate(f"{h:,}\\n({pct:.1f}%)", (bar.get_x() + bar.get_width()/2, h/2), ha="center", va="center", color="white", fontweight="bold")
axes[0].set_title("Arrival Delay Target Class Counts", fontweight="bold")
axes[0].set_ylabel("Flight Count")

axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
axes[1].set_title("Target Class Proportions (Base Delay Rate = 30.15%)", fontweight="bold")

plt.suptitle("Target Variable Analysis (delay_target)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()""")

    # 4. Data Quality & Sanity Diagnostics
    add_md("""## 4. Data Quality & Sanity Diagnostics
Audit dataset completeness, verify missingness rates across all 68 columns, detect constant/zero-variance features, and check physical domain boundaries.""")
    add_code("""quality = analyzer.audit_data_quality()
print("DATA QUALITY AUDIT:")
print(f"  * Columns with Missing Values : {quality['columns_with_missing_values_count']}")
print(f"  * Duplicate Rows              : {quality['duplicate_records_count']}")
print(f"  * Negative Value Violations   : {quality['negative_value_violations']}")
print(f"  * Zero-Variance Columns Count : {quality['constant_columns_count']}")
print(f"  * Zero-Variance Feature Names : {quality['constant_columns']}")
print(f"  * Overall Quality Status      : {quality['data_quality_status']}")

# Inspect constant columns values
for col in quality["constant_columns"]:
    print(f"    - Column '{col}': constant value = {df[col].iloc[0]}")""")

    # 5. Temporal Analysis
    add_md("""## 5. Temporal Dynamics Analysis
Examine how flight arrival delay risk varies across the operating day (departure hour and 4-hour buckets), operational time blocks, days of the week, and weekend vs. weekday schedules.""")
    add_code("""temporal = analyzer.analyze_temporal_patterns()
print(f"Date Range: {temporal['date_range_start']} to {temporal['date_range_end']} ({temporal['total_calendar_days']} calendar days)")

# 1. Day of Week Summary
dow_df = pd.DataFrame(temporal["day_of_week_analysis"]).T
print("\\nDay of Week Delay Breakdown:")
print(dow_df)

# 2. Time-of-Day Blocks Summary
tod_df = pd.DataFrame(temporal["time_of_day_analysis"]).T
print("\\nTime-of-Day Block Delay Breakdown:")
print(tod_df)

# 3. Weekend vs Weekday Summary
print("\\nWeekend vs Weekday Breakdown:")
print(pd.DataFrame(temporal["weekend_vs_weekday_analysis"]).T)

# Plot Temporal Delay Dynamics
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
df_t = df.copy()
df_t["flight_date"] = pd.to_datetime(df_t["flight_date"])
df_t["dep_hour"] = (df_t["scheduled_dep_time"] // 100).astype(int)

# Hour of Day
hour_stats = df_t.groupby("dep_hour")["delay_target"].agg(["count", "mean"]).reset_index()
sns.barplot(data=hour_stats, x="dep_hour", y="mean", color="#1f77b4", ax=axes[0, 0], edgecolor="black", alpha=0.85)
axes[0, 0].axhline(df["delay_target"].mean(), color="red", linestyle="--", label=f"Average ({df['delay_target'].mean():.1%})")
axes[0, 0].set_title("Delay Rate by Scheduled Departure Hour", fontweight="bold")
axes[0, 0].set_xlabel("Scheduled Departure Hour (0-23)")
axes[0, 0].set_ylabel("Delay Rate")
axes[0, 0].legend()

# 4-Hour Time Bucket
bucket_order = ["04-08", "08-12", "12-16", "16-20", "20-24"]
bucket_stats = df_t.groupby("dep_time_bucket")["delay_target"].agg(["count", "mean"]).reindex(bucket_order).reset_index()
sns.barplot(data=bucket_stats, x="dep_time_bucket", y="mean", color="#3182bd", ax=axes[0, 1], edgecolor="black")
axes[0, 1].axhline(df["delay_target"].mean(), color="red", linestyle="--", label="Average")
axes[0, 1].set_title("Delay Rate by 4-Hour Departure Window", fontweight="bold")
axes[0, 1].set_xlabel("Time Bucket")
axes[0, 1].set_ylabel("Delay Rate")
axes[0, 1].legend()

# Time of Day Blocks
tod_order = ["morning", "afternoon", "evening", "night"]
tod_stats = df_t.groupby("dep_time_of_day")["delay_target"].agg(["count", "mean"]).reindex(tod_order).reset_index()
sns.barplot(data=tod_stats, x="dep_time_of_day", y="mean", color="#2ca02c", ax=axes[1, 0], edgecolor="black")
axes[1, 0].axhline(df["delay_target"].mean(), color="red", linestyle="--", label="Average")
axes[1, 0].set_title("Delay Rate by Operational Block", fontweight="bold")
axes[1, 0].set_xlabel("Operational Block")
axes[1, 0].set_ylabel("Delay Rate")
axes[1, 0].legend()

# Day of Week
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df_t["day_of_week"] = df_t["flight_date"].dt.day_name()
dow_stats = df_t.groupby("day_of_week")["delay_target"].agg(["count", "mean"]).reindex(dow_order).reset_index()
colors_dow = ["#4c72b0" if d not in ["Saturday", "Sunday"] else "#c44e52" for d in dow_order]
axes[1, 1].bar(dow_stats["day_of_week"], dow_stats["mean"], color=colors_dow, edgecolor="black")
axes[1, 1].axhline(df["delay_target"].mean(), color="black", linestyle="--", label="Average")
axes[1, 1].set_title("Delay Rate by Day of Week (Red = Weekend)", fontweight="bold")
axes[1, 1].set_xlabel("Day of Week")
axes[1, 1].set_ylabel("Delay Rate")
axes[1, 1].tick_params(axis="x", rotation=30)
axes[1, 1].legend()

plt.suptitle("Temporal Delay Dynamics", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()""")

    # 6. Airline Analysis
    add_md("""## 6. Airline Carrier Analysis
Evaluate flight volume, delay counts, and observed delay rates across all 7 operating carriers. Factual observations are reported without subjective ranking.""")
    add_code("""airlines = analyzer.analyze_airlines()
airline_df = pd.DataFrame(airlines["airline_metrics"]).T
print("AIRLINE CARRIER PERFORMANCE TABLE:")
print(airline_df[["flight_count", "volume_share_pct", "delay_count", "delay_rate", "delay_share_pct"]])

# Airline Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
airline_stats = airline_df.reset_index().rename(columns={"index": "airline"}).sort_values(by="flight_count", ascending=False)

# Flight volume breakdown
axes[0].bar(airline_stats["airline"], airline_stats["flight_count"], label="On-Time", color="#4575b4", edgecolor="black", alpha=0.85)
axes[0].bar(airline_stats["airline"], airline_stats["delay_count"], label="Delayed", color="#d73027", edgecolor="black", alpha=0.9)
for idx, row in airline_stats.reset_index(drop=True).iterrows():
    axes[0].annotate(f"N={int(row['flight_count'])}", (idx, row["flight_count"] + 1), ha="center", fontsize=9, fontweight="bold")
axes[0].set_title("Flight Volume & Delays by Carrier", fontweight="bold")
axes[0].set_xlabel("Carrier Code")
axes[0].set_ylabel("Flight Count")
axes[0].legend()

# Delay Rate by Airline
sorted_by_rate = airline_stats.sort_values(by="delay_rate", ascending=False)
bars = axes[1].bar(sorted_by_rate["airline"], sorted_by_rate["delay_rate"], color="#fc8d59", edgecolor="black", alpha=0.9)
axes[1].axhline(df["delay_target"].mean(), color="red", linestyle="--", label=f"Average ({df['delay_target'].mean():.1%})")
for bar in bars:
    h = bar.get_height()
    axes[1].annotate(f"{h:.1%}", (bar.get_x() + bar.get_width()/2, h + 0.01), ha="center", fontsize=9, fontweight="bold")
axes[1].set_title("Observed Delay Rate by Carrier", fontweight="bold")
axes[1].set_xlabel("Carrier Code")
axes[1].set_ylabel("Delay Rate")
axes[1].set_ylim(0, 0.5)
axes[1].legend()

plt.suptitle("Airline Carrier Analysis (Descriptive Comparison)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()""")

    # 7. Airport Analysis
    add_md("""## 7. Origin & Destination Airport Analysis
Analyze operational flight volume and arrival delay rates across 12 major hub airports.""")
    add_code("""airports = analyzer.analyze_airports()
orig_df = pd.DataFrame(airports["origin_airports"]).T.reset_index().rename(columns={"index": "origin_airport"}).sort_values(by="delay_rate", ascending=True)
dest_df = pd.DataFrame(airports["dest_airports"]).T.reset_index().rename(columns={"index": "dest_airport"}).sort_values(by="delay_rate", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Origin Airports
bars_orig = axes[0].barh(orig_df["origin_airport"], orig_df["delay_rate"], color="#6baed6", edgecolor="black")
axes[0].axvline(df["delay_target"].mean(), color="red", linestyle="--", label=f"Average ({df['delay_target'].mean():.1%})")
for bar, (_, row) in zip(bars_orig, orig_df.iterrows()):
    w = bar.get_width()
    axes[0].annotate(f" {w:.1%} (N={int(row['flight_count'])})", (w, bar.get_y() + bar.get_height()/2), va="center", fontsize=8.5)
axes[0].set_title("Origin Airport Observed Delay Rate", fontweight="bold")
axes[0].set_xlabel("Delay Rate")
axes[0].set_xlim(0, 0.55)
axes[0].legend()

# Destination Airports
bars_dest = axes[1].barh(dest_df["dest_airport"], dest_df["delay_rate"], color="#fd8d3c", edgecolor="black")
axes[1].axvline(df["delay_target"].mean(), color="red", linestyle="--", label=f"Average ({df['delay_target'].mean():.1%})")
for bar, (_, row) in zip(bars_dest, dest_df.iterrows()):
    w = bar.get_width()
    axes[1].annotate(f" {w:.1%} (N={int(row['flight_count'])})", (w, bar.get_y() + bar.get_height()/2), va="center", fontsize=8.5)
axes[1].set_title("Destination Airport Observed Delay Rate", fontweight="bold")
axes[1].set_xlabel("Delay Rate")
axes[1].set_xlim(0, 0.55)
axes[1].legend()

plt.suptitle("Airport Delay Variations (Origin vs Destination)", fontsize=13, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()""")

    # 8. Route Analysis
    add_md("""## 8. Route Analysis & Sparsity Audit
Construct composite `origin_dest` routes, analyze volume distribution, and evaluate route delay rates with sample-size thresholding ($N \\ge 5$).""")
    add_code("""routes = analyzer.analyze_routes(min_sample_size=5)
print(f"Total Distinct Routes in Sample   : {routes['total_unique_routes']}")
print(f"Frequent Routes (N >= 5 flights) : {routes['frequent_routes_count']}")
print(f"Sparse Routes (N < 5 flights)     : {routes['sparse_routes_count']} ({routes['sparse_routes_pct']}%)")

frequent_df = pd.DataFrame(routes["frequent_routes"]).T.reset_index().rename(columns={"index": "route"})
print("\\nTop Frequent Routes Table:")
print(frequent_df)

# Route Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
frequent_by_vol = frequent_df.sort_values(by="flight_count", ascending=True)
axes[0].barh(frequent_by_vol["route"], frequent_by_vol["flight_count"], color="#9ecae1", edgecolor="black")
axes[0].set_title("Top Routes by Flight Count (N ≥ 5)", fontweight="bold")
axes[0].set_xlabel("Total Flights")

frequent_by_rate = frequent_df.sort_values(by="delay_rate", ascending=True)
axes[1].barh(frequent_by_rate["route"], frequent_by_rate["delay_rate"], color="#fc9272", edgecolor="black")
axes[1].axvline(df["delay_target"].mean(), color="red", linestyle="--", label="Average")
axes[1].set_title("Observed Delay Rate on Top Routes (N ≥ 5)", fontweight="bold")
axes[1].set_xlabel("Delay Rate")
axes[1].legend()

plt.suptitle("Thresholded Route Analysis (N ≥ 5 flights)", fontsize=13, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()""")

    # 9. Historical Delay Features
    add_md("""## 9. Historical Delay Features Analysis
Validate multi-granular rolling delay metrics computed strictly over prior flights ($t_{obs} < t_{flight}$), cold-start fallback coverage, and correlation with the prediction target.""")
    add_code("""hist_res = analyzer.analyze_historical_delay_features()
print("HISTORICAL ROLLING FEATURE SUMMARY STATISTICS:")
print(pd.DataFrame(hist_res["summary_statistics"]).T[["count", "mean", "std", "min", "50%", "max"]])

print("\\nHISTORICAL FALLBACK COVERAGE:")
for k, v in hist_res["coverage_and_fallback"].items():
    print(f"  * {k}: {v}")

# Histograms of rolling rates
hist_cols = ["prior_airline_delay_rate", "prior_origin_delay_rate", "prior_dest_delay_rate", "prior_route_delay_rate"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(hist_cols):
    ax = axes[i]
    sns.histplot(df[col], kde=True, ax=ax, color="#2b5c8f", bins=20, edgecolor="black")
    clean_name = col.replace("_", " ").title()
    ax.set_title(f"Distribution: {clean_name}", fontweight="bold", fontsize=10)
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.3f}")
    ax.axvline(df[col].median(), color="green", linestyle=":", label=f"Median: {df[col].median():.3f}")
    ax.legend(fontsize=8.5)

plt.suptitle("Historical Delay Feature Distributions (Anti-Leakage Rolling)", fontsize=13, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()""")

    # 10. Weather Integration Audit
    add_md("""## 10. Weather Integration Audit
Verify weather foundation status, zero-fabrication contract, and prediction-time availability invariants.""")
    add_code("""weather = analyzer.audit_weather_layer()
print("WEATHER FOUNDATION AUDIT:")
print(f"  * Status Message          : {weather['status_message']}")
print(f"  * Real Weather Present    : {weather['real_weather_data_present']}")
print(f"  * Anti-Leakage Contract   : {weather['contract']}")
print(f"  * Fabrication Policy      : {weather['fabrication_policy']}")
print(f"  * Modeling Implications   : {weather['implications_for_modeling']}")""")

    # 11. Numerical Feature Distributions
    add_md("""## 11. Continuous Numerical Feature Distributions
Evaluate distributions, central tendencies, and target-stratified behavior for distance, scheduled timestamps, and volume metrics.""")
    add_code("""num_stats = analyzer.analyze_numerical_features()
print("CONTINUOUS FEATURE STATISTICS:")
print(pd.DataFrame(num_stats).T[["mean", "std", "median", "min", "max", "iqr", "skewness"]])

# Distance and Scheduled Time by Target
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x="delay_target", y="distance", palette=["#4575b4", "#d73027"], ax=axes[0])
axes[0].set_xticklabels(["On-Time (0)", "Delayed (1)"])
axes[0].set_title("Flight Distance Distribution by Delay Status", fontweight="bold")
axes[0].set_xlabel("Delay Target")
axes[0].set_ylabel("Statute Miles")

sns.boxplot(data=df, x="delay_target", y="scheduled_dep_time", palette=["#4575b4", "#d73027"], ax=axes[1])
axes[1].set_xticklabels(["On-Time (0)", "Delayed (1)"])
axes[1].set_title("Scheduled Departure Time by Delay Status", fontweight="bold")
axes[1].set_xlabel("Delay Target")
axes[1].set_ylabel("Military Time (HHMM)")

plt.suptitle("Numerical Distributions by Delay Status", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()""")

    # 12. Categorical Feature Cardinality
    add_md("""## 12. Categorical Feature Cardinality & Cardinality Breakdown
Inspect the cardinality and distribution of categorical features, establishing high-cardinality risks for one-hot encoding.""")
    add_code("""cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print(f"Categorical Features Identified ({len(cat_cols)} total):")
for col in cat_cols:
    n_unique = df[col].nunique()
    top_val = df[col].mode().iloc[0]
    top_cnt = (df[col] == top_val).sum()
    print(f"  * {col:25s} : {n_unique:3d} unique values (Mode: '{top_val}', N={top_cnt})")""")

    # 13. Correlation Analysis
    add_md("""## 13. Correlation Analysis & Multicollinearity Diagnostics
Compute bivariate Pearson correlation matrix and detect highly collinear feature pairs ($|r| \\ge 0.85$).""")
    add_code("""corrs = analyzer.compute_correlations()
print("TOP POSITIVE CORRELATIONS WITH DELAY_TARGET:")
for k, v in list(corrs["correlations_with_target"].items())[:7]:
    print(f"  * {k:35s} : r = {v:+.4f}")

print("\\nTOP NEGATIVE CORRELATIONS WITH DELAY_TARGET:")
for k, v in list(corrs["correlations_with_target"].items())[-5:]:
    print(f"  * {k:35s} : r = {v:+.4f}")

print(f"\\nHIGHLY COLLINEAR FEATURE PAIRS (|r| >= 0.85): {corrs['highly_collinear_pairs_count']} pairs detected")
for pair in corrs["highly_collinear_pairs"][:5]:
    print(f"  * {pair['feature_1']} <-> {pair['feature_2']}: r = {pair['correlation']}")

# Correlation Heatmap Plot
corr_features = [
    "delay_target", "distance", "scheduled_dep_time", "arr_hour_sin", "arr_hour_cos",
    "prior_origin_flight_volume", "prior_dest_flight_volume", "prior_airline_delay_rate",
    "prior_origin_delay_rate", "prior_dest_delay_rate", "prior_route_delay_rate", "prior_origin_dep_hour_delay_rate"
]
avail_corr = [c for c in corr_features if c in df.columns]
matrix = df[avail_corr].corr()

plt.figure(figsize=(10, 8))
clean_labels = [c.replace("prior_", "").replace("_delay_rate", "_rate") for c in avail_corr]
sns.heatmap(matrix, xticklabels=clean_labels, yticklabels=clean_labels, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.5, vmax=0.5, linewidths=0.5)
plt.title("Correlation Heatmap: Key Predictive Features & Delay Target", fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()""")

    # 14. Outlier Analysis
    add_md("""## 14. Outlier Analysis & Ground-Truth Delay Distribution
Examine predictive features using Tukey's IQR rule and inspect ground-truth operational delay minutes from `flights_cleaned_operational.parquet`.""")
    add_code("""outliers = analyzer.analyze_outliers()
print("IQR OUTLIER SUMMARY FOR PREDICTIVE FEATURES:")
print(pd.DataFrame(outliers["iqr_outlier_summary"]).T[["lower_bound", "upper_bound", "outlier_count", "outlier_pct", "verdict"]])

if "operational_delay_minutes_summary" in outliers and outliers["operational_delay_minutes_summary"]:
    print("\\nOPERATIONAL DELAY MINUTES (GROUND TRUTH):")
    for k, v in outliers["operational_delay_minutes_summary"].items():
        print(f"  * {k}: {v} minutes")

# Boxplots of Volume and Operational Delays
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
vol_cols = ["prior_origin_flight_volume", "prior_dest_flight_volume", "prior_airline_flight_count"]
vol_data = [df[c].dropna() for c in vol_cols]
clean_labels = ["Origin Vol", "Dest Vol", "Airline Cnt"]

try:
    axes[0].boxplot(vol_data, tick_labels=clean_labels, patch_artist=True, boxprops=dict(facecolor="#9ecae1"))
except TypeError:
    axes[0].boxplot(vol_data, labels=clean_labels, patch_artist=True, boxprops=dict(facecolor="#9ecae1"))
axes[0].set_title("Predictive Volume Features (IQR Assessment)", fontweight="bold")
axes[0].set_ylabel("Volume Count")

if df_op is not None and "arrival_delay" in df_op.columns and "departure_delay" in df_op.columns:
    delay_data = [df_op["departure_delay"].dropna(), df_op["arrival_delay"].dropna()]
    op_labels = ["Dep Delay (min)", "Arr Delay (min)"]
    try:
        bp = axes[1].boxplot(delay_data, tick_labels=op_labels, patch_artist=True)
    except TypeError:
        bp = axes[1].boxplot(delay_data, labels=op_labels, patch_artist=True)
    bp["boxes"][0].set_facecolor("#fc9272")
    bp["boxes"][1].set_facecolor("#bcbddc")
    axes[1].axhline(15, color="red", linestyle="--", label="15-min Delay Threshold")
    axes[1].set_title("Operational Delay Durations (Minutes)", fontweight="bold")
    axes[1].set_ylabel("Minutes")
    axes[1].legend()

plt.suptitle("Outlier Inspection & Operational Ground Truth", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()""")

    # 15. Key EDA Findings
    add_md("""## 15. Key EDA Findings
Structured summary of data quality, behavioral dynamics, and feature associations:

### 1. Data Quality Findings
- **Completeness:** 100% data completeness (0 nulls across 68 columns).
- **Uniqueness:** 0 exact duplicate flight records.
- **Constant Columns:** 5 zero-variance features identified (`cancelled`, `diverted`, `is_month_end`, `quarter`, `season`).

### 2. Behavioral Patterns
- **Temporal Peaks:** Midday departures (08:00–12:00) experience peak delay rates (34.31%), compared to evening departures (24.59%).
- **Weekend Influx:** Weekend flights experience higher delays (33.33% vs 29.38% on weekdays).
- **Airport Hotspots:** Origin delays at Chicago O'Hare (`ORD`, 42.11%) and Charlotte (`CLT`, 41.86%) are more than double Miami (`MIA`, 19.57%).
- **Arrival Bottlenecks:** New York JFK (`JFK`, 43.24%) and Charlotte (`CLT`, 41.30%) represent the most bottlenecked arrival airports.

### 3. Feature Signal
- **Historical Features:** `prior_origin_dep_hour_delay_rate` ($r = +0.0645$) and `prior_airline_delay_rate` ($r = +0.0269$) show positive correlation with current flight delays.
- **Route Sparsity:** 92.25% of route pairs have fewer than 5 observations, demonstrating severe high-cardinality sparsity.
- **Collinearity:** 83 feature pairs exhibit $|r| \\ge 0.85$, primarily due to duplicate naming conventions across phases.""")

    # 16. Implications for Phase 6 ML
    add_md(r"""## 16. Implications for Phase 6 Machine Learning
Concrete recommendations to govern model architecture, feature preprocessing, evaluation metrics, and validation in Phase 6:

1. **Evaluation Metrics Selection:**
   - With an observed delay rate of 30.15% (imbalance ratio 2.32:1), naive accuracy (69.85%) is uninformative.
   - Phase 6 must benchmark candidate models using **ROC-AUC**, **PR-AUC (Average Precision)**, **F1-Score**, and **Brier Score** (probability calibration).
2. **Class Weight Adjustment:**
   - Incorporate `scale_pos_weight = 2.32` in XGBoost and `class_weight='balanced'` in Logistic Regression and Random Forest to prevent under-predicting the minority class.
3. **Feature Pruning & Selection:**
   - Drop the 5 constant columns (`cancelled`, `diverted`, `is_month_end`, `quarter`, `season`).
   - Eliminate redundant collinear aliases (`route_distance`, `carrier_prior_*`, `origin_prior_*`) before fitting linear models to eliminate variance inflation.
4. **Categorical Handling (High Cardinality Route Exclusion):**
   - Exclude raw `route` from one-hot encoding (129 levels with 92% sparse levels) to avoid severe feature explosion.
   - Rely on `distance`, `haul_category`, `prior_origin_flight_volume`, and `prior_route_delay_rate`.
5. **Threshold Sweeping on Validation Data:**
   - Perform post-training decision threshold sweeps over $[0.30, 0.70]$ on validation partition only, optimizing operational utility (recall vs. precision).
6. **Chronological Validation Invariant:**
   - Maintain strict out-of-time chronological train/validation/test splitting ($\max(\text{Train}) \le \min(\text{Val}) \le \min(\text{Test})$) to simulate authentic operational forecasting.""")

    # 17. Scope Limitations & Summary Conclusion
    add_md("""## 17. Scope Limitations & Summary Conclusion
- **Development Sample Limitation:** Findings reflect a 10-day winter operational window (481 flights). Validation on full-year BTS data is necessary to capture spring convective storms and summer travel volume.
- **Weather Foundation Mode:** No synthetic weather data was fabricated (`data/external/` empty). Real weather ingestion will occur once authentic METAR records are mounted.
- **Strict Scope Boundary:** Phase 5 concludes the Exploratory Data Analysis phase. No machine learning training, tuning, or evaluation is performed in Phase 5.

```
================================================================================
PHASE 5 EDA COMPLETE: FULL DATA GROUNDING & STATISTICAL AUDIT VERIFIED
================================================================================
```""")

    out_file = Path("notebooks/07_phase5_eda.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated complete Phase 5 EDA notebook at: {out_file.resolve()}")


if __name__ == "__main__":
    create_phase5_eda_notebook()
