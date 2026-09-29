"""Generate 9 publication-grade Phase 5 EDA figures for reports/figures/."""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda.eda_analyzer import EDAAnalyzer
from src.utils.config import get_config


def set_plotting_style():
    """Apply consistent, professional styling for all charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def plot_target_distribution(df: pd.DataFrame, output_dir: Path):
    """Figure 1: Target Variable Distribution (Counts & Percentages)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target_counts = df["delay_target"].value_counts().sort_index()
    labels = ["On-Time (<15 min)", "Delayed (≥15 min)"]
    colors = ["#2b5c8f", "#d95f02"]

    # Bar chart
    bars = axes[0].bar(labels, target_counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        axes[0].annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
    axes[0].set_title("Flight Arrival Delay Target Distribution (Counts)", fontweight="bold")
    axes[0].set_ylabel("Number of Flights")
    axes[0].set_ylim(0, target_counts.max() * 1.15)

    # Donut chart
    axes[1].pie(
        target_counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    axes[1].set_title("Target Class Proportions (Base Delay Rate = 30.15%)", fontweight="bold")

    plt.suptitle("Phase 5 EDA — Target Variable Analysis (delay_target)", fontsize=14, fontweight="bold", y=1.02)
    out_path = output_dir / "phase5_target_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_temporal_patterns(df: pd.DataFrame, output_dir: Path):
    """Figure 2: Multi-Panel Temporal Delay Dynamics."""
    df_t = df.copy()
    df_t["flight_date"] = pd.to_datetime(df_t["flight_date"])
    df_t["day_of_week"] = df_t["flight_date"].dt.day_name()
    df_t["dep_hour"] = (df_t["scheduled_dep_time"] // 100).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Hour of Day
    hour_stats = df_t.groupby("dep_hour")["delay_target"].agg(["count", "mean"]).reset_index()
    sns.barplot(data=hour_stats, x="dep_hour", y="mean", color="#1f77b4", ax=axes[0, 0], edgecolor="black", alpha=0.85)
    axes[0, 0].axhline(df["delay_target"].mean(), color="red", linestyle="--", label=f"Average ({df['delay_target'].mean():.1%})")
    axes[0, 0].set_title("Observed Delay Rate by Scheduled Departure Hour", fontweight="bold")
    axes[0, 0].set_xlabel("Scheduled Departure Hour (0-23)")
    axes[0, 0].set_ylabel("Delay Rate")
    axes[0, 0].set_ylim(0, max(hour_stats["mean"].max() * 1.2, 0.6))
    axes[0, 0].legend()

    # 2. Time Bucket (4-hour)
    bucket_order = ["04-08", "08-12", "12-16", "16-20", "20-24"]
    bucket_stats = df_t.groupby("dep_time_bucket")["delay_target"].agg(["count", "mean"]).reindex(bucket_order).reset_index()
    sns.barplot(data=bucket_stats, x="dep_time_bucket", y="mean", palette="Blues_r", ax=axes[0, 1], edgecolor="black")
    axes[0, 1].axhline(df["delay_target"].mean(), color="red", linestyle="--", label="Average")
    for idx, row in bucket_stats.iterrows():
        if pd.notna(row["mean"]):
            axes[0, 1].annotate(f"{row['mean']:.1%}\n(N={int(row['count'])})", (idx, row["mean"] + 0.02), ha="center", fontsize=9)
    axes[0, 1].set_title("Observed Delay Rate by 4-Hour Time Bucket", fontweight="bold")
    axes[0, 1].set_xlabel("Departure Window")
    axes[0, 1].set_ylabel("Delay Rate")
    axes[0, 1].set_ylim(0, 0.5)

    # 3. Time of Day Blocks
    tod_order = ["morning", "afternoon", "evening", "night"]
    tod_stats = df_t.groupby("dep_time_of_day")["delay_target"].agg(["count", "mean"]).reindex(tod_order).reset_index()
    sns.barplot(data=tod_stats, x="dep_time_of_day", y="mean", palette="viridis", ax=axes[1, 0], edgecolor="black")
    axes[1, 0].axhline(df["delay_target"].mean(), color="red", linestyle="--", label="Average")
    for idx, row in tod_stats.iterrows():
        axes[1, 0].annotate(f"{row['mean']:.1%}\n(N={int(row['count'])})", (idx, row["mean"] + 0.02), ha="center", fontsize=9)
    axes[1, 0].set_title("Observed Delay Rate by Operational Time-of-Day Block", fontweight="bold")
    axes[1, 0].set_xlabel("Operational Block")
    axes[1, 0].set_ylabel("Delay Rate")
    axes[1, 0].set_ylim(0, 0.45)

    # 4. Day of Week
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = df_t.groupby("day_of_week")["delay_target"].agg(["count", "mean"]).reindex(dow_order).reset_index()
    bar_colors = ["#4c72b0" if d not in ["Saturday", "Sunday"] else "#c44e52" for d in dow_order]
    sns.barplot(data=dow_stats, x="day_of_week", y="mean", palette=bar_colors, ax=axes[1, 1], edgecolor="black")
    axes[1, 1].axhline(df["delay_target"].mean(), color="black", linestyle="--", label="Average")
    axes[1, 1].set_title("Observed Delay Rate by Day of Week (Red = Weekend)", fontweight="bold")
    axes[1, 1].set_xlabel("Day of Week")
    axes[1, 1].set_ylabel("Delay Rate")
    axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].set_ylim(0, 0.55)

    plt.suptitle("Phase 5 EDA — Temporal Flight Delay Dynamics", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = output_dir / "phase5_temporal_delay_patterns.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_airline_performance(df: pd.DataFrame, output_dir: Path):
    """Figure 3: Airline Flight Volume and Observed Delay Rate."""
    airline_stats = (
        df.groupby("airline")["delay_target"]
        .agg(total="count", delayed="sum", delay_rate="mean")
        .sort_values(by="total", ascending=False)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Volume & Delayed Stack
    axes[0].bar(airline_stats["airline"], airline_stats["total"], label="On-Time", color="#4575b4", edgecolor="black", alpha=0.85)
    axes[0].bar(airline_stats["airline"], airline_stats["delayed"], label="Delayed", color="#d73027", edgecolor="black", alpha=0.9)
    for idx, row in airline_stats.iterrows():
        axes[0].annotate(f"N={row['total']}", (idx, row["total"] + 1), ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Flight Volume & Delay Counts by Carrier", fontweight="bold")
    axes[0].set_xlabel("Airline Carrier Code")
    axes[0].set_ylabel("Flight Count")
    axes[0].set_ylim(0, airline_stats["total"].max() * 1.15)
    axes[0].legend()

    # Delay Rate
    airline_stats_sorted = airline_stats.sort_values(by="delay_rate", ascending=False)
    bars = axes[1].bar(
        airline_stats_sorted["airline"],
        airline_stats_sorted["delay_rate"],
        color="#fc8d59",
        edgecolor="black",
        alpha=0.9,
    )
    axes[1].axhline(df["delay_target"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Fleet Average ({df['delay_target'].mean():.1%})")
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f"{h:.1%}", (bar.get_x() + bar.get_width() / 2, h + 0.01), ha="center", fontsize=10, fontweight="bold")
    axes[1].set_title("Observed Delay Rate by Carrier", fontweight="bold")
    axes[1].set_xlabel("Airline Carrier Code")
    axes[1].set_ylabel("Delay Rate")
    axes[1].set_ylim(0, max(airline_stats_sorted["delay_rate"].max() * 1.25, 0.5))
    axes[1].legend()

    plt.suptitle("Phase 5 EDA — Airline Carrier Performance (Descriptive Comparison)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "phase5_airline_delay_performance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_airport_hotspots(df: pd.DataFrame, output_dir: Path):
    """Figure 4: Origin and Destination Airport Congestion & Delay Rates."""
    orig_stats = (
        df.groupby("origin_airport")["delay_target"]
        .agg(total="count", delay_rate="mean")
        .sort_values(by="delay_rate", ascending=True)
        .reset_index()
    )
    dest_stats = (
        df.groupby("dest_airport")["delay_target"]
        .agg(total="count", delay_rate="mean")
        .sort_values(by="delay_rate", ascending=True)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Origin
    bars_orig = axes[0].barh(orig_stats["origin_airport"], orig_stats["delay_rate"], color="#6baed6", edgecolor="black")
    axes[0].axvline(df["delay_target"].mean(), color="red", linestyle="--", label=f"Average ({df['delay_target'].mean():.1%})")
    for bar, (_, row) in zip(bars_orig, orig_stats.iterrows()):
        w = bar.get_width()
        axes[0].annotate(f" {w:.1%} (N={row['total']})", (w, bar.get_y() + bar.get_height() / 2), va="center", fontsize=9)
    axes[0].set_title("Origin Airport Observed Delay Rate (N=Total Flights)", fontweight="bold")
    axes[0].set_xlabel("Delay Rate")
    axes[0].set_xlim(0, max(orig_stats["delay_rate"].max() * 1.35, 0.55))
    axes[0].legend()

    # Destination
    bars_dest = axes[1].barh(dest_stats["dest_airport"], dest_stats["delay_rate"], color="#fd8d3c", edgecolor="black")
    axes[1].axvline(df["delay_target"].mean(), color="red", linestyle="--", label=f"Average ({df['delay_target'].mean():.1%})")
    for bar, (_, row) in zip(bars_dest, dest_stats.iterrows()):
        w = bar.get_width()
        axes[1].annotate(f" {w:.1%} (N={row['total']})", (w, bar.get_y() + bar.get_height() / 2), va="center", fontsize=9)
    axes[1].set_title("Destination Airport Observed Delay Rate (N=Total Flights)", fontweight="bold")
    axes[1].set_xlabel("Delay Rate")
    axes[1].set_xlim(0, max(dest_stats["delay_rate"].max() * 1.35, 0.55))
    axes[1].legend()

    plt.suptitle("Phase 5 EDA — Airport Delay Variation (Origin vs Destination)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = output_dir / "phase5_airport_hotspots.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_route_analysis(df: pd.DataFrame, output_dir: Path):
    """Figure 5: Route Analysis (Top Frequent Routes N >= 5)."""
    route_stats = (
        df.groupby("route")["delay_target"]
        .agg(total="count", delayed="sum", delay_rate="mean")
        .reset_index()
    )
    # Top 12 routes by volume (with tie-break on delay rate)
    frequent = (
        route_stats[route_stats["total"] >= 5]
        .sort_values(by=["total", "delay_rate"], ascending=[False, False])
        .head(12)
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Route volume
    frequent_by_vol = frequent.sort_values(by="total", ascending=True).reset_index(drop=True)
    sns.barplot(data=frequent_by_vol, x="total", y="route", color="#9ecae1", ax=axes[0], edgecolor="black")
    for idx, row in frequent_by_vol.iterrows():
        axes[0].annotate(f" {int(row['total'])} flights", (row["total"], idx), va="center", fontsize=9.5, fontweight="bold")
    axes[0].set_title("Top 12 Frequent Routes by Flight Count (Sample Size N ≥ 5)", fontweight="bold")
    axes[0].set_xlabel("Total Flights")
    axes[0].set_ylabel("Route (Origin_Dest)")
    axes[0].set_xlim(0, frequent["total"].max() * 1.25)

    # Route delay rate
    frequent_by_rate = frequent.sort_values(by="delay_rate", ascending=True).reset_index(drop=True)
    sns.barplot(data=frequent_by_rate, x="delay_rate", y="route", color="#fc9272", ax=axes[1], edgecolor="black")
    axes[1].axvline(df["delay_target"].mean(), color="red", linestyle="--", label=f"Average ({df['delay_target'].mean():.1%})")
    for idx, row in frequent_by_rate.iterrows():
        axes[1].annotate(f" {row['delay_rate']:.1%} (N={int(row['total'])})", (row["delay_rate"], idx), va="center", fontsize=9.5)
    axes[1].set_title("Observed Delay Rate on Top Frequent Routes (N ≥ 5)", fontweight="bold")
    axes[1].set_xlabel("Delay Rate")
    axes[1].set_ylabel("Route (Origin_Dest)")
    axes[1].set_xlim(0, max(frequent_by_rate["delay_rate"].max() * 1.35, 0.6))
    axes[1].legend()

    plt.suptitle("Phase 5 EDA — Route Analysis (Thresholded Sample Size N ≥ 5)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = output_dir / "phase5_route_delay_analysis.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_historical_features(df: pd.DataFrame, output_dir: Path):
    """Figure 6: Historical Delay Feature Distributions & Target Relationships."""
    hist_cols = [
        "prior_airline_delay_rate",
        "prior_origin_delay_rate",
        "prior_dest_delay_rate",
        "prior_route_delay_rate",
    ]
    available = [c for c in hist_cols if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(available):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color="#2b5c8f", bins=20, edgecolor="black")
        clean_name = col.replace("_", " ").title()
        ax.set_title(f"Distribution of {clean_name}", fontweight="bold")
        ax.set_xlabel(clean_name)
        ax.set_ylabel("Flight Count")

        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
        ax.axvline(median_val, color="green", linestyle=":", label=f"Median: {median_val:.3f}")
        ax.legend(fontsize=9)

    plt.suptitle("Phase 5 EDA — Historical Delay Feature Distributions (Anti-Leakage Rolling)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = output_dir / "phase5_historical_delay_features.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_numerical_distributions(df: pd.DataFrame, output_dir: Path):
    """Figure 7: Continuous Feature Distributions & Group Comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distance by Target
    sns.boxplot(data=df, x="delay_target", y="distance", palette=["#4575b4", "#d73027"], ax=axes[0, 0])
    axes[0, 0].set_xticklabels(["On-Time (0)", "Delayed (1)"])
    axes[0, 0].set_title("Flight Distance Distribution by Delay Status", fontweight="bold")
    axes[0, 0].set_xlabel("Delay Target")
    axes[0, 0].set_ylabel("Flight Distance (statute miles)")

    # 2. Scheduled Departure Time by Target
    sns.boxplot(data=df, x="delay_target", y="scheduled_dep_time", palette=["#4575b4", "#d73027"], ax=axes[0, 1])
    axes[0, 1].set_xticklabels(["On-Time (0)", "Delayed (1)"])
    axes[0, 1].set_title("Scheduled Departure Time by Delay Status", fontweight="bold")
    axes[0, 1].set_xlabel("Delay Target")
    axes[0, 1].set_ylabel("Military Time (HHMM)")

    # 3. Distance Category Delay Rates
    dist_cat = df.groupby("distance_category")["delay_target"].agg(["count", "mean"]).reset_index()
    sns.barplot(data=dist_cat, x="distance_category", y="mean", palette="Greens_r", ax=axes[1, 0], edgecolor="black")
    axes[1, 0].axhline(df["delay_target"].mean(), color="red", linestyle="--", label="Average")
    for idx, row in dist_cat.iterrows():
        axes[1, 0].annotate(f"{row['mean']:.1%}\n(N={int(row['count'])})", (idx, row["mean"] + 0.01), ha="center", fontsize=9)
    axes[1, 0].set_title("Observed Delay Rate by Distance Category", fontweight="bold")
    axes[1, 0].set_xlabel("Haul Category")
    axes[1, 0].set_ylabel("Delay Rate")
    axes[1, 0].set_ylim(0, 0.45)
    axes[1, 0].legend()

    # 4. Prior Route Frequency by Target
    sns.boxplot(data=df, x="delay_target", y="prior_route_frequency", palette=["#4575b4", "#d73027"], ax=axes[1, 1])
    axes[1, 1].set_xticklabels(["On-Time (0)", "Delayed (1)"])
    axes[1, 1].set_title("Prior Route Frequency by Delay Status", fontweight="bold")
    axes[1, 1].set_xlabel("Delay Target")
    axes[1, 1].set_ylabel("Prior Frequency Count")

    plt.suptitle("Phase 5 EDA — Continuous Feature Distributions & Target Group Comparisons", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = output_dir / "phase5_numerical_distributions.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Figure 8: Correlation Matrix of Key Features and Target."""
    corr_features = [
        "delay_target",
        "distance",
        "scheduled_dep_time",
        "arr_hour_sin",
        "arr_hour_cos",
        "prior_origin_flight_volume",
        "prior_dest_flight_volume",
        "prior_airline_delay_rate",
        "prior_origin_delay_rate",
        "prior_dest_delay_rate",
        "prior_route_delay_rate",
        "prior_origin_dep_hour_delay_rate",
        "prior_airline_dep_hour_delay_rate",
        "prior_route_frequency",
    ]
    available = [c for c in corr_features if c in df.columns]
    corr_matrix = df[available].corr()

    # Rename for readability
    clean_labels = [c.replace("prior_", "").replace("_delay_rate", "_rate") for c in available]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.6,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation Coefficient (r)"},
    )
    ax.set_title("Correlation Heatmap: Key Predictive Features & Delay Target", fontweight="bold", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    out_path = output_dir / "phase5_correlation_heatmap.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_outlier_boxplots(df: pd.DataFrame, df_op: Optional[pd.DataFrame], output_dir: Path):
    """Figure 9: Numerical Outlier Analysis & Ground-Truth Operational Delays."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Volume Features
    vol_cols = ["prior_origin_flight_volume", "prior_dest_flight_volume", "prior_airline_flight_count"]
    avail_vol = [c for c in vol_cols if c in df.columns]
    vol_data = [df[c].dropna() for c in avail_vol]
    clean_vol_labels = [c.replace("prior_", "").replace("_flight_", " ").replace("_count", " cnt") for c in avail_vol]

    try:
        axes[0].boxplot(vol_data, tick_labels=clean_vol_labels, patch_artist=True, boxprops=dict(facecolor="#9ecae1"))
    except TypeError:
        axes[0].boxplot(vol_data, labels=clean_vol_labels, patch_artist=True, boxprops=dict(facecolor="#9ecae1"))
    axes[0].set_title("Predictive Volume Features (IQR Outlier Assessment)", fontweight="bold")
    axes[0].set_ylabel("Volume Count")
    axes[0].tick_params(axis="x", rotation=15)

    # Subplot 2: Operational Delays (from df_op if present)
    if df_op is not None and "arrival_delay" in df_op.columns and "departure_delay" in df_op.columns:
        delay_data = [df_op["departure_delay"].dropna(), df_op["arrival_delay"].dropna()]
        op_labels = ["Departure Delay (min)", "Arrival Delay (min)"]
        try:
            bp = axes[1].boxplot(delay_data, tick_labels=op_labels, patch_artist=True)
        except TypeError:
            bp = axes[1].boxplot(delay_data, labels=op_labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("#fc9272")
        bp["boxes"][1].set_facecolor("#bcbddc")
        axes[1].axhline(15, color="red", linestyle="--", linewidth=1.5, label="15-min Delay Threshold")
        axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
        axes[1].set_title("Ground-Truth Delay Durations (Operational Data Only)", fontweight="bold")
        axes[1].set_ylabel("Minutes")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Operational records not loaded", ha="center", va="center")

    plt.suptitle("Phase 5 EDA — Numerical Feature Outlier Inspection & Operational Delays", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "phase5_outlier_boxplots.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_all_figures():
    """Main function to generate and save all 9 Phase 5 figures."""
    set_plotting_style()
    config = get_config()
    output_dir = config.figures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = EDAAnalyzer(config=config)
    df_p3 = analyzer.load_data()
    df_op = analyzer.df_op

    print("\n" + "=" * 70)
    print("PHASE 5 EDA: GENERATING PUBLICATION-GRADE VISUALIZATIONS")
    print("=" * 70)

    plot_target_distribution(df_p3, output_dir)
    plot_temporal_patterns(df_p3, output_dir)
    plot_airline_performance(df_p3, output_dir)
    plot_airport_hotspots(df_p3, output_dir)
    plot_route_analysis(df_p3, output_dir)
    plot_historical_features(df_p3, output_dir)
    plot_numerical_distributions(df_p3, output_dir)
    plot_correlation_heatmap(df_p3, output_dir)
    plot_outlier_boxplots(df_p3, df_op, output_dir)

    print("\nAll 9 figures generated successfully in:", output_dir)


if __name__ == "__main__":
    generate_all_figures()
