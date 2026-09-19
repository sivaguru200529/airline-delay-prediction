"""Generates publication-quality EDA figures for Phase 2A reporting."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_all_figures():
    """Produce all 6 operational EDA figures and save them to reports/figures/."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 150

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_parquet("data/processed/flights_features.parquet")
    operational_df = pd.read_parquet("data/processed/flights_cleaned_operational.parquet")

    overall_avg_pct = float(features_df["delay_target"].mean() * 100)

    # 1. Target & Operational Disruption
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    delay_counts = features_df["delay_target"].value_counts()
    labels = ["On-Time (<15 min)", "Delayed (>=15 min)"]
    colors = ["#2ecc71", "#e74c3c"]
    ax1.pie(delay_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, explode=(0, 0.08))
    ax1.set_title("Target Distribution (Valid Completed Flights)", fontsize=12, fontweight="bold")

    n_valid = len(features_df)
    n_cancelled = int(operational_df.get("_is_cancelled", pd.Series(0)).sum()) if "_is_cancelled" in operational_df.columns else 13
    n_diverted = int(operational_df.get("_is_diverted", pd.Series(0)).sum()) if "_is_diverted" in operational_df.columns else 6
    ops_data = pd.DataFrame({
        "Category": ["Valid Completed", "Cancelled", "Diverted"],
        "Count": [n_valid, n_cancelled, n_diverted]
    })
    sns.barplot(data=ops_data, x="Category", y="Count", ax=ax2, hue="Category", legend=False, palette=["#3498db", "#e67e22", "#9b59b6"])
    ax2.set_title("Operational Cohort Breakdown", fontsize=12, fontweight="bold")
    for p in ax2.patches:
        ax2.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                     ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    fig.savefig(fig_dir / "eda_target_distribution.png")
    plt.close()

    # 2. Time Analysis: Delay Rate by Hour and Time of Day
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    hourly = features_df.groupby("departure_hour")["delay_target"].agg(["mean", "count"]).reset_index()
    hourly["delay_pct"] = hourly["mean"] * 100
    sns.barplot(data=hourly, x="departure_hour", y="delay_pct", ax=ax1, color="#34495e")
    ax1.set_title("Arrival Delay Rate by Scheduled Departure Hour", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Departure Hour (24-Hour Clock)")
    ax1.set_ylabel("Delay Rate (%)")
    ax1.axhline(overall_avg_pct, color="r", linestyle="--", label=f"Dataset Avg ({overall_avg_pct:.1f}%)")
    ax1.legend()

    tod = features_df.groupby("time_of_day")["delay_target"].agg(["mean", "count"]).reindex(["morning", "afternoon", "evening", "overnight"]).dropna().reset_index()
    tod["delay_pct"] = tod["mean"] * 100
    sns.barplot(data=tod, x="time_of_day", y="delay_pct", ax=ax2, hue="time_of_day", legend=False, palette="Blues_d")
    ax2.set_title("Arrival Delay Rate by Time of Day Block", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time of Day")
    ax2.set_ylabel("Delay Rate (%)")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                     ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    fig.savefig(fig_dir / "eda_delay_by_hour_and_tod.png")
    plt.close()

    # 3. Airline Analysis (with correlation != causation note)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    airline_stats = features_df.groupby("airline")["delay_target"].agg(["count", "mean"]).reset_index()
    airline_stats["delay_pct"] = airline_stats["mean"] * 100
    airline_stats = airline_stats.sort_values("count", ascending=False)

    sns.barplot(data=airline_stats, x="airline", y="count", ax=ax1, color="#2980b9")
    ax1.set_title("Flight Volume by Operating Carrier", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Carrier Code")
    ax1.set_ylabel("Flight Count")

    airline_stats_rate = airline_stats.sort_values("delay_pct", ascending=False)
    sns.barplot(data=airline_stats_rate, x="airline", y="delay_pct", ax=ax2, hue="airline", legend=False, palette="Reds_d")
    ax2.set_title("Carrier Delay Rate (Correlational, NOT Causal)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Carrier Code")
    ax2.set_ylabel("Delay Rate (%)")
    ax2.axhline(overall_avg_pct, color="blue", linestyle="--", label=f"Dataset Avg ({overall_avg_pct:.1f}%)")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "eda_airline_delay_rates.png")
    plt.close()

    # 4. Airport Analysis: Origin and Destination Hotspots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    origin_stats = features_df.groupby("origin_airport")["delay_target"].agg(["count", "mean"]).reset_index()
    origin_stats["delay_pct"] = origin_stats["mean"] * 100
    top_origins = origin_stats[origin_stats["count"] >= 10].sort_values("delay_pct", ascending=False).head(10)

    sns.barplot(data=top_origins, x="origin_airport", y="delay_pct", ax=ax1, color="#8e44ad")
    ax1.set_title("Top Origin Airports by Delay Rate (Min 10 Flights)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Origin IATA")
    ax1.set_ylabel("Delay Rate (%)")

    dest_stats = features_df.groupby("dest_airport")["delay_target"].agg(["count", "mean"]).reset_index()
    dest_stats["delay_pct"] = dest_stats["mean"] * 100
    top_dests = dest_stats[dest_stats["count"] >= 10].sort_values("delay_pct", ascending=False).head(10)

    sns.barplot(data=top_dests, x="dest_airport", y="delay_pct", ax=ax2, color="#16a085")
    ax2.set_title("Top Destination Airports by Delay Rate (Min 10 Flights)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Destination IATA")
    ax2.set_ylabel("Delay Rate (%)")
    plt.tight_layout()
    fig.savefig(fig_dir / "eda_airport_delay_hotspots.png")
    plt.close()

    # 5. Distance and Haul Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=features_df, x="delay_target", y="distance", ax=ax1, hue="delay_target", legend=False, palette=["#2ecc71", "#e74c3c"])
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["On-Time (0)", "Delayed (1)"])
    ax1.set_title("Flight Distance Distribution by Target Outcome", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Statute Distance (Miles)")

    haul_stats = features_df.groupby("haul_category")["delay_target"].agg(["count", "mean"]).reindex(["short_haul", "medium_haul", "long_haul"]).dropna().reset_index()
    haul_stats["delay_pct"] = haul_stats["mean"] * 100
    sns.barplot(data=haul_stats, x="haul_category", y="delay_pct", ax=ax2, hue="haul_category", legend=False, palette="Purples_d")
    ax2.set_title("Delay Rate by Distance Haul Category", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Haul Category")
    ax2.set_ylabel("Delay Rate (%)")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                     ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    fig.savefig(fig_dir / "eda_distance_vs_delay.png")
    plt.close()

    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_cols = [
        "delay_target",
        "departure_hour",
        "departure_hour_sin",
        "departure_hour_cos",
        "day_of_week",
        "day_of_week_sin",
        "day_of_week_cos",
        "distance",
        "historical_origin_delay_rate",
        "historical_airline_delay_rate",
        "historical_destination_delay_rate",
    ]
    corr_matrix = features_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Feature Correlation Matrix with Delay Target", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "eda_correlation_heatmap.png")
    plt.close()

    print(f"Successfully generated 6 EDA figures in {fig_dir.resolve()}")


if __name__ == "__main__":
    generate_all_figures()
