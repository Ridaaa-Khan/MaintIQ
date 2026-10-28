"""
analyze_final_features.py
=========================
Reports low-variance, duplicate, and highly-correlated features
in the selected feature set.

LEAKAGE WARNING
---------------
This script must be run on the TRAINING partition only, not on the
full dataset.  Running variance / correlation analysis on the full
dataset (including val and test rows) leaks distributional information
from val/test into the feature selection decision, invalidating any
held-out evaluation.

Correct workflow::

    # 1. Split first (creates features_train.csv etc.)
    python select_model_features.py

    # 2. Run this analysis ONLY on the training partition
    #    by setting INPUT_FILE = "data/processed/features_train.csv"

The default INPUT_FILE below points to features_selected.csv for
historical compatibility.  Change it to features_train.csv before
making any decisions that will affect model training.
"""
import os
import time
import itertools

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# WARNING: change this to features_train.csv once select_model_features.py
# has been run.  Using the full dataset introduces selection leakage.
INPUT_FILE = "data/processed/features_train.csv"

OUTPUT_DIR = "data/processed/final_feature_analysis"

METADATA_COLUMNS = [
    "test_id",
    "bearing_id",
    "timestamp",
    "source_file",
]

# Features with variance below this value will be reported.
# This does NOT mean they will automatically be removed.
VARIANCE_THRESHOLD = 1e-10

# Correlation thresholds for redundancy analysis.
HIGH_CORRELATION_THRESHOLD = 0.95
VERY_HIGH_CORRELATION_THRESHOLD = 0.99


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def get_feature_columns(df):
    """
    Return only ML feature columns.
    Metadata columns are excluded.
    """
    return [
        col
        for col in df.columns
        if col not in METADATA_COLUMNS
    ]


def get_temporal_features(feature_columns):
    """
    Identify temporal features based on their naming convention.
    """
    temporal_features = []

    for col in feature_columns:
        if (
            "_delta" in col
            or "_rate_per_hour" in col
            or "_rolling_" in col
            or col == "time_since_previous_hours"
        ):
            temporal_features.append(col)

    return temporal_features


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("FINAL FEATURE QUALITY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    feature_columns = get_feature_columns(df)

    print(f"Dataset shape: {df.shape}")
    print(f"Numeric features: {len(feature_columns)}")

    # Ensure feature columns are numeric
    numeric_features = df[feature_columns].select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print(f"Numeric features detected: {len(numeric_features)}")

    # --------------------------------------------------------
    # 1. BASIC FEATURE QUALITY
    # --------------------------------------------------------

    print_section("1. BASIC FEATURE QUALITY")

    missing_counts = df[numeric_features].isna().sum()

    missing_features = missing_counts[
        missing_counts > 0
    ]

    infinite_mask = np.isinf(
        df[numeric_features].to_numpy()
    )

    infinite_counts = pd.Series(
        infinite_mask.sum(axis=0),
        index=numeric_features
    )

    infinite_features = infinite_counts[
        infinite_counts > 0
    ]

    print(
        f"Features with missing values: "
        f"{len(missing_features)}"
    )

    print(
        f"Features with infinite values: "
        f"{len(infinite_features)}"
    )

    # --------------------------------------------------------
    # 2. LOW-VARIANCE FEATURES
    # --------------------------------------------------------

    print_section("2. LOW-VARIANCE FEATURES")

    variances = df[numeric_features].var(
        ddof=0
    )

    low_variance = variances[
        variances <= VARIANCE_THRESHOLD
    ].sort_values()

    low_variance_df = pd.DataFrame({
        "feature": low_variance.index,
        "variance": low_variance.values
    })

    low_variance_path = os.path.join(
        OUTPUT_DIR,
        "low_variance_features.csv"
    )

    low_variance_df.to_csv(
        low_variance_path,
        index=False
    )

    print(
        f"Low-variance features: "
        f"{len(low_variance_df)}"
    )

    if len(low_variance_df) > 0:
        print()
        print(low_variance_df.to_string(index=False))
    else:
        print("No low-variance features found.")

    print(
        f"Saved: {low_variance_path}"
    )

    # --------------------------------------------------------
    # 3. EXACT DUPLICATE FEATURES
    # --------------------------------------------------------

    print_section("3. EXACT DUPLICATE FEATURES")

    duplicate_pairs = []

    # Compare columns efficiently using pandas.
    # Only compare unique feature pairs.

    for col1, col2 in itertools.combinations(
        numeric_features,
        2
    ):

        if df[col1].equals(df[col2]):

            duplicate_pairs.append({
                "feature_1": col1,
                "feature_2": col2
            })

    duplicate_df = pd.DataFrame(
        duplicate_pairs,
        columns=[
            "feature_1",
            "feature_2"
        ]
    )

    duplicate_path = os.path.join(
        OUTPUT_DIR,
        "duplicate_features.csv"
    )

    duplicate_df.to_csv(
        duplicate_path,
        index=False
    )

    print(
        f"Exact duplicate feature pairs: "
        f"{len(duplicate_df)}"
    )

    if len(duplicate_df) > 0:
        print()
        print(duplicate_df.to_string(index=False))
    else:
        print("No exact duplicate feature pairs found.")

    print(
        f"Saved: {duplicate_path}"
    )

    # --------------------------------------------------------
    # 4. HIGH CORRELATION ANALYSIS
    # --------------------------------------------------------

    print_section("4. HIGH CORRELATION ANALYSIS")

    print("Calculating correlation matrix...")

    correlation_matrix = df[numeric_features].corr()

    high_correlation_pairs = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            col1 = columns[i]
            col2 = columns[j]

            correlation = correlation_matrix.iloc[i, j]

            if pd.isna(correlation):
                continue

            absolute_correlation = abs(correlation)

            if (
                absolute_correlation
                >= HIGH_CORRELATION_THRESHOLD
            ):

                high_correlation_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": correlation,
                    "absolute_correlation": absolute_correlation
                })

    high_correlation_df = pd.DataFrame(
        high_correlation_pairs
    )

    if not high_correlation_df.empty:

        high_correlation_df = (
            high_correlation_df
            .sort_values(
                "absolute_correlation",
                ascending=False
            )
            .reset_index(drop=True)
        )

    high_correlation_path = os.path.join(
        OUTPUT_DIR,
        "high_correlation_pairs.csv"
    )

    high_correlation_df.to_csv(
        high_correlation_path,
        index=False
    )

    very_high_count = 0

    if not high_correlation_df.empty:
        very_high_count = (
            high_correlation_df[
                "absolute_correlation"
            ]
            >= VERY_HIGH_CORRELATION_THRESHOLD
        ).sum()

    print(
        f"Pairs with |correlation| >= "
        f"{HIGH_CORRELATION_THRESHOLD}: "
        f"{len(high_correlation_df)}"
    )

    print(
        f"Pairs with |correlation| >= "
        f"{VERY_HIGH_CORRELATION_THRESHOLD}: "
        f"{very_high_count}"
    )

    if not high_correlation_df.empty:

        print()
        print("Top 30 highly correlated pairs:")

        print(
            high_correlation_df
            .head(30)
            .to_string(index=False)
        )

    print(
        f"Saved: {high_correlation_path}"
    )

    # --------------------------------------------------------
    # 5. TEMPORAL FEATURE QUALITY
    # --------------------------------------------------------

    print_section("5. TEMPORAL FEATURE QUALITY")

    temporal_features = get_temporal_features(
        numeric_features
    )

    print(
        f"Temporal features detected: "
        f"{len(temporal_features)}"
    )

    temporal_analysis = []

    for feature in temporal_features:

        series = df[feature]

        finite_series = series[
            np.isfinite(series)
        ]

        if len(finite_series) == 0:

            temporal_analysis.append({
                "feature": feature,
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
                "p01": np.nan,
                "p99": np.nan,
                "abs_max": np.nan
            })

            continue

        temporal_analysis.append({
            "feature": feature,
            "count": len(finite_series),
            "mean": finite_series.mean(),
            "std": finite_series.std(),
            "min": finite_series.min(),
            "max": finite_series.max(),
            "median": finite_series.median(),
            "p01": finite_series.quantile(0.01),
            "p99": finite_series.quantile(0.99),
            "abs_max": finite_series.abs().max()
        })

    temporal_quality_df = pd.DataFrame(
        temporal_analysis
    )

    if not temporal_quality_df.empty:

        temporal_quality_df = (
            temporal_quality_df
            .sort_values(
                "abs_max",
                ascending=False
            )
            .reset_index(drop=True)
        )

    temporal_quality_path = os.path.join(
        OUTPUT_DIR,
        "temporal_quality_analysis.csv"
    )

    temporal_quality_df.to_csv(
        temporal_quality_path,
        index=False
    )

    print(
        f"Temporal features analyzed: "
        f"{len(temporal_quality_df)}"
    )

    print()
    print(
        "Top 20 temporal features by absolute maximum:"
    )

    if not temporal_quality_df.empty:

        print(
            temporal_quality_df
            .head(20)
            .to_string(index=False)
        )

    print(
        f"Saved: {temporal_quality_path}"
    )

    # --------------------------------------------------------
    # 6. FEATURE STATISTICS
    # --------------------------------------------------------

    print_section("6. FEATURE STATISTICS")

    feature_statistics = []

    for feature in numeric_features:

        series = df[feature]

        finite_series = series[
            np.isfinite(series)
        ]

        if len(finite_series) == 0:
            continue

        feature_statistics.append({
            "feature": feature,
            "mean": finite_series.mean(),
            "std": finite_series.std(),
            "variance": finite_series.var(ddof=0),
            "min": finite_series.min(),
            "max": finite_series.max(),
            "median": finite_series.median(),
            "p01": finite_series.quantile(0.01),
            "p99": finite_series.quantile(0.99),
        })

    feature_statistics_df = pd.DataFrame(
        feature_statistics
    )

    # --------------------------------------------------------
    # 7. SUMMARY
    # --------------------------------------------------------

    print_section("7. ANALYSIS SUMMARY")

    summary = {
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "numeric_features": len(numeric_features),

        "features_with_missing_values":
            len(missing_features),

        "features_with_infinite_values":
            len(infinite_features),

        "low_variance_features":
            len(low_variance_df),

        "exact_duplicate_pairs":
            len(duplicate_df),

        "high_correlation_pairs_0.95":
            len(high_correlation_df),

        "very_high_correlation_pairs_0.99":
            int(very_high_count),

        "temporal_features":
            len(temporal_features),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "feature_analysis_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Dataset rows: "
        f"{summary['dataset_rows']}"
    )

    print(
        f"Dataset columns: "
        f"{summary['dataset_columns']}"
    )

    print(
        f"Numeric features: "
        f"{summary['numeric_features']}"
    )

    print(
        f"Missing-value features: "
        f"{summary['features_with_missing_values']}"
    )

    print(
        f"Infinite-value features: "
        f"{summary['features_with_infinite_values']}"
    )

    print(
        f"Low-variance features: "
        f"{summary['low_variance_features']}"
    )

    print(
        f"Exact duplicate pairs: "
        f"{summary['exact_duplicate_pairs']}"
    )

    print(
        f"High-correlation pairs (>= 0.95): "
        f"{summary['high_correlation_pairs_0.95']}"
    )

    print(
        f"Very-high-correlation pairs (>= 0.99): "
        f"{summary['very_high_correlation_pairs_0.99']}"
    )

    print(
        f"Temporal features: "
        f"{summary['temporal_features']}"
    )

    print(
        f"Saved: {summary_path}"
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("FINAL FEATURE QUALITY ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Time taken: {elapsed:.2f} seconds"
    )

    print()
    print("Output files:")

    print(
        f"  1. {high_correlation_path}"
    )

    print(
        f"  2. {low_variance_path}"
    )

    print(
        f"  3. {duplicate_path}"
    )

    print(
        f"  4. {temporal_quality_path}"
    )

    print(
        f"  5. {summary_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()