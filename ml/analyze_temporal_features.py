import os
import time
import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/processed/features_temporal.csv"

OUTPUT_DIR = "data/processed/temporal_analysis"

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "temporal_feature_summary.csv"
)

RATE_FILE = os.path.join(
    OUTPUT_DIR,
    "temporal_rate_analysis.csv"
)

CORRELATION_FILE = os.path.join(
    OUTPUT_DIR,
    "temporal_high_correlations.csv"
)

BASE_CORRELATION_FILE = os.path.join(
    OUTPUT_DIR,
    "temporal_vs_base_correlations.csv"
)

ROLLING_CORRELATION_FILE = os.path.join(
    OUTPUT_DIR,
    "rolling_window_correlations.csv"
)


# ============================================================
# METADATA COLUMNS
# ============================================================

METADATA_COLUMNS = [
    "test_id",
    "bearing_id",
    "timestamp",
    "source_file",
]

GROUP_COLS = [
    "test_id",
    "bearing_id",
]

TIMESTAMP_COL = "timestamp"


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

CORRELATION_THRESHOLD = 0.95
VERY_HIGH_CORRELATION = 0.99


# ============================================================
# HELPER: FIND TEMPORAL FEATURES
# ============================================================

def get_temporal_features(feature_columns):

    temporal_features = []

    for column in feature_columns:

        if (
            "_delta" in column
            or "_rate_per_hour" in column
            or "_rolling_" in column
            or column == "time_since_previous_hours"
        ):
            temporal_features.append(column)

    return temporal_features


# ============================================================
# HELPER: FIND BASE FEATURES
# ============================================================

def get_base_feature(temporal_feature):

    suffixes = [
        "_delta",
        "_rate_per_hour",
        "_rolling_mean_1h",
        "_rolling_std_1h",
        "_rolling_mean_6h",
        "_rolling_std_6h",
    ]

    for suffix in suffixes:

        if temporal_feature.endswith(suffix):

            return temporal_feature[
                :-len(suffix)
            ]

    return None


# ============================================================
# HELPER: FIND ROLLING BASE FEATURES
# ============================================================

def get_rolling_base_features(temporal_columns):

    bases = set()

    rolling_suffixes = [
        "_rolling_mean_1h",
        "_rolling_std_1h",
        "_rolling_mean_6h",
        "_rolling_std_6h",
    ]

    for column in temporal_columns:

        for suffix in rolling_suffixes:

            if column.endswith(suffix):

                base = column[
                    :-len(suffix)
                ]

                bases.add(base)

                break

    return sorted(bases)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("TEMPORAL FEATURE ANALYSIS")
    print("=" * 70)

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # READ DATA
    # ========================================================

    print(
        f"\nReading: {INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # ========================================================
    # IDENTIFY FEATURES
    # ========================================================

    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    temporal_columns = get_temporal_features(
        feature_columns
    )

    original_columns = [
        column
        for column in feature_columns
        if column not in temporal_columns
    ]

    print(
        f"Original features: {len(original_columns)}"
    )

    print(
        f"Temporal features: {len(temporal_columns)}"
    )

    # ========================================================
    # 1. TEMPORAL FEATURE SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("1. TEMPORAL FEATURE SUMMARY")
    print("=" * 70)

    summary_rows = []

    for column in temporal_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        absolute = series.abs()

        summary_rows.append(
            {
                "feature": column,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "abs_median": absolute.median(),
                "abs_95th": absolute.quantile(0.95),
                "abs_99th": absolute.quantile(0.99),
                "zero_fraction": (
                    (series == 0).mean()
                ),
                "unique_values": (
                    series.nunique()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        "abs_99th",
        ascending=False
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"Saved: {SUMMARY_FILE}"
    )

    print(
        "\n20 temporal features with "
        "largest 99th-percentile absolute values:"
    )

    print(
        summary_df[
            [
                "feature",
                "abs_99th",
                "min",
                "max",
                "zero_fraction",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # ========================================================
    # 2. RATE-OF-CHANGE ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("2. RATE-OF-CHANGE ANALYSIS")
    print("=" * 70)

    rate_columns = [
        column
        for column in feature_columns
        if "_rate_per_hour" in column
    ]

    rate_rows = []

    for column in rate_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        absolute = series.abs()

        rate_rows.append(
            {
                "feature": column,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "abs_95th": absolute.quantile(0.95),
                "abs_99th": absolute.quantile(0.99),
                "abs_max": absolute.max(),
                "zero_fraction": (
                    (series == 0).mean()
                ),
            }
        )

    rate_df = pd.DataFrame(
        rate_rows
    )

    rate_df = rate_df.sort_values(
        "abs_99th",
        ascending=False
    )

    rate_df.to_csv(
        RATE_FILE,
        index=False
    )

    print(
        f"Rate features analyzed: "
        f"{len(rate_columns)}"
    )

    print(
        f"Saved: {RATE_FILE}"
    )

    print(
        "\n20 rate features with "
        "largest 99th-percentile absolute values:"
    )

    print(
        rate_df.head(20).to_string(
            index=False
        )
    )

    # ========================================================
    # 3. FEATURE-TO-FEATURE CORRELATION
    # ========================================================

    print("\n" + "=" * 70)
    print("3. HIGH-CORRELATION ANALYSIS")
    print("=" * 70)

    print(
        "\nCalculating correlation matrix..."
    )

    numeric_features = df[
        feature_columns
    ].select_dtypes(
        include=np.number
    )

    correlation_matrix = (
        numeric_features
        .corr()
        .abs()
    )

    # --------------------------------------------------------
    # Extract each pair only once.
    # --------------------------------------------------------

    upper_triangle = np.triu(
        np.ones(
            correlation_matrix.shape,
            dtype=bool
        ),
        k=1
    )

    correlation_pairs = (
        correlation_matrix
        .where(upper_triangle)
        .stack()
        .reset_index()
    )

    correlation_pairs.columns = [
        "feature_1",
        "feature_2",
        "absolute_correlation",
    ]

    high_correlation_pairs = (
        correlation_pairs[
            correlation_pairs[
                "absolute_correlation"
            ] >= CORRELATION_THRESHOLD
        ]
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .reset_index(drop=True)
    )

    high_correlation_pairs.to_csv(
        CORRELATION_FILE,
        index=False
    )

    very_high_count = (
        high_correlation_pairs[
            high_correlation_pairs[
                "absolute_correlation"
            ] >= VERY_HIGH_CORRELATION
        ]
        .shape[0]
    )

    print(
        f"\nPairs with |correlation| >= "
        f"{CORRELATION_THRESHOLD}: "
        f"{len(high_correlation_pairs)}"
    )

    print(
        f"Pairs with |correlation| >= "
        f"{VERY_HIGH_CORRELATION}: "
        f"{very_high_count}"
    )

    print(
        f"Saved: {CORRELATION_FILE}"
    )

    print(
        "\nTop 50 highly correlated pairs:"
    )

    if len(high_correlation_pairs) > 0:

        print(
            high_correlation_pairs
            .head(50)
            .to_string(index=False)
        )

    else:

        print(
            "No highly correlated pairs found."
        )

    # ========================================================
    # 4. TEMPORAL FEATURE vs BASE FEATURE
    # ========================================================

    print("\n" + "=" * 70)
    print("4. TEMPORAL vs BASE FEATURE CORRELATION")
    print("=" * 70)

    base_relationships = []

    for temporal_feature in temporal_columns:

        base_feature = get_base_feature(
            temporal_feature
        )

        if (
            base_feature is not None
            and base_feature in df.columns
        ):

            correlation = (
                df[temporal_feature]
                .corr(
                    df[base_feature]
                )
            )

            base_relationships.append(
                {
                    "temporal_feature": (
                        temporal_feature
                    ),
                    "base_feature": (
                        base_feature
                    ),
                    "correlation": (
                        correlation
                    ),
                    "absolute_correlation": (
                        abs(correlation)
                    ),
                }
            )

    base_relationship_df = pd.DataFrame(
        base_relationships
    )

    base_relationship_df = (
        base_relationship_df
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
    )

    base_relationship_df.to_csv(
        BASE_CORRELATION_FILE,
        index=False
    )

    print(
        f"Saved: {BASE_CORRELATION_FILE}"
    )

    print(
        "\n30 strongest temporal/base "
        "relationships:"
    )

    print(
        base_relationship_df
        .head(30)
        .to_string(index=False)
    )

    # ========================================================
    # 5. 1-HOUR vs 6-HOUR ROLLING CORRELATION
    # ========================================================

    print("\n" + "=" * 70)
    print("5. 1-HOUR vs 6-HOUR ROLLING REDUNDANCY")
    print("=" * 70)

    rolling_bases = (
        get_rolling_base_features(
            temporal_columns
        )
    )

    rolling_rows = []

    for base_feature in rolling_bases:

        # ----------------------------------------------------
        # Rolling means
        # ----------------------------------------------------

        mean_1h = (
            f"{base_feature}_rolling_mean_1h"
        )

        mean_6h = (
            f"{base_feature}_rolling_mean_6h"
        )

        if (
            mean_1h in df.columns
            and mean_6h in df.columns
        ):

            correlation = (
                df[mean_1h]
                .corr(
                    df[mean_6h]
                )
            )

            rolling_rows.append(
                {
                    "base_feature": base_feature,
                    "feature_type": "rolling_mean",
                    "correlation_1h_vs_6h": (
                        correlation
                    ),
                    "absolute_correlation": (
                        abs(correlation)
                    ),
                }
            )

        # ----------------------------------------------------
        # Rolling standard deviations
        # ----------------------------------------------------

        std_1h = (
            f"{base_feature}_rolling_std_1h"
        )

        std_6h = (
            f"{base_feature}_rolling_std_6h"
        )

        if (
            std_1h in df.columns
            and std_6h in df.columns
        ):

            correlation = (
                df[std_1h]
                .corr(
                    df[std_6h]
                )
            )

            rolling_rows.append(
                {
                    "base_feature": base_feature,
                    "feature_type": "rolling_std",
                    "correlation_1h_vs_6h": (
                        correlation
                    ),
                    "absolute_correlation": (
                        abs(correlation)
                    ),
                }
            )

    rolling_df = pd.DataFrame(
        rolling_rows
    )

    rolling_df = rolling_df.sort_values(
        "absolute_correlation",
        ascending=False
    )

    rolling_df.to_csv(
        ROLLING_CORRELATION_FILE,
        index=False
    )

    print(
        f"Saved: {ROLLING_CORRELATION_FILE}"
    )

    print(
        "\nRolling-window correlations:"
    )

    print(
        rolling_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 6. TIME GAP ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("6. TIME GAP ANALYSIS")
    print("=" * 70)

    time_gap = pd.to_numeric(
        df["time_since_previous_hours"],
        errors="coerce"
    )

    print(
        "\nTime gap statistics:"
    )

    print(
        f"  Minimum : "
        f"{time_gap.min():.6f} hours"
    )

    print(
        f"  Median  : "
        f"{time_gap.median():.6f} hours"
    )

    print(
        f"  Mean    : "
        f"{time_gap.mean():.6f} hours"
    )

    print(
        f"  95th %  : "
        f"{time_gap.quantile(0.95):.6f} hours"
    )

    print(
        f"  99th %  : "
        f"{time_gap.quantile(0.99):.6f} hours"
    )

    print(
        f"  Maximum : "
        f"{time_gap.max():.6f} hours"
    )

    # ========================================================
    # 7. LARGE TIME GAPS
    # ========================================================

    print(
        "\n20 largest time gaps:"
    )

    largest_gaps = (
        df[
            GROUP_COLS
            + [
                TIMESTAMP_COL,
                "time_since_previous_hours",
            ]
        ]
        .sort_values(
            "time_since_previous_hours",
            ascending=False
        )
        .head(20)
    )

    print(
        largest_gaps.to_string(
            index=False
        )
    )

    # ========================================================
    # FINISH
    # ========================================================

    elapsed_time = (
        time.time() - start_time
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Time taken: "
        f"{elapsed_time:.2f} seconds"
    )

    print(
        f"\nAnalysis files saved in:"
        f"\n{OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()