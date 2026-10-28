import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = str(DATA_DIR / "features_temporal.csv")
OUTPUT_FILE = str(DATA_DIR / "features_selected.csv")
REPORT_FILE = str(DATA_DIR / "temporal_feature_selection_report.csv")


# ============================================================
# METADATA
# ============================================================

METADATA_COLUMNS = [
    "test_id",
    "bearing_id",
    "timestamp",
    "source_file",
]


# ============================================================
# TEMPORAL FEATURE RULES
# ============================================================

# These rate features showed extreme instability and are
# therefore excluded from the first ML dataset.
REMOVE_RATE_BASES = {
    "kurtosis_mean",
    "kurtosis_max",
    "envelope_kurtosis_mean",
    "envelope_kurtosis_max",
    "bpfi_local_ratio_mean",
    "bpfi_local_ratio_max",
    "bpfo_local_ratio_mean",
    "bpfo_local_ratio_max",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_temporal_feature(column):

    if column == "time_since_previous_hours":
        return "time_gap"

    if column.endswith("_delta"):
        return "delta"

    if column.endswith("_rate_per_hour"):
        return "rate"

    if "_rolling_mean_1h" in column:
        return "rolling_mean_1h"

    if "_rolling_std_1h" in column:
        return "rolling_std_1h"

    if "_rolling_mean_6h" in column:
        return "rolling_mean_6h"

    if "_rolling_std_6h" in column:
        return "rolling_std_6h"

    return "unknown"


def get_base_feature(column):

    suffixes = [
        "_delta",
        "_rate_per_hour",
        "_rolling_mean_1h",
        "_rolling_std_1h",
        "_rolling_mean_6h",
        "_rolling_std_6h",
    ]

    for suffix in suffixes:

        if column.endswith(suffix):

            return column[:-len(suffix)]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("TEMPORAL FEATURE SELECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # READ DATA
    # --------------------------------------------------------

    print(f"\nReading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input shape: {df.shape}")

    # --------------------------------------------------------
    # IDENTIFY FEATURES
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    temporal_columns = [
        column
        for column in feature_columns
        if (
            "_delta" in column
            or "_rate_per_hour" in column
            or "_rolling_" in column
            or column == "time_since_previous_hours"
        )
    ]

    original_columns = [
        column
        for column in feature_columns
        if column not in temporal_columns
    ]

    print(f"Original features: {len(original_columns)}")
    print(f"Temporal features: {len(temporal_columns)}")

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    selected_temporal = []
    removed_temporal = []

    for column in temporal_columns:

        feature_type = classify_temporal_feature(column)

        base_feature = get_base_feature(column)

        reason = "retained"

        # --------------------------------------------
        # Remove unstable rate features
        # --------------------------------------------

        if (
            feature_type == "rate"
            and base_feature in REMOVE_RATE_BASES
        ):

            reason = (
                "removed: unstable rate feature"
            )

            removed_temporal.append(
                (column, feature_type, base_feature, reason)
            )

            continue

        # --------------------------------------------
        # Keep everything else for now
        # --------------------------------------------

        selected_temporal.append(column)

    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    selected_columns = (
        METADATA_COLUMNS
        + original_columns
        + selected_temporal
    )

    selected_df = df[selected_columns].copy()

    # --------------------------------------------------------
    # BUILD REPORT
    # --------------------------------------------------------

    report_rows = []

    # Original features

    for column in original_columns:

        report_rows.append({
            "feature": column,
            "feature_group": "original",
            "base_feature": "",
            "status": "retained",
            "reason": "previously refined feature",
        })

    # Selected temporal features

    for column in selected_temporal:

        feature_type = classify_temporal_feature(column)
        base_feature = get_base_feature(column)

        report_rows.append({
            "feature": column,
            "feature_group": "temporal",
            "base_feature": base_feature,
            "status": "retained",
            "reason": "retained for further ML evaluation",
        })

    # Removed temporal features

    for (
        column,
        feature_type,
        base_feature,
        reason
    ) in removed_temporal:

        report_rows.append({
            "feature": column,
            "feature_group": "temporal",
            "base_feature": base_feature,
            "status": "removed",
            "reason": reason,
        })

    report_df = pd.DataFrame(report_rows)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    selected_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    report_df.to_csv(
        REPORT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)

    print(
        f"\nOriginal features retained : "
        f"{len(original_columns)}"
    )

    print(
        f"Temporal features before   : "
        f"{len(temporal_columns)}"
    )

    print(
        f"Temporal features retained : "
        f"{len(selected_temporal)}"
    )

    print(
        f"Temporal features removed  : "
        f"{len(removed_temporal)}"
    )

    print(
        f"\nTotal features retained: "
        f"{len(original_columns) + len(selected_temporal)}"
    )

    print(
        f"Final dataset shape: "
        f"{selected_df.shape}"
    )

    print("\nRemoved temporal features:")

    for (
        column,
        feature_type,
        base_feature,
        reason
    ) in removed_temporal:

        print(
            f"  - {column}"
        )

    print("\nSaved:")
    print(f"  {OUTPUT_FILE}")
    print(f"  {REPORT_FILE}")

    elapsed = time.time() - start_time

    print(
        f"\nTime taken: "
        f"{elapsed:.2f} seconds"
    )

    print("\n" + "=" * 70)
    print("SELECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()