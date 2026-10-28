import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_config import load_dataset_spec


# ============================================================
# FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = str(PROJECT_ROOT / "data" / "processed" / "features_refined.csv")
OUTPUT_FILE = str(PROJECT_ROOT / "data" / "processed" / "features_temporal.csv")


# ============================================================
# GROUPING / TIMESTAMP
# ============================================================

GROUP_COLS = ["test_id", "bearing_id"]
TIMESTAMP_COL = "timestamp"


# ============================================================
# GAP MASKING
# ============================================================
#
# Test 1 has an acquisition pause of ~148 h between early and
# late segments.  Rows immediately after a large gap produce
# misleading delta / rate-of-change values: the "change from
# previous" reflects a long dormancy, not real degradation.
#
# Any row where time_since_previous_hours > GAP_THRESHOLD_HOURS
# has delta/rate set to 0 and large_gap_flag = 1.
#
# 2 h is a safe threshold: normal sampling cadence is ~10 min;
# the known Test1 gap is ~148 h.  Adjust if needed.
GAP_THRESHOLD_HOURS = float(
    load_dataset_spec()["temporal"]["gap_threshold_hours"]
)

# ============================================================
# FEATURES FOR TEMPORAL ANALYSIS
# ============================================================
#
# We do NOT create temporal versions of all 90 features.
# These are the features most directly related to vibration
# severity, impulsiveness and bearing-fault behavior.
#

TEMPORAL_FEATURES = [
    # Overall vibration
    "rms_mean",
    "rms_max",

    # Impulsive behavior
    "kurtosis_mean",
    "kurtosis_max",

    # Vibration amplitude range
    "peak_to_peak_mean",
    "peak_to_peak_max",

    # Envelope analysis
    "envelope_rms_mean",
    "envelope_rms_max",

    "envelope_kurtosis_mean",
    "envelope_kurtosis_max",

    # Bearing characteristic-frequency energy
    "bpfi_energy_mean",
    "bpfi_energy_max",

    "bpfo_energy_mean",
    "bpfo_energy_max",

    # BPFI / BPFO strength relative to local spectrum
    "bpfi_local_ratio_mean",
    "bpfi_local_ratio_max",
    "bpfo_local_ratio_mean",
    "bpfo_local_ratio_max",

    # Roller defect: published IMS signature is 2xBSF
    "bsf_2x_energy_mean",
    "bsf_2x_energy_max",
]


# ============================================================
# ROLLING WINDOWS
# ============================================================
#
# The dataset does not have perfectly constant time intervals.
# Therefore we use actual time-based windows instead of
# "last N rows".
#

ROLLING_WINDOWS = {
    "1h": "1h",
    "6h": "6h",
}


# ============================================================
# TEMPORAL FEATURE FUNCTION
# ============================================================

def add_temporal_features(group):
    """
    Add temporal/degradation features to one
    test + bearing sequence.

    Only the current observation and previous observations
    are used.

    No future observations are used.
    """

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    group = group.sort_values(
        TIMESTAMP_COL
    ).copy()

    # --------------------------------------------------------
    # Calculate actual elapsed time between measurements
    # --------------------------------------------------------

    elapsed_hours = (
        group[TIMESTAMP_COL]
        .diff()
        .dt.total_seconds()
        / 3600.0
    )

    group["time_since_previous_hours"] = elapsed_hours

    # --------------------------------------------------------
    # Process each selected feature
    # --------------------------------------------------------

    for feature in TEMPORAL_FEATURES:

        # ====================================================
        # 1. PREVIOUS VALUE
        # ====================================================

        previous = group[feature].shift(1)

        # ====================================================
        # 2. CHANGE FROM PREVIOUS OBSERVATION
        # ====================================================

        group[f"{feature}_delta"] = (
            group[feature] - previous
        )

        # ====================================================
        # 3. RATE OF CHANGE PER HOUR
        # ====================================================

        group[f"{feature}_rate_per_hour"] = np.where(
            elapsed_hours > 0,
            (group[feature] - previous) / elapsed_hours,
            np.nan
        )

        # ====================================================
        # 4. TIME-BASED ROLLING FEATURES
        # ====================================================

        # Create temporary timestamp-indexed series.
        #
        # This allows us to use actual time windows such as
        # 1 hour and 6 hours instead of assuming every row
        # represents the same amount of time.

        temp = pd.DataFrame(
            {
                "timestamp": group[TIMESTAMP_COL].values,
                "value": group[feature].values,
            }
        )

        temp = temp.set_index("timestamp")

        for window_name, window_size in ROLLING_WINDOWS.items():

            rolling = temp["value"].rolling(
                window=window_size,
                min_periods=1
            )

            # ------------------------------------------------
            # Rolling mean
            # ------------------------------------------------

            group[
                f"{feature}_rolling_mean_{window_name}"
            ] = rolling.mean().values

            # ------------------------------------------------
            # Rolling standard deviation
            # ------------------------------------------------

            group[
                f"{feature}_rolling_std_{window_name}"
            ] = (
                rolling
                .std()
                .fillna(0)
                .values
            )

    # --------------------------------------------------------
    # Mask delta / rate columns across large acquisition gaps
    # --------------------------------------------------------
    #
    # Rows immediately following a long pause have meaningless
    # delta / rate values. Zero them and set large_gap_flag so
    # the model sees the gap without a fake degradation spike.
    #

    large_gap_mask = (
        group["time_since_previous_hours"] > GAP_THRESHOLD_HOURS
    ).fillna(False)

    group["large_gap_flag"] = large_gap_mask.astype(int)

    for feature in TEMPORAL_FEATURES:
        delta_col = f"{feature}_delta"
        rate_col = f"{feature}_rate_per_hour"
        if delta_col in group.columns:
            group.loc[large_gap_mask.fillna(False), delta_col] = 0.0
        if rate_col in group.columns:
            group.loc[large_gap_mask.fillna(False), rate_col] = 0.0

    return group


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("TEMPORAL FEATURE ENGINEERING")
    print("=" * 70)

    # ========================================================
    # READ DATA
    # ========================================================

    print(f"\nReading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input shape: {df.shape}")

    # ========================================================
    # VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_columns = (
        GROUP_COLS
        + [TIMESTAMP_COL]
        + TEMPORAL_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # ========================================================
    # CONVERT TIMESTAMP
    # ========================================================

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL]
    )

    # ========================================================
    # SORT ORIGINAL DATA
    # ========================================================

    df = df.sort_values(
        GROUP_COLS + [TIMESTAMP_COL]
    ).reset_index(drop=True)

    # ========================================================
    # COUNT SEQUENCES
    # ========================================================

    total_groups = df.groupby(
        GROUP_COLS
    ).ngroups

    print(
        f"Number of test/bearing sequences: "
        f"{total_groups}"
    )

    # ========================================================
    # PROCESS EACH TEST + BEARING
    # ========================================================

    print("\nCreating temporal features...")

    processed_groups = []

    for group_number, (_, group) in enumerate(
        df.groupby(
            GROUP_COLS,
            sort=False
        ),
        start=1
    ):

        processed_group = add_temporal_features(
            group
        )

        processed_groups.append(
            processed_group
        )

        # Progress
        if (
            group_number % 4 == 0
            or group_number == total_groups
        ):

            print(
                f"Processed "
                f"{group_number}/{total_groups} sequences"
            )

    # ========================================================
    # COMBINE PROCESSED GROUPS
    # ========================================================

    result = pd.concat(
        processed_groups,
        ignore_index=True
    )

    # ========================================================
    # REMOVE FIRST ROW OF EACH SEQUENCE
    # ========================================================
    #
    # The first observation has no previous observation.
    #
    # Therefore:
    #
    # delta = unavailable
    # rate  = unavailable
    #
    # We remove those 12 rows (one from each
    # test + bearing sequence).
    #
    # IMPORTANT:
    # We use cumcount() instead of groupby().apply()
    # so test_id and bearing_id remain in the dataframe.
    #

    sequence_position = (
        result
        .groupby(
            GROUP_COLS,
            sort=False
        )
        .cumcount()
    )

    result = result[
        sequence_position > 0
    ].reset_index(drop=True)

    result["large_gap_flag"] = (
        result["large_gap_flag"].fillna(0).astype(int)
    )

    # ========================================================
    # FINAL SORT
    # ========================================================

    result = result.sort_values(
        GROUP_COLS + [TIMESTAMP_COL]
    ).reset_index(drop=True)

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # ========================================================
    # SAVE
    # ========================================================

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    elapsed_time = time.time() - start_time

    added_columns = [
        column
        for column in result.columns
        if column not in df.columns
    ]

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Input rows:        {len(df):,}"
    )

    print(
        f"Output rows:       {len(result):,}"
    )

    print(
        f"Input columns:     {len(df.columns)}"
    )

    print(
        f"Output columns:    {len(result.columns)}"
    )

    print(
        f"New temporal cols: {len(added_columns)}"
    )

    print(
        f"Output file:       {OUTPUT_FILE}"
    )

    print(
        f"Time taken:        {elapsed_time:.2f} seconds"
    )

    # ========================================================
    # PRINT NEW FEATURES
    # ========================================================

    print("\nTemporal features added:")

    for column in added_columns:

        print(
            f"  - {column}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()