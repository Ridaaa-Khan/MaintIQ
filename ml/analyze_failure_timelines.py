import time

import numpy as np
import pandas as pd

from pipeline_config import (
    ARTIFACT_RMS_THRESHOLD,
    FAILED_BEARINGS,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/features_model_ready.csv"

TIMESTAMP_COL = "timestamp"

# Features that are particularly relevant to bearing degradation.
# Updated to include 2*BSF features and BPFO local ratio features.
KEY_FEATURES = [
    "rms_mean",
    "rms_max",
    "kurtosis_mean",
    "kurtosis_max",
    "peak_to_peak_mean",
    "peak_to_peak_max",
    "envelope_rms_mean",
    "envelope_rms_max",
    "envelope_kurtosis_mean",
    "envelope_kurtosis_max",
    "bpfi_energy_mean",
    "bpfi_energy_max",
    "bpfo_energy_mean",
    "bpfo_energy_max",
    "bsf_energy_mean",
    "bsf_energy_max",
    "bsf_2x_energy_mean",
    "bsf_2x_energy_max",
    "bpfi_local_ratio_mean",
    "bpfi_local_ratio_max",
    "bpfo_local_ratio_mean",
    "bpfo_local_ratio_max",
]


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_percentage_change(early, late):
    if pd.isna(early) or pd.isna(late):
        return np.nan

    if abs(early) < 1e-12:
        return np.nan

    return ((late - early) / abs(early)) * 100


def describe_periods(group, feature):
    """
    Divide a bearing's operational timeline into three chronological
    sections and return the mean feature value for each.
    """
    n = len(group)

    if n < 3:
        return np.nan, np.nan, np.nan

    third = n // 3

    early = group.iloc[:third][feature].mean()

    middle = group.iloc[
        third:2 * third
    ][feature].mean()

    late = group.iloc[
        2 * third:
    ][feature].mean()

    return early, middle, late


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print_header("FAILURE TIMELINE ANALYSIS (WITH ARTIFACT FILTER)")

    # --------------------------------------------------------
    # 1. LOAD DATA ONCE
    # --------------------------------------------------------

    print()
    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[TIMESTAMP_COL]
    )

    print(f"Dataset shape: {df.shape}")

    # Keep only features that actually exist.
    available_features = [
        feature
        for feature in KEY_FEATURES
        if feature in df.columns
    ]

    missing_features = [
        feature
        for feature in KEY_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        print()
        print("WARNING: These features were not found:")
        for feature in missing_features:
            print(f"  {feature}")

    print()
    print(
        f"Key features available: "
        f"{len(available_features)}"
    )

    # --------------------------------------------------------
    # 2. SORT ONCE
    # --------------------------------------------------------

    df.sort_values(
        ["test_id", "bearing_id", TIMESTAMP_COL],
        inplace=True
    )

    # --------------------------------------------------------
    # 3. IDENTIFY & REPORT ARTIFACT ROWS
    # --------------------------------------------------------

    print_header("ARTIFACT FILTERING (rms_mean <= 0.01)")

    if "rms_mean" in df.columns:
        artifact_mask = df["rms_mean"] <= ARTIFACT_RMS_THRESHOLD
    elif "rms_max" in df.columns:
        artifact_mask = df["rms_max"] <= ARTIFACT_RMS_THRESHOLD
    else:
        artifact_mask = pd.Series(False, index=df.index)

    artifact_df = df[artifact_mask]

    print(f"Total artifact rows detected: {len(artifact_df)}")

    if not artifact_df.empty:
        for (test_id, bearing_id), art_group in artifact_df.groupby(["test_id", "bearing_id"]):
            print(f"  - {test_id} / Bearing {bearing_id}: {len(art_group)} artifact files")
            for _, row in art_group.iterrows():
                print(f"      File: {row.get('source_file', 'N/A')}, Timestamp: {row[TIMESTAMP_COL]}, rms_mean: {row.get('rms_mean', np.nan):.6f}")

    # --------------------------------------------------------
    # 4. ANALYZE EACH FAILED BEARING (OPERATIONAL DATA ONLY)
    # --------------------------------------------------------

    all_results = []

    for test_id, bearing_ids in FAILED_BEARINGS.items():

        for bearing_id in bearing_ids:

            print_header(
                f"{test_id.upper()} / BEARING {bearing_id}"
            )

            raw_group = df[
                (df["test_id"] == test_id)
                & (df["bearing_id"] == bearing_id)
            ].copy()

            if raw_group.empty:
                print("ERROR: No data found.")
                continue

            raw_group.sort_values(
                TIMESTAMP_COL,
                inplace=True
            )

            # Filter out artifact rows to compute true operational timeline
            if "rms_mean" in raw_group.columns:
                op_group = raw_group[raw_group["rms_mean"] > ARTIFACT_RMS_THRESHOLD].copy()
            else:
                op_group = raw_group.copy()

            start_time_stamp = op_group[TIMESTAMP_COL].iloc[0]
            failure_endpoint = op_group[TIMESTAMP_COL].iloc[-1]
            raw_end_timestamp = raw_group[TIMESTAMP_COL].iloc[-1]

            duration_hours = (
                failure_endpoint - start_time_stamp
            ).total_seconds() / 3600

            num_artifacts = len(raw_group) - len(op_group)

            print()
            print("TIMELINE & ARTIFACT FILTER SUMMARY")
            print("-" * 70)

            print(f"Total observations in raw dataset : {len(raw_group)}")
            print(f"Artifact observations removed     : {num_artifacts}")
            print(f"Operational observations          : {len(op_group)}")
            print(f"Start timestamp                   : {start_time_stamp}")
            print(f"Raw final timestamp               : {raw_end_timestamp}")
            print(f"Operational failure endpoint      : {failure_endpoint}")
            print(f"Operational duration (hours)      : {duration_hours:.2f}")
            print(f"Operational duration (days)       : {duration_hours / 24:.2f}")

            # ------------------------------------------------
            # Time gaps in operational data
            # ------------------------------------------------

            time_differences = (
                op_group[TIMESTAMP_COL]
                .diff()
                .dt.total_seconds()
                / 3600
            )

            print()
            print("TIMESTAMP GAPS (OPERATIONAL DATA)")
            print("-" * 70)

            print(
                f"Median gap (hours): "
                f"{time_differences.median():.4f}"
            )

            print(
                f"Maximum gap (hours): "
                f"{time_differences.max():.4f}"
            )

            # ------------------------------------------------
            # Feature progression
            # ------------------------------------------------

            print()
            print("FEATURE PROGRESSION (OPERATIONAL DATA ONLY)")
            print("-" * 70)

            print(
                f"{'Feature':<40}"
                f"{'Early':>12}"
                f"{'Middle':>12}"
                f"{'Late':>12}"
                f"{'Late Delta%':>14}"
            )

            print("-" * 90)

            for feature in available_features:

                early, middle, late = describe_periods(
                    op_group,
                    feature
                )

                change = safe_percentage_change(
                    early,
                    late
                )

                print(
                    f"{feature:<40}"
                    f"{early:>12.6g}"
                    f"{middle:>12.6g}"
                    f"{late:>12.6g}"
                    f"{change:>11.2f}%"
                )

            # ------------------------------------------------
            # Late-stage operational values
            # ------------------------------------------------

            print()
            print("LAST 10 OPERATIONAL OBSERVATIONS (BEFORE SHUTDOWN)")
            print("-" * 70)

            columns_to_show = [
                TIMESTAMP_COL
            ] + [
                feature
                for feature in [
                    "rms_mean",
                    "rms_max",
                    "kurtosis_mean",
                    "kurtosis_max",
                    "envelope_rms_mean",
                    "envelope_rms_max",
                    "bpfi_energy_mean",
                    "bpfi_energy_max",
                    "bpfo_energy_mean",
                    "bpfo_energy_max",
                    "bsf_2x_energy_mean",
                    "bsf_2x_energy_max",
                    "bpfi_local_ratio_mean",
                    "bpfo_local_ratio_mean",
                ]
                if feature in op_group.columns
            ]

            print(
                op_group[
                    columns_to_show
                ]
                .tail(10)
                .to_string(index=False)
            )

            # ------------------------------------------------
            # Record summary
            # ------------------------------------------------

            result = {
                "test_id": test_id,
                "bearing_id": bearing_id,
                "raw_observations": len(raw_group),
                "artifacts_filtered": num_artifacts,
                "op_observations": len(op_group),
                "start_timestamp": start_time_stamp,
                "raw_end_timestamp": raw_end_timestamp,
                "failure_endpoint": failure_endpoint,
                "duration_hours": duration_hours,
            }

            for feature in available_features:

                early, middle, late = describe_periods(
                    op_group,
                    feature
                )

                result[f"{feature}_early"] = early
                result[f"{feature}_middle"] = middle
                result[f"{feature}_late"] = late
                result[f"{feature}_late_change_pct"] = (
                    safe_percentage_change(
                        early,
                        late
                    )
                )

            all_results.append(result)

    # --------------------------------------------------------
    # 5. OVERALL FAILURE SUMMARY
    # --------------------------------------------------------

    print_header("OPERATIONAL FAILURE TIMELINE SUMMARY")

    summary_df = pd.DataFrame(all_results)

    if not summary_df.empty:

        display_columns = [
            "test_id",
            "bearing_id",
            "raw_observations",
            "artifacts_filtered",
            "op_observations",
            "start_timestamp",
            "failure_endpoint",
            "duration_hours",
        ]

        print(
            summary_df[
                display_columns
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # 6. SUMMARY & NEXT STEPS
    # --------------------------------------------------------

    print_header("SUMMARY & ARTIFACT FILTERING IMPACT")

    print(
        """
Summary of findings:
1. Test 2 had 2 post-shutdown artifact files (2004.02.19.06.12.39 & 06.22.39) with ~0 RMS.
2. Test 3 had 1 post-shutdown artifact file (2004.04.18.02.42.55) with ~0 RMS.
3. Filtering artifact rows (rms_mean <= 0.01) restores true run-to-failure endpoints:
   - Test 1 B3 & B4: Operational failure endpoint = 2003-11-25 23:39:56
   - Test 2 B1:     Operational failure endpoint = 2004-02-19 06:02:39
   - Test 3 B3:     Operational failure endpoint = 2004-04-18 02:32:55

These non-artifact timestamps represent the authoritative failure endpoints for target creation.
All artifact rows will be excluded from RUL countdowns, classification targets, and anomaly labels.
"""
    )

    print()
    print(
        f"Analysis time: "
        f"{time.time() - start_time:.2f} seconds"
    )

    print()
    print("=" * 70)
    print("FAILURE TIMELINE ANALYSIS COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()