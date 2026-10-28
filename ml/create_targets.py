"""
Target Engineering
==================

Input:
    data/processed/features_model_ready.csv

Outputs:
    data/processed/features_model_ready_with_targets.csv
    data/processed/features_train.csv
    data/processed/features_val.csv
    data/processed/features_test.csv

Targets / masks:
    - is_artifact
    - time_to_operational_endpoint_hours
    - failure_within_24h
    - isolation_forest_train

Important:
    - Starts from the unlabeled model-ready feature dataset.
    - Does not modify the input feature dataset.
    - No rows are deleted.
    - Artifacts remain in the enriched dataset but are excluded from
      supervised targets and Isolation Forest baseline training.
    - RUL-style target exists only for documented failed trajectories.
    - Healthy classification negatives require a fully observed
      24-hour follow-up period.
    - Existing trajectory-level train/val/test assignments are preserved.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import (
    ARTIFACT_RMS_THRESHOLD,
    FAILED_BEARINGS,
    FAILURE_HORIZON_HOURS,
    NORMAL_BASELINE_FRACTION,
    TARGET_COLUMNS,
    TRAJECTORY_SPLIT,
    failed_bearing_set,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "features_model_ready.csv"

OUTPUT_FILE = (
    DATA_DIR / "features_model_ready_with_targets.csv"
)

TRAIN_FILE = DATA_DIR / "features_train.csv"
VAL_FILE = DATA_DIR / "features_val.csv"
TEST_FILE = DATA_DIR / "features_test.csv"


# ============================================================
# Helpers
# ============================================================

def build_failed_bearing_set():
    """Return all documented failed trajectories."""

    return failed_bearing_set()


def validate_input(df):
    """Validate required input structure."""

    required_columns = {
        "test_id",
        "bearing_id",
        "timestamp",
        "rms_mean",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df["timestamp"].isna().any():
        raise ValueError(
            "Input contains missing timestamps."
        )

    if df["rms_mean"].isna().any():
        raise ValueError(
            "Input contains missing rms_mean values."
        )


def sort_and_validate_chronology(df):
    """
    Sort the dataset chronologically within each trajectory and
    validate that timestamps are strictly increasing.

    This makes create_targets() safe even when called directly.
    """

    df = df.sort_values(
        ["test_id", "bearing_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = df.groupby(
        ["test_id", "bearing_id"],
        sort=False,
    )

    for key, group in grouped:

        timestamps = group["timestamp"]

        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"Trajectory {key} is not chronologically sorted."
            )

        if timestamps.duplicated().any():
            raise ValueError(
                f"Trajectory {key} contains duplicate timestamps."
            )

    return df


def get_failure_endpoints(df, artifact_mask):
    """
    Return the last non-artifact timestamp for each documented
    failed trajectory.
    """

    endpoints = {}

    for test_id, bearing_ids in FAILED_BEARINGS.items():

        for bearing_id in bearing_ids:

            mask = (
                (df["test_id"] == test_id)
                & (df["bearing_id"] == bearing_id)
                & (~artifact_mask)
            )

            timestamps = df.loc[
                mask,
                "timestamp",
            ]

            if timestamps.empty:
                raise ValueError(
                    "No operational observations found for "
                    f"failed trajectory "
                    f"({test_id}, {bearing_id})."
                )

            endpoints[
                (test_id, bearing_id)
            ] = timestamps.iloc[-1]

    return endpoints


# ============================================================
# Target generation
# ============================================================

def create_targets(df):
    """
    Create target/mask columns.

    Returns:
        df_with_targets
        failure_endpoints
    """

    # --------------------------------------------------------
    # Ensure chronological ordering here as well.
    # --------------------------------------------------------

    df = sort_and_validate_chronology(df)

    failed_set = build_failed_bearing_set()

    # --------------------------------------------------------
    # Artifact mask
    # --------------------------------------------------------

    artifact_mask = (
        df["rms_mean"] <= ARTIFACT_RMS_THRESHOLD
    )

    # --------------------------------------------------------
    # Allocate target arrays separately.
    #
    # This avoids repeatedly inserting columns into the wide
    # feature DataFrame and prevents fragmentation warnings.
    # --------------------------------------------------------

    is_artifact = artifact_mask.to_numpy(
        dtype=bool
    )

    time_to_endpoint = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )

    failure_within_24h = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )

    isolation_forest_train = np.zeros(
        len(df),
        dtype=bool,
    )

    # --------------------------------------------------------
    # Failure endpoints
    # --------------------------------------------------------

    failure_endpoints = get_failure_endpoints(
        df,
        artifact_mask,
    )

    # --------------------------------------------------------
    # Group trajectories
    # --------------------------------------------------------

    grouped = df.groupby(
        ["test_id", "bearing_id"],
        sort=False,
    )

    for (
        test_id,
        bearing_id,
    ), group in grouped:

        positions = group.index.to_numpy()

        # ----------------------------------------------------
        # Operational observations
        # ----------------------------------------------------

        operational_positions = positions[
            ~is_artifact[positions]
        ]

        if len(operational_positions) == 0:
            continue

        # ----------------------------------------------------
        # Isolation Forest baseline
        #
        # IMPORTANT:
        # Use integer truncation/floor, matching the project's
        # verified expected baseline counts.
        # ----------------------------------------------------

        baseline_count = max(
            1,
            int(
                len(operational_positions)
                * NORMAL_BASELINE_FRACTION
            ),
        )

        baseline_positions = (
            operational_positions[:baseline_count]
        )

        isolation_forest_train[
            baseline_positions
        ] = True

        # ----------------------------------------------------
        # Failed trajectory
        # ----------------------------------------------------

        if (
            test_id,
            bearing_id,
        ) in failed_set:

            endpoint = failure_endpoints[
                (test_id, bearing_id)
            ]

            timestamps = df.loc[
                operational_positions,
                "timestamp",
            ]

            endpoint_delta = (
                endpoint - timestamps
            )

            endpoint_hours = (
                endpoint_delta.dt.total_seconds()
                / 3600.0
            )

            # Numerical safety.
            endpoint_hours = endpoint_hours.clip(
                lower=0.0
            )

            time_to_endpoint[
                operational_positions
            ] = endpoint_hours.to_numpy()

            failure_within_24h[
                operational_positions
            ] = (
                endpoint_hours
                <= FAILURE_HORIZON_HOURS
            ).astype(float).to_numpy()

        # ----------------------------------------------------
        # Healthy trajectory
        # ----------------------------------------------------

        else:

            run_end = df.loc[
                operational_positions,
                "timestamp",
            ].iloc[-1]

            timestamps = df.loc[
                operational_positions,
                "timestamp",
            ]

            observed_followup_hours = (
                (run_end - timestamps).dt.total_seconds() / 3600.0
            )

            valid_negative = (
                observed_followup_hours
                >= FAILURE_HORIZON_HOURS
            )

            valid_positions = (
                operational_positions[
                    valid_negative.to_numpy()
                ]
            )

            failure_within_24h[
                valid_positions
            ] = 0.0

    # --------------------------------------------------------
    # Explicitly force artifacts out of targets/masks.
    # --------------------------------------------------------

    time_to_endpoint[is_artifact] = np.nan
    failure_within_24h[is_artifact] = np.nan
    isolation_forest_train[is_artifact] = False

    # --------------------------------------------------------
    # Add all four columns at once.
    # --------------------------------------------------------

    target_df = pd.DataFrame(
        {
            "is_artifact": is_artifact,
            "time_to_operational_endpoint_hours":
                time_to_endpoint,
            "failure_within_24h":
                failure_within_24h,
            "isolation_forest_train":
                isolation_forest_train,
        },
        index=df.index,
    )

    df = pd.concat(
        [df, target_df],
        axis=1,
    )

    return df, failure_endpoints


# ============================================================
# Trajectory split
# ============================================================

def build_split_series(df):
    """
    Build the train/val/test assignment separately rather than
    inserting _split into the main target dataset.
    """

    keys = list(
        zip(
            df["test_id"],
            df["bearing_id"].astype(int),
        )
    )

    unknown = sorted(
        {
            key
            for key in keys
            if key not in TRAJECTORY_SPLIT
        }
    )

    if unknown:
        raise ValueError(
            "Missing trajectory split assignments for: "
            f"{unknown}"
        )

    return pd.Series(
        [TRAJECTORY_SPLIT[key] for key in keys],
        index=df.index,
        name="_split",
    )


# ============================================================
# Audit
# ============================================================

def print_audit(
    df,
    failure_endpoints,
):

    print("\n" + "=" * 70)
    print("TARGET ENGINEERING AUDIT")
    print("=" * 70)

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    artifact_count = int(
        df["is_artifact"].sum()
    )

    print("\nARTIFACTS")
    print("-" * 40)
    print(
        f"Artifact rows: {artifact_count:,}"
    )

    print(
        f"Operational rows: "
        f"{len(df) - artifact_count:,}"
    )

    # --------------------------------------------------------
    # Endpoints
    # --------------------------------------------------------

    print("\nFAILED TRAJECTORY ENDPOINTS")
    print("-" * 40)

    for (
        test_id,
        bearing_id,
    ), endpoint in failure_endpoints.items():

        print(
            f"{test_id} / B{bearing_id}: "
            f"{endpoint}"
        )

    # --------------------------------------------------------
    # RUL-style target
    # --------------------------------------------------------

    rul_mask = df[
        "time_to_operational_endpoint_hours"
    ].notna()

    print(
        "\nTIME-TO-OPERATIONAL-ENDPOINT"
    )
    print("-" * 40)

    print(
        f"Labeled rows: "
        f"{rul_mask.sum():,}"
    )

    print(
        f"NaN rows: "
        f"{df['time_to_operational_endpoint_hours'].isna().sum():,}"
    )

    if rul_mask.any():

        rul = df.loc[
            rul_mask,
            "time_to_operational_endpoint_hours",
        ]

        print(
            f"Minimum: {rul.min():.6f} h"
        )

        print(
            f"Maximum: {rul.max():.6f} h"
        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification = df[
        "failure_within_24h"
    ]

    print("\nFAILURE WITHIN 24 HOURS")
    print("-" * 40)

    print(
        f"Eligible rows: "
        f"{classification.notna().sum():,}"
    )

    print(
        f"Positive (1): "
        f"{(classification == 1).sum():,}"
    )

    print(
        f"Negative (0): "
        f"{(classification == 0).sum():,}"
    )

    print(
        f"NaN / ineligible: "
        f"{classification.isna().sum():,}"
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    if_count = int(
        df["isolation_forest_train"].sum()
    )

    print(
        "\nISOLATION FOREST BASELINE"
    )
    print("-" * 40)

    print(
        f"Training rows: {if_count:,}"
    )

    print(
        f"Non-training rows: "
        f"{len(df) - if_count:,}"
    )

    # --------------------------------------------------------
    # Structural checks
    # --------------------------------------------------------

    expected_target_columns = set(
        TARGET_COLUMNS
    )

    actual_target_columns = {
        column
        for column in df.columns
        if column in expected_target_columns
    }

    print("\nTARGET COLUMN CHECK")
    print("-" * 40)

    print(
        f"Target columns present: "
        f"{sorted(actual_target_columns)}"
    )

    # --------------------------------------------------------
    # Dynamic sanity checks
    # --------------------------------------------------------

    failed_rul_count = sum(
        (
            df["test_id"].eq(test_id)
            & df["bearing_id"].eq(bearing_id)
            & df[
                "time_to_operational_endpoint_hours"
            ].notna()
        ).sum()
        for test_id, bearing_id
        in build_failed_bearing_set()
    )

    positive_count = int(
        (df["failure_within_24h"] == 1).sum()
    )

    negative_count = int(
        (df["failure_within_24h"] == 0).sum()
    )

    print("\nSANITY CHECKS")
    print("-" * 40)

    checks = {
        "Target columns = 4":
            actual_target_columns
            == expected_target_columns,

        "No _split in enriched dataset":
            "_split" not in df.columns,

        "RUL rows match failed operational rows":
            failed_rul_count
            == int(
                df[
                    "time_to_operational_endpoint_hours"
                ].notna().sum()
            ),

        "Classification counts reconcile":
            positive_count
            + negative_count
            + int(
                df[
                    "failure_within_24h"
                ].isna().sum()
            )
            == len(df),

        "IF mask excludes artifacts":
            not df.loc[
                df["is_artifact"],
                "isolation_forest_train",
            ].any(),
    }

    all_passed = True

    for name, passed in checks.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"[{status}] {name}"
        )

        if not passed:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print(
            "STATUS: TARGET ENGINEERING PASSED"
        )
    else:
        print(
            "STATUS: TARGET ENGINEERING REQUIRES REVIEW"
        )

    print("=" * 70 + "\n")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TARGET ENGINEERING")
    print("=" * 70)

    print(
        f"\nLoading: {INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    print(
        f"Loaded dataset: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    validate_input(df)

    original_columns = list(
        df.columns
    )

    # --------------------------------------------------------
    # Target generation
    # --------------------------------------------------------

    df, failure_endpoints = create_targets(
        df
    )

    # --------------------------------------------------------
    # Verify exactly four target columns were added.
    # --------------------------------------------------------

    new_columns = [
        column
        for column in df.columns
        if column not in original_columns
    ]

    expected_target_columns = list(TARGET_COLUMNS)

    if new_columns != expected_target_columns:
        raise ValueError(
            "Unexpected target columns.\n"
            f"Expected: {expected_target_columns}\n"
            f"Actual:   {new_columns}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Main enriched output contains NO `_split`.
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved target-enriched dataset:"
        f"\n  {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Build split separately.
    # --------------------------------------------------------

    split_series = build_split_series(df)

    train_mask = (
        split_series == "train"
    )

    val_mask = (
        split_series == "val"
    )

    test_mask = (
        split_series == "test"
    )

    train_df = df.loc[
        train_mask
    ].copy()

    val_df = df.loc[
        val_mask
    ].copy()

    test_df = df.loc[
        test_mask
    ].copy()

    # --------------------------------------------------------
    # Write split datasets.
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    val_df.to_csv(
        VAL_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
    )

    print(
        f"Saved train: "
        f"{TRAIN_FILE} "
        f"({len(train_df):,} rows)"
    )

    print(
        f"Saved validation: "
        f"{VAL_FILE} "
        f"({len(val_df):,} rows)"
    )

    print(
        f"Saved test: "
        f"{TEST_FILE} "
        f"({len(test_df):,} rows)"
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    print_audit(
        df,
        failure_endpoints,
    )


if __name__ == "__main__":
    main()