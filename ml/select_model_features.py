"""
select_model_features.py
========================
Trajectory-aware feature selection for the IMS bearing dataset.

WHY TRAJECTORY-AWARE?
---------------------
Each (test_id, bearing_id) pair is a single physical run that produces
hundreds of temporally correlated rows.  A random row-level split leaks
future measurements of the same bearing into the training set, giving
optimistic estimates for any model that uses temporal features.

SPLIT STRATEGY
--------------
We split by *trajectory* (unique test_id x bearing_id combinations),
not by row.  There are 12 trajectories total (3 tests x 4 bearings).
The default split assigns 7 / 2 / 3 trajectories to train / val / test,
keeping at least one failed bearing in every partition.

SELECTION RULES (fit on TRAIN only)
-------------------------------------
1. Low-variance features computed from TRAIN rows only.
2. All *_rate_per_hour temporal features.

Both rules are applied identically to val and test after being decided
on train.  No variance, correlation, or scaling decisions touch val/test
data before those rules are applied.

OUTPUTS
-------
data/processed/
    features_model_ready.csv  - full unlabeled model-ready feature table
    split_metadata.json       - trajectory assignments + dropped features

Train / val / test CSVs are written only by create_targets.py so they
always include labels. Re-running this script must not overwrite those
labeled splits with unlabeled rows.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd

from pipeline_config import METADATA_COLUMNS, TRAJECTORY_SPLIT


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = str(DATA_DIR / "features_selected.csv")
OUTPUT_DIR = str(DATA_DIR)
OUTPUT_MODEL_READY = str(DATA_DIR / "features_model_ready.csv")
OUTPUT_METADATA = str(DATA_DIR / "split_metadata.json")

METADATA_COLUMN_SET = set(METADATA_COLUMNS)

# Low-variance cutoff is computed at runtime from TRAIN rows only.
VARIANCE_THRESHOLD = 1e-10

# Characteristic-frequency features must stay even if they are nearly
# constant on the train trajectories (e.g. 2xBSF is most relevant to
# the Test1 Bearing 4 roller fault, which is held out in val).
PROTECTED_FEATURE_PREFIXES = (
    "bsf_2x_",
    "bsf_energy_",
    "bpfo_local_ratio_",
    "bpfi_local_ratio_",
    "bpfo_energy_",
    "bpfi_energy_",
)


# ============================================================
# HELPERS
# ============================================================

def assign_split(df):
    """
    Add a 'split' column based on TRAJECTORY_SPLIT.

    Raises KeyError if any trajectory in df is not listed in
    TRAJECTORY_SPLIT, preventing silent mis-assignment.
    """
    keys = list(
        zip(df["test_id"], df["bearing_id"].astype(int))
    )
    splits = []
    for key in keys:
        if key not in TRAJECTORY_SPLIT:
            raise KeyError(
                f"Trajectory {key} not listed in TRAJECTORY_SPLIT. "
                f"Add it before running selection."
            )
        splits.append(TRAJECTORY_SPLIT[key])
    df = df.copy()
    df["split"] = splits
    return df


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("TRAJECTORY-AWARE FEATURE SELECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print()
    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    original_shape    = df.shape
    original_row_count = len(df)

    print(f"Input shape: {original_shape}")

    # --------------------------------------------------------
    # 2. ASSIGN TRAJECTORIES TO SPLITS
    # --------------------------------------------------------

    df = assign_split(df)

    train_mask = df["split"] == "train"
    val_mask   = df["split"] == "val"
    test_mask  = df["split"] == "test"

    n_train = int(train_mask.sum())
    n_val   = int(val_mask.sum())
    n_test  = int(test_mask.sum())

    print()
    print("=" * 70)
    print("TRAJECTORY SPLIT")
    print("=" * 70)

    for partition in ("train", "val", "test"):
        trajs = sorted({
            k for k, v in TRAJECTORY_SPLIT.items()
            if v == partition
        })
        count = (df["split"] == partition).sum()
        print(
            f"\n  {partition.upper()} "
            f"({len(trajs)} trajectories, {count:,} rows):"
        )
        for traj in trajs:
            print(f"    {traj}")

    assert n_train + n_val + n_test == original_row_count, (
        "Split row counts do not sum to total rows."
    )

    # --------------------------------------------------------
    # 3. IDENTIFY FEATURES
    # --------------------------------------------------------

    all_feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMN_SET and col != "split"
    ]

    print()
    print(f"Total feature columns: {len(all_feature_columns)}")

    # --------------------------------------------------------
    # 4. DECIDE WHAT TO DROP  (train data only)
    # --------------------------------------------------------
    #
    # ALL selection decisions are derived exclusively from the
    # train partition.  Val and test are transformed by the same
    # rule set without contributing any information to the decision.

    print()
    print("=" * 70)
    print("FEATURE SELECTION  (fit on TRAIN only)")
    print("=" * 70)

    train_features = [
        col for col in all_feature_columns
        if col in df.columns
    ]
    train_numeric = (
        df.loc[train_mask, train_features]
        .select_dtypes(include="number")
    )
    train_variances = train_numeric.var(ddof=0)
    low_variance_to_remove = set(
        train_variances[train_variances <= VARIANCE_THRESHOLD].index
    )
    low_variance_to_remove = {
        col for col in low_variance_to_remove
        if not col.startswith(PROTECTED_FEATURE_PREFIXES)
    }

    # All temporal rate-per-hour features
    rate_features_to_remove = {
        col for col in df.columns
        if col.endswith("_rate_per_hour")
    }

    features_to_remove = (
        low_variance_to_remove | rate_features_to_remove
    )

    # Safety: never drop metadata or split label
    features_to_remove -= METADATA_COLUMN_SET
    features_to_remove.discard("split")

    print(
        f"\nLow-variance features "
        f"({len(low_variance_to_remove)}):"
    )
    for f in sorted(low_variance_to_remove):
        print(f"  - {f}")

    print(
        f"\nRate-per-hour features "
        f"({len(rate_features_to_remove)}):"
    )
    for f in sorted(rate_features_to_remove):
        print(f"  - {f}")

    print(f"\nTotal features to remove: {len(features_to_remove)}")

    # --------------------------------------------------------
    # 5. APPLY SELECTION TO ALL PARTITIONS
    # --------------------------------------------------------

    columns_to_keep = [
        col for col in df.columns
        if col not in features_to_remove
    ]

    df = df[columns_to_keep]

    remaining_feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS and col != "split"
    ]

    print(f"Remaining features: {len(remaining_feature_columns)}")

    # --------------------------------------------------------
    # 6. INTEGRITY CHECKS
    # --------------------------------------------------------

    # Row count must never change
    if len(df) != original_row_count:
        raise RuntimeError(
            "ERROR: Row count changed during feature selection."
        )

    # No metadata lost
    missing_metadata = METADATA_COLUMN_SET - set(df.columns)
    if missing_metadata:
        raise RuntimeError(
            f"ERROR: Metadata columns missing: "
            f"{sorted(missing_metadata)}"
        )

    # No removed features survived
    surviving_removed = features_to_remove & set(df.columns)
    if surviving_removed:
        raise RuntimeError(
            "ERROR: Some removed features still present: "
            f"{sorted(surviving_removed)}"
        )

    # No missing or infinite values
    numeric_cols = df.select_dtypes(include="number").columns
    n_missing  = int(df[numeric_cols].isna().sum().sum())
    n_infinite = int(
        df[numeric_cols]
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    if n_missing != 0:
        raise RuntimeError(
            f"ERROR: {n_missing} missing value(s) detected."
        )
    if n_infinite != 0:
        raise RuntimeError(
            f"ERROR: {n_infinite} infinite value(s) detected."
        )

    print()
    print("=" * 70)
    print("DATA INTEGRITY OK")
    print("=" * 70)
    print(f"  Missing values : {n_missing}")
    print(f"  Infinite values: {n_infinite}")

    # --------------------------------------------------------
    # 7. SAVE UNLABELED MODEL-READY TABLE ONLY
    #
    # Do NOT write features_train/val/test here. Those files are
    # owned by create_targets.py and must stay label-enriched.
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Drop the helper 'split' column before saving so downstream
    # code does not accidentally treat it as a feature.
    output_columns = [
        col for col in df.columns if col != "split"
    ]

    df[output_columns].to_csv(OUTPUT_MODEL_READY, index=False)

    # --------------------------------------------------------
    # 8. SAVE SPLIT METADATA
    # --------------------------------------------------------

    metadata = {
        "input_file": INPUT_FILE,
        "split_strategy": "trajectory (test_id, bearing_id)",
        "leakage_note": (
            "Selection rules fit on TRAIN only. "
            "Val and test are transformed by the same rules "
            "without contributing to selection decisions."
        ),
        "labeled_splits_note": (
            "features_train.csv / features_val.csv / "
            "features_test.csv are written by create_targets.py "
            "after target engineering. Re-run create_targets.py "
            "after this script."
        ),
        "row_counts": {
            "total": original_row_count,
            "train": n_train,
            "val":   n_val,
            "test":  n_test,
        },
        "trajectory_assignments": {
            f"{k[0]}_b{k[1]}": v
            for k, v in TRAJECTORY_SPLIT.items()
        },
        "features": {
            "total_input":     len(all_feature_columns),
            "removed_low_var": sorted(low_variance_to_remove),
            "removed_rate":    sorted(rate_features_to_remove),
            "remaining":       sorted(remaining_feature_columns),
        },
        "outputs": {
            "model_ready": OUTPUT_MODEL_READY,
            "metadata": OUTPUT_METADATA,
            "labeled_splits_from": "create_targets.py",
        },
    }

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # --------------------------------------------------------
    # 9. FINAL REPORT
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("MODEL-READY DATASET CREATED")
    print("=" * 70)
    print(f"  Full unlabeled : {OUTPUT_MODEL_READY}")
    print(
        f"                   {original_row_count:,} rows, "
        f"{len(remaining_feature_columns)} features"
    )
    print(f"  Meta           : {OUTPUT_METADATA}")
    print(
        "  Labeled splits : written later by create_targets.py "
        "(not overwritten here)"
    )
    print()
    print(
        "Planned trajectory row counts "
        f"(train/val/test): {n_train:,} / {n_val:,} / {n_test:,}"
    )
    print(f"Features removed : {len(features_to_remove)}")
    print(f"Features remaining: {len(remaining_feature_columns)}")
    print(f"Time taken: {elapsed:.2f} s")
    print()
    print("Selection complete. Next: run create_targets.py")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
