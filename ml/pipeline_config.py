"""
Shared pipeline configuration.

Split, failures, target names, and thresholds are loaded from
configs/datasets/nasa_ims.json so a second IMS-like dataset can be
wired without rewriting training code.
"""

from dataset_config import (
    failed_bearings as _failed_bearings,
    load_dataset_spec,
    trajectory_split as _trajectory_split,
)

_SPEC = load_dataset_spec()
_TARGETS = _SPEC["targets"]
_ROLES = _SPEC["column_roles"]

# ============================================================
# Trajectory-level train / val / test split
# ============================================================
#
# Keys are (test_id, bearing_id). All 12 IMS trajectories are listed.
# Never split by shuffled rows — that leaks future measurements.
#
# Failures (IMS paper):
#   1st_test B3 inner-race, 1st_test B4 rolling-element,
#   2nd_test B1 outer-race, 3rd_test B3 outer-race.
# At least one failed trajectory is in every partition.

TRAJECTORY_SPLIT = _trajectory_split(_SPEC)


# ============================================================
# Documented failed trajectories
# ============================================================

FAILED_BEARINGS = _failed_bearings(_SPEC)


# ============================================================
# Target-engineering constants
# ============================================================

ARTIFACT_RMS_THRESHOLD = float(_TARGETS["artifact_rms_threshold"])
FAILURE_HORIZON_HOURS = float(_TARGETS["failure_horizon_hours"])
NORMAL_BASELINE_FRACTION = float(_TARGETS["normal_baseline_fraction"])


# ============================================================
# Column roles (features vs metadata vs targets)
# ============================================================
#
# Training code must build X from feature columns only.
# Never put metadata or target/mask columns into the model matrix.

METADATA_COLUMNS = tuple(_ROLES["metadata"])

TARGET_COLUMNS = tuple(_ROLES["targets"])

NON_FEATURE_COLUMNS = METADATA_COLUMNS + TARGET_COLUMNS


def failed_bearing_set():
    """Return documented failed trajectories as a set of tuples."""
    return {
        (test_id, bearing_id)
        for test_id, bearing_ids in FAILED_BEARINGS.items()
        for bearing_id in bearing_ids
    }


def feature_columns(columns):
    """
    Return model feature names from a column list or Index.

    Excludes metadata and all target / mask columns.
    """
    non_features = set(NON_FEATURE_COLUMNS)
    return [column for column in columns if column not in non_features]
