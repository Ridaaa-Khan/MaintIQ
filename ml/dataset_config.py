"""
Load the active dataset specification.

IMS constants that used to live as Python literals in extract_features.py
and pipeline_config.py now come from configs/datasets/nasa_ims.json.

A future vibration dataset should add another JSON file and point
DATASET_SPEC_PATH at it — waveform math stays in extract_features.py.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC_PATH = PROJECT_ROOT / "configs" / "datasets" / "nasa_ims.json"

REQUIRED_TOP_LEVEL = (
    "dataset_id",
    "raw_root",
    "acquisition",
    "bearing",
    "envelope_band",
    "channel_map",
    "failed_bearings",
    "trajectory_split",
    "targets",
    "temporal",
    "column_roles",
)


class DatasetConfigError(ValueError):
    """Raised when the dataset spec is missing or malformed."""


@lru_cache(maxsize=4)
def load_dataset_spec(path: str | Path | None = None) -> dict[str, Any]:
    spec_path = Path(path) if path else DEFAULT_SPEC_PATH
    if not spec_path.exists():
        raise DatasetConfigError(f"Dataset spec not found: {spec_path}")

    with spec_path.open(encoding="utf-8") as f:
        spec = json.load(f)

    missing = [key for key in REQUIRED_TOP_LEVEL if key not in spec]
    if missing:
        raise DatasetConfigError(
            f"{spec_path.name} missing required keys: {missing}"
        )

    acq = spec["acquisition"]
    for key in ("sampling_rate_hz", "signal_length", "timestamp_pattern"):
        if key not in acq:
            raise DatasetConfigError(f"acquisition.{key} is required")

    bearing = spec["bearing"]
    for key in (
        "rpm",
        "number_of_rolling_elements",
        "pitch_diameter_mm",
        "rolling_element_diameter_mm",
        "contact_angle_degrees",
    ):
        if key not in bearing:
            raise DatasetConfigError(f"bearing.{key} is required")

    return spec


def channel_map(spec: dict[str, Any] | None = None) -> dict[str, dict[int, list[int]]]:
    spec = spec or load_dataset_spec()
    mapped: dict[str, dict[int, list[int]]] = {}
    for test_id, bearings in spec["channel_map"].items():
        mapped[test_id] = {
            int(bearing_id): [int(ch) for ch in channels]
            for bearing_id, channels in bearings.items()
        }
    return mapped


def trajectory_split(spec: dict[str, Any] | None = None) -> dict[tuple, str]:
    spec = spec or load_dataset_spec()
    split: dict[tuple, str] = {}
    for partition, pairs in spec["trajectory_split"].items():
        for test_id, bearing_id in pairs:
            key = (str(test_id), int(bearing_id))
            if key in split:
                raise DatasetConfigError(
                    f"Trajectory {key} listed in more than one split."
                )
            split[key] = str(partition)
    return split


def failed_bearings(spec: dict[str, Any] | None = None) -> dict[str, list[int]]:
    spec = spec or load_dataset_spec()
    return {
        str(test_id): [int(b) for b in bearings]
        for test_id, bearings in spec["failed_bearings"].items()
    }


def raw_root(spec: dict[str, Any] | None = None) -> Path:
    spec = spec or load_dataset_spec()
    return PROJECT_ROOT / spec["raw_root"]
