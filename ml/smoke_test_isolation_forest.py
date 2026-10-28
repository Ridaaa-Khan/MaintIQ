"""
Smoke test for Isolation Forest anomaly detection.

Verifies:
1. Selected model artifacts exist and load.
2. Inference helper scores rows.
3. Training baseline rows are mostly inliers.
4. Failed validation bearing shows late-life score lift / flags.

Run:
    python smoke_test_isolation_forest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from isolation_forest_infer import load_detector, score_dataframe
from pipeline_config import failed_bearing_set

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "isolation_forest"
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "features_val.csv"
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "features_train.csv"


def _parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"]
    )


def main() -> int:
    print("=" * 60)
    print("ISOLATION FOREST SMOKE TEST")
    print("=" * 60)

    required = [
        MODEL_DIR / "isolation_forest_selected.joblib",
        MODEL_DIR / "selected_model.json",
        MODEL_DIR / "feature_columns.json",
        MODEL_DIR / "failure_trajectory_analysis.csv",
    ]
    for path in required:
        if not path.exists():
            print(f"FAIL missing artifact: {path}")
            return 1
        print(f"OK  artifact: {path.name}")

    detector = load_detector(MODEL_DIR)
    print(
        f"OK  loaded selected model "
        f"({detector['selected_label']}, "
        f"contamination={detector['contamination']})"
    )
    print(f"OK  feature count: {len(detector['features'])}")

    with (MODEL_DIR / "selected_model.json").open(encoding="utf-8") as f:
        selected = json.load(f)

    verification = selected.get("verification", {})
    if not verification.get("all_critical_passed", False):
        print("FAIL selected_model.json verification.all_critical_passed is False")
        return 1
    print("OK  training verification flags all passed")

    train = pd.read_csv(TRAIN_FILE)
    baseline = _parse_bool(train["isolation_forest_train"])
    artifact = _parse_bool(train["is_artifact"])
    baseline_df = train.loc[baseline & ~artifact].head(200)
    scored_baseline = score_dataframe(baseline_df, detector)
    baseline_flag_rate = float(scored_baseline["anomaly_flag"].mean())
    print(f"OK  baseline sample flag rate: {baseline_flag_rate:.3%}")
    if baseline_flag_rate >= 0.15:
        print("FAIL baseline sample is flagged too often")
        return 1

    val = pd.read_csv(VAL_FILE, parse_dates=["timestamp"])
    failed = failed_bearing_set()
    val_failed = None
    for (test_id, bearing_id), group in val.groupby(
        ["test_id", "bearing_id"]
    ):
        if (test_id, int(bearing_id)) in failed:
            val_failed = group.sort_values("timestamp")
            break

    if val_failed is None or val_failed.empty:
        print("FAIL no failed validation trajectory found")
        return 1

    val_failed = val_failed.loc[~_parse_bool(val_failed["is_artifact"])]
    n = len(val_failed)
    window = max(1, n // 5)
    early = score_dataframe(val_failed.iloc[:window], detector)
    late = score_dataframe(val_failed.iloc[-window:], detector)
    early_mean = float(early["anomaly_score"].mean())
    late_mean = float(late["anomaly_score"].mean())
    late_flag_rate = float(late["anomaly_flag"].mean())

    print(
        f"OK  val failed trajectory: "
        f"{val_failed['test_id'].iloc[0]} "
        f"B{int(val_failed['bearing_id'].iloc[0])}"
    )
    print(f"OK  early mean score: {early_mean:.4f}")
    print(f"OK  late mean score : {late_mean:.4f}")
    print(f"OK  late flag rate  : {late_flag_rate:.3%}")

    if late_mean <= early_mean:
        print("FAIL late-life score did not rise on failed val bearing")
        return 1
    if late_flag_rate < 0.5:
        print("FAIL late-life flag rate too low on failed val bearing")
        return 1

    analysis = pd.read_csv(MODEL_DIR / "failure_trajectory_analysis.csv")
    failed_val_rows = analysis[
        (analysis["split"] == "val") & (analysis["is_failed"])
    ]
    if failed_val_rows.empty:
        print("FAIL failure_trajectory_analysis.csv missing val failed rows")
        return 1
    if not bool((failed_val_rows["score_lift_late_minus_early"] > 0).all()):
        print("FAIL analysis file shows non-positive val failed lift")
        return 1

    print("=" * 60)
    print("SMOKE TEST PASSED — anomaly detection looks operational")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
