"""
End-to-end ML readiness smoke test.

Checks dataset config, trained artifacts, schema validation, and that
all three infer helpers (plus SHAP artifacts) still work.

    python smoke_test_ml.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import smoke_test_failure_classifier
import smoke_test_isolation_forest
import smoke_test_rul_regressor
import smoke_test_shap
from dataset_config import failed_bearings, load_dataset_spec, trajectory_split
from failure_classifier_infer import load_classifier
from isolation_forest_infer import load_detector
from pipeline_config import FAILED_BEARINGS, TRAJECTORY_SPLIT
from rul_regressor_infer import load_regressor
from score_all_models import score_all
from scoring_utils import FeatureSchemaError, validate_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "features_val.csv"


def _ok(message: str) -> None:
    print(f"OK  {message}")


def _fail(message: str) -> int:
    print(f"FAIL {message}")
    return 1


def check_config() -> int:
    spec = load_dataset_spec()
    if spec["dataset_id"] != "nasa_ims":
        return _fail(f"unexpected dataset_id: {spec['dataset_id']}")
    if spec["acquisition"]["sampling_rate_hz"] != 20000:
        return _fail("sampling_rate_hz drifted from IMS spec 20000")
    if spec["acquisition"]["signal_length"] != 20480:
        return _fail("signal_length drifted from IMS spec 20480")
    if spec["bearing"]["rpm"] != 2000:
        return _fail("rpm drifted from IMS spec 2000")

    loaded_split = trajectory_split(spec)
    if loaded_split != TRAJECTORY_SPLIT:
        return _fail("pipeline_config.TRAJECTORY_SPLIT does not match JSON spec")
    if FAILED_BEARINGS != failed_bearings(spec):
        return _fail("FAILED_BEARINGS does not match JSON spec")

    n_traj = len(TRAJECTORY_SPLIT)
    if n_traj != 12:
        return _fail(f"expected 12 trajectories, found {n_traj}")

    _ok(f"dataset spec {spec['dataset_id']} ({n_traj} trajectories)")
    return 0


def check_schema_and_joint_score() -> int:
    if not VAL_FILE.exists():
        return _fail(f"missing {VAL_FILE}")

    df = pd.read_csv(VAL_FILE)
    detector = load_detector()
    classifier = load_classifier()
    regressor = load_regressor()

    if len(detector["features"]) < 50:
        return _fail(f"IF feature count {len(detector['features'])} is too small")
    if detector["features"] != classifier["features"]:
        return _fail("IF and classifier feature lists differ")
    if detector["features"] != regressor["features"]:
        return _fail("IF and RUL feature lists differ")

    _, report = validate_feature_matrix(df, detector["features"])
    _ok(
        f"val schema: {report.n_rows} rows, "
        f"{report.n_required_features} features, "
        f"nan={report.n_nan_cells}"
    )

    try:
        validate_feature_matrix(df.drop(columns=[detector["features"][0]]), detector["features"])
        return _fail("schema validator accepted a table with a missing feature")
    except FeatureSchemaError:
        _ok("schema validator rejects missing features")

    sample = df.head(25)
    scored = score_all(sample)
    required_cols = [
        "anomaly_score",
        "anomaly_flag",
        "failure_probability",
        "predicted_failure_within_24h",
        "predicted_rul_hours",
        "operational_alert",
    ]
    missing = [c for c in required_cols if c not in scored.columns]
    if missing:
        return _fail(f"joint scoring missing columns: {missing}")
    if scored["failure_probability"].isna().any():
        return _fail("NaN failure_probability from joint scoring")
    if (scored["predicted_rul_hours"] < 0).any():
        return _fail("negative RUL hours")

    _ok(
        f"joint scoring sample n={len(scored)} "
        f"clf_threshold={classifier['threshold']:.3f} "
        f"if_label={detector['selected_label']}"
    )
    return 0


def check_metrics_files() -> int:
    clf = pd.read_csv(
        PROJECT_ROOT / "models" / "failure_classifier" / "metrics.csv"
    )
    val = clf.loc[clf["split"] == "validation"].iloc[0]
    test = clf.loc[clf["split"] == "test"].iloc[0]
    _ok(
        "classifier val  "
        f"P={val['precision']:.3f} R={val['recall']:.3f} "
        f"F1={val['f1']:.3f} Acc={val['accuracy']:.3f} "
        f"ROC-AUC={val['roc_auc']:.3f} AP={val['average_precision']:.3f}"
    )
    _ok(
        "classifier test "
        f"P={test['precision']:.3f} R={test['recall']:.3f} "
        f"F1={test['f1']:.3f} Acc={test['accuracy']:.3f} "
        f"ROC-AUC={test['roc_auc']:.3f} AP={test['average_precision']:.3f}"
    )

    rul = pd.read_csv(PROJECT_ROOT / "models" / "rul_regressor" / "metrics.csv")
    rul_val = rul.loc[rul["split"] == "validation"].iloc[0]
    rul_test = rul.loc[rul["split"] == "test"].iloc[0]
    _ok(
        "RUL val  MAE="
        f"{rul_val['mae_hours']:.1f}h  test MAE={rul_test['mae_hours']:.1f}h"
    )

    if_sel = json.loads(
        (PROJECT_ROOT / "models" / "isolation_forest" / "selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    if not if_sel.get("verification", {}).get("all_critical_passed"):
        return _fail("Isolation Forest verification.all_critical_passed is false")
    _ok(
        f"IF selected={if_sel['selected_label']} "
        f"contamination={if_sel['selected_contamination']}"
    )
    return 0


def main() -> int:
    print("=" * 60)
    print("ML PIPELINE SMOKE TEST")
    print("=" * 60)

    for fn in (check_config, check_schema_and_joint_score, check_metrics_files):
        code = fn()
        if code != 0:
            return code

    print()
    print("--- isolation forest ---")
    code = smoke_test_isolation_forest.main()
    if code != 0:
        return code

    print()
    print("--- failure classifier ---")
    code = smoke_test_failure_classifier.main()
    if code != 0:
        return code

    print()
    print("--- RUL regressor ---")
    code = smoke_test_rul_regressor.main()
    if code != 0:
        return code

    print()
    print("--- SHAP artifacts ---")
    code = smoke_test_shap.main()
    if code != 0:
        return code

    print()
    print("=" * 60)
    print("ML PIPELINE SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
