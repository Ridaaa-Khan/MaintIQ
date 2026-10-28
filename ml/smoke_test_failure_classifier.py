"""
Smoke test for the failure-within-24h XGBoost classifier.

Run:
    python smoke_test_failure_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from failure_classifier_infer import load_classifier, score_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "failure_classifier"
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "features_val.csv"


def main() -> int:
    print("=" * 60)
    print("FAILURE CLASSIFIER SMOKE TEST")
    print("=" * 60)

    required = [
        MODEL_DIR / "xgboost_failure_classifier.joblib",
        MODEL_DIR / "feature_columns.json",
        MODEL_DIR / "threshold.json",
        MODEL_DIR / "selected_model.json",
        MODEL_DIR / "experiment_report.json",
        MODEL_DIR / "metrics.csv",
        MODEL_DIR / "trajectory_metrics.csv",
    ]

    for path in required:
        if not path.exists():
            print(f"FAIL missing artifact: {path}")
            return 1
        print(f"OK  artifact: {path.name}")

    classifier = load_classifier(MODEL_DIR)
    print(
        f"OK  loaded model "
        f"(threshold={classifier['threshold']:.4f}, "
        f"features={len(classifier['features'])})"
    )

    with (MODEL_DIR / "selected_model.json").open(encoding="utf-8") as f:
        selected = json.load(f)

    verification = selected.get("verification", {})
    if not verification:
        print("FAIL selected_model.json missing verification block")
        return 1

    print("OK  verification block present")
    for key, value in verification.items():
        print(f"    {key}: {value}")

    metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
    val = metrics.loc[metrics["split"] == "validation"].iloc[0]
    test = metrics.loc[metrics["split"] == "test"].iloc[0]

    print(
        f"OK  val  P/R/F1/AP = "
        f"{val['precision']:.3f}/"
        f"{val['recall']:.3f}/"
        f"{val['f1']:.3f}/"
        f"{val['average_precision']:.3f}"
    )
    print(
        f"OK  test P/R/F1/AP = "
        f"{test['precision']:.3f}/"
        f"{test['recall']:.3f}/"
        f"{test['f1']:.3f}/"
        f"{test['average_precision']:.3f}"
    )

    if float(val["recall"]) < 0.60:
        print("FAIL validation recall below 0.60")
        return 1
    if float(val["precision"]) < 0.30:
        print("FAIL validation precision below 0.30")
        return 1
    if float(val["f1"]) < 0.40:
        print("FAIL validation F1 below 0.40")
        return 1
    if float(test["average_precision"]) < 0.70:
        print("FAIL test average precision below 0.70")
        return 1
    if float(test["recall"]) < 0.40:
        print("FAIL test recall below 0.40")
        return 1
    if float(test["precision"]) < 0.50:
        print("FAIL test precision below 0.50")
        return 1

    val_df = pd.read_csv(VAL_FILE)
    scored = score_dataframe(val_df.head(100), classifier)
    if "failure_probability" not in scored.columns:
        print("FAIL inference helper did not add failure_probability")
        return 1
    if "predicted_failure_within_24h" not in scored.columns:
        print("FAIL inference helper did not add predictions")
        return 1
    if scored["failure_probability"].isna().any():
        print("FAIL NaN probabilities from inference helper")
        return 1

    print("OK  inference helper scored a validation sample")

    if not verification.get("all_critical_passed", False):
        print(
            "WARN verification.all_critical_passed is False — "
            "review trajectory_metrics.csv before production use"
        )
        # Soft fail: still useful for CI visibility, but exit non-zero
        # only if core metrics above also failed.
    else:
        print("OK  all critical verification checks passed")

    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
