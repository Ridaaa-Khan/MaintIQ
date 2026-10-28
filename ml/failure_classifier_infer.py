"""
Failure-within-24h classifier inference helper.

Loads the calibrated XGBoost model + validation-selected threshold
from models/failure_classifier/ and scores feature rows.

Usage:
    from failure_classifier_infer import load_classifier, score_dataframe

    clf = load_classifier()
    scored = score_dataframe(df, clf)

CLI:
    python failure_classifier_infer.py data/processed/features_val.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from scoring_utils import validate_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "failure_classifier"


def load_classifier(model_dir: Path | None = None) -> dict:
    """Load calibrated classifier, features, and decision threshold."""
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    model_path = model_dir / "xgboost_failure_classifier.joblib"
    feature_path = model_dir / "feature_columns.json"
    threshold_path = model_dir / "threshold.json"
    selected_path = model_dir / "selected_model.json"

    for path in (model_path, feature_path, threshold_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run train_failure_classifier.py first."
            )

    model = joblib.load(model_path)

    with feature_path.open(encoding="utf-8") as f:
        features = json.load(f)

    with threshold_path.open(encoding="utf-8") as f:
        threshold_meta = json.load(f)

    selected_meta = None
    if selected_path.exists():
        with selected_path.open(encoding="utf-8") as f:
            selected_meta = json.load(f)

    return {
        "model": model,
        "features": list(features),
        "threshold": float(threshold_meta["threshold"]),
        "threshold_meta": threshold_meta,
        "selected_meta": selected_meta,
        "model_dir": model_dir,
    }


def score_dataframe(
    df: pd.DataFrame,
    classifier: dict | None = None,
) -> pd.DataFrame:
    """
    Score rows with the failure classifier.

    Adds:
        failure_probability
        predicted_failure_within_24h
    """
    if classifier is None:
        classifier = load_classifier()

    features = classifier["features"]
    result, _report = validate_feature_matrix(
        df, features, allow_nan=True, replace_inf=True
    )
    X = result[features]

    probability = classifier["model"].predict_proba(X)[:, 1]
    threshold = classifier["threshold"]

    result["failure_probability"] = probability
    result["predicted_failure_within_24h"] = (
        probability >= threshold
    ).astype(int)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a feature CSV with the failure-within-24h classifier."
        )
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    args = parser.parse_args()

    classifier = load_classifier(args.model_dir)
    df = pd.read_csv(args.input_csv)
    scored = score_dataframe(df, classifier)

    output = args.output
    if output is None:
        output = args.input_csv.with_name(
            args.input_csv.stem + "_failure_scored.csv"
        )

    scored.to_csv(output, index=False)

    positive_rate = float(scored["predicted_failure_within_24h"].mean())
    print(
        f"Loaded classifier (threshold={classifier['threshold']:.4f})"
    )
    print(f"Scored rows     : {len(scored):,}")
    print(f"Predicted pos % : {positive_rate:.3%}")
    print(f"Wrote           : {output}")


if __name__ == "__main__":
    main()
