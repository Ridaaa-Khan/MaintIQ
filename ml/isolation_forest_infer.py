"""
Isolation Forest inference helper (anomaly detection).

Loads the selected production pipeline from models/isolation_forest/
and scores feature rows without fitting.

Usage:
    from isolation_forest_infer import load_detector, score_dataframe

    detector = load_detector()
    scored = score_dataframe(df, detector)

CLI:
    python isolation_forest_infer.py data/processed/features_val.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scoring_utils import validate_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "isolation_forest"


def load_detector(model_dir: Path | None = None) -> dict:
    """
    Load the selected Isolation Forest pipeline and metadata.

    Returns a dict with:
        pipeline, features, selected_label, contamination, model_dir
    """
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    selected_path = model_dir / "selected_model.json"
    feature_path = model_dir / "feature_columns.json"
    model_path = model_dir / "isolation_forest_selected.joblib"

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing {selected_path}. Run train_isolation_forest.py first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run train_isolation_forest.py first."
        )
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing {feature_path}. Run train_isolation_forest.py first."
        )

    with selected_path.open(encoding="utf-8") as f:
        selected = json.load(f)

    with feature_path.open(encoding="utf-8") as f:
        features = json.load(f)

    pipeline: Pipeline = joblib.load(model_path)

    return {
        "pipeline": pipeline,
        "features": list(features),
        "selected_label": selected["selected_label"],
        "contamination": selected["selected_contamination"],
        "model_dir": model_dir,
        "selected_meta": selected,
    }


def _transform_features(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Xt = X
    for _, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def score_dataframe(
    df: pd.DataFrame,
    detector: dict | None = None,
) -> pd.DataFrame:
    """
    Score rows with the selected anomaly detector.

    Adds:
        anomaly_score   (higher = more anomalous)
        decision_score  (sklearn decision_function)
        anomaly_flag    (True = predicted outlier)
    """
    if detector is None:
        detector = load_detector()

    pipeline: Pipeline = detector["pipeline"]
    features = detector["features"]

    result, _report = validate_feature_matrix(
        df, features, allow_nan=True, replace_inf=True
    )
    X = result[features]
    estimator = pipeline.named_steps["isolation_forest"]
    X_prepared = _transform_features(pipeline, X)

    raw_score = estimator.score_samples(X_prepared)
    result["anomaly_score"] = -raw_score
    result["decision_score"] = estimator.decision_function(X_prepared)
    result["anomaly_flag"] = estimator.predict(X_prepared) == -1

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a feature CSV with the selected Isolation Forest."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Feature CSV with the model-ready columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (default: <input>_scored.csv).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing isolation_forest_selected.joblib.",
    )
    args = parser.parse_args()

    detector = load_detector(args.model_dir)
    df = pd.read_csv(args.input_csv)
    scored = score_dataframe(df, detector)

    output = args.output
    if output is None:
        output = args.input_csv.with_name(
            args.input_csv.stem + "_scored.csv"
        )

    scored.to_csv(output, index=False)

    flag_rate = float(scored["anomaly_flag"].mean())
    print(f"Loaded model: {detector['selected_label']} "
          f"(contamination={detector['contamination']})")
    print(f"Scored rows : {len(scored):,}")
    print(f"Flag rate   : {flag_rate:.3%}")
    print(f"Wrote       : {output}")


if __name__ == "__main__":
    main()
