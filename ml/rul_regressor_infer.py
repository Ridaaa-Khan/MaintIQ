"""
RUL regressor inference helper.

Loads models/rul_regressor/xgboost_rul_regressor.joblib and scores
feature rows. The model was trained on log1p(RUL); this helper
returns hours on the original scale.

Usage:
    from rul_regressor_infer import load_regressor, score_dataframe

    reg = load_regressor()
    scored = score_dataframe(df, reg)

CLI:
    python rul_regressor_infer.py data/processed/features_val.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scoring_utils import validate_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "rul_regressor"


def load_regressor(model_dir: Path | None = None) -> dict:
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    model_path = model_dir / "xgboost_rul_regressor.joblib"
    feature_path = model_dir / "feature_columns.json"
    selected_path = model_dir / "selected_model.json"

    for path in (model_path, feature_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run train_rul_regressor.py first."
            )

    model = joblib.load(model_path)
    with feature_path.open(encoding="utf-8") as f:
        features = json.load(f)

    selected_meta = None
    target_transform = "log1p / expm1"
    if selected_path.exists():
        with selected_path.open(encoding="utf-8") as f:
            selected_meta = json.load(f)
        target_transform = selected_meta.get(
            "target_transform", target_transform
        )

    return {
        "model": model,
        "features": list(features),
        "target_transform": target_transform,
        "selected_meta": selected_meta,
        "model_dir": model_dir,
    }


def score_dataframe(
    df: pd.DataFrame,
    regressor: dict | None = None,
) -> pd.DataFrame:
    """
    Score rows with the RUL regressor.

    Adds:
        predicted_rul_hours
    """
    if regressor is None:
        regressor = load_regressor()

    features = regressor["features"]
    result, _report = validate_feature_matrix(
        df, features, allow_nan=True, replace_inf=True
    )
    X = result[features]

    # Model outputs log1p(RUL); invert to hours.
    log_pred = regressor["model"].predict(X)
    result["predicted_rul_hours"] = np.maximum(np.expm1(log_pred), 0.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a feature CSV with the RUL regressor."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    args = parser.parse_args()

    regressor = load_regressor(args.model_dir)
    df = pd.read_csv(args.input_csv)
    scored = score_dataframe(df, regressor)

    output = args.output
    if output is None:
        output = args.input_csv.with_name(
            args.input_csv.stem + "_rul_scored.csv"
        )

    scored.to_csv(output, index=False)
    print(
        f"Loaded RUL model "
        f"(transform={regressor['target_transform']})"
    )
    print(f"Scored rows: {len(scored):,}")
    print(
        "Predicted RUL hours: "
        f"min={scored['predicted_rul_hours'].min():.2f}, "
        f"median={scored['predicted_rul_hours'].median():.2f}, "
        f"max={scored['predicted_rul_hours'].max():.2f}"
    )
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
