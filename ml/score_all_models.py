"""
Score a model-ready feature CSV with Isolation Forest, the failure
classifier, and the RUL regressor in one pass.

Usage:
    python score_all_models.py data/processed/features_val.csv
    python score_all_models.py data/processed/features_val.csv --output scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from failure_classifier_infer import load_classifier, score_dataframe as score_classifier
from isolation_forest_infer import load_detector, score_dataframe as score_detector
from rul_regressor_infer import load_regressor, score_dataframe as score_regressor
from scoring_utils import validate_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parent


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    detector = load_detector()
    classifier = load_classifier()
    regressor = load_regressor()

    # Shared schema check against the trained 190-column list
    # (all three models use the same feature_columns.json content).
    _, report = validate_feature_matrix(df, detector["features"])
    print(
        f"Schema OK: {report.n_rows} rows, "
        f"{report.n_required_features} features, "
        f"{report.n_nan_cells} NaN cells, "
        f"{report.n_inf_cells} inf cells replaced"
    )

    scored = score_detector(df, detector)
    scored = score_classifier(scored, classifier)
    scored = score_regressor(scored, regressor)
    scored["operational_alert"] = (
        scored["anomaly_flag"].astype(bool)
        | (scored["predicted_failure_within_24h"].astype(int) == 1)
    )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score IF + classifier + RUL on one feature CSV."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    scored = score_all(df)

    output = args.output
    if output is None:
        output = args.input_csv.with_name(args.input_csv.stem + "_all_scored.csv")

    scored.to_csv(output, index=False)
    print(f"Wrote {len(scored):,} rows -> {output}")
    print(
        "Columns added: anomaly_score, anomaly_flag, "
        "failure_probability, predicted_failure_within_24h, "
        "predicted_rul_hours, operational_alert"
    )
    print(
        "Note: RUL is trained on failed trajectories only. "
        "Treat predicted_rul_hours on healthy/censored rows as exploratory."
    )


if __name__ == "__main__":
    main()
