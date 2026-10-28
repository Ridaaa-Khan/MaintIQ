"""
SHAP explanations for the selected Isolation Forest pipeline.

The saved model is imputer -> scaler -> IsolationForest. TreeExplainer
runs on the IsolationForest step using the already-transformed feature
matrix. Feature names match the training feature_columns.json order.

Run:
    python explain_isolation_forest_shap.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "isolation_forest"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = MODEL_DIR / "shap_explanations"

MODEL_PATH = MODEL_DIR / "isolation_forest_selected.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

ARTIFACT_COL = "is_artifact"
DEFAULT_BACKGROUND_SIZE = 200
DEFAULT_EXPLANATION_SIZE = 500
DEFAULT_RANDOM_STATE = 42
ADDITIVITY_TOL = 1e-4


def parse_boolean(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    result = series.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError(f"Unexpected boolean values in {name}.")
    return result.astype(bool)


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Xt = X
    for name, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
    return np.asarray(Xt, dtype=float)


def prepare_raw_matrix(
    data_path: Path,
    feature_names: list[str],
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Required data not found: {data_path}")

    df = pd.read_csv(data_path)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            f"Model features missing from data: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}"
        )

    keep = pd.Series(True, index=df.index)
    if ARTIFACT_COL in df.columns:
        keep &= ~parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)

    X = df.loc[keep, feature_names].copy()
    if X.empty:
        raise ValueError("No non-artifact rows available for IF SHAP.")
    return X


def expected_path_length(
    iforest: IsolationForest,
    X_transformed: np.ndarray,
) -> np.ndarray:
    """
    Invert IsolationForest.score_samples to the average path length E that
    TreeExplainer reconstructs:

        score_samples = -2 ** (-E / c(max_samples))
    """
    scores = np.asarray(iforest.score_samples(X_transformed), dtype=float)
    c = float(_average_path_length([iforest.max_samples_])[0])
    # Guard tiny numerical overshoots above 0 for -score.
    neg_scores = np.clip(-scores, 1e-300, None)
    return -c * np.log2(neg_scores)


def check_additivity(
    iforest: IsolationForest,
    X_transformed: np.ndarray,
    shap_explanation,
) -> dict:
    base_value = np.asarray(shap_explanation.base_values)
    shap_sum = np.asarray(shap_explanation.values).sum(axis=1)
    # TreeExplainer for IsolationForest explains expected path length E,
    # not decision_function / score_samples directly.
    model_output = expected_path_length(iforest, X_transformed)

    if base_value.ndim == 0:
        reconstructed = float(base_value) + shap_sum
    else:
        reconstructed = base_value.reshape(-1) + shap_sum

    errors = np.abs(model_output - reconstructed)
    return {
        "max_additivity_error": float(errors.max()),
        "mean_additivity_error": float(errors.mean()),
        "tolerance": ADDITIVITY_TOL,
        "passed": bool(errors.max() <= ADDITIVITY_TOL),
        "model_output_name": "expected_path_length",
    }


def run(
    data_path: Path,
    background_size: int,
    explanation_size: int,
    random_state: int,
) -> dict:
    for path in (MODEL_PATH, FEATURES_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. "
                "Run train_isolation_forest.py first."
            )

    pipeline = joblib.load(MODEL_PATH)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(pipeline)}.")

    iforest = pipeline.named_steps.get("isolation_forest")
    if not isinstance(iforest, IsolationForest):
        raise TypeError(
            "Pipeline final step is not IsolationForest "
            f"(got {type(iforest)})."
        )

    with FEATURES_PATH.open(encoding="utf-8") as f:
        feature_names = list(json.load(f))

    X_raw = prepare_raw_matrix(data_path, feature_names)

    rng = np.random.default_rng(random_state)
    background_n = min(background_size, len(X_raw))
    explanation_n = min(explanation_size, len(X_raw))

    background_idx = rng.choice(len(X_raw), size=background_n, replace=False)
    explanation_idx = rng.choice(len(X_raw), size=explanation_n, replace=False)

    X_background_raw = X_raw.iloc[background_idx]
    X_explain_raw = X_raw.iloc[explanation_idx]

    X_background = transform_features(pipeline, X_background_raw)
    X_explain = transform_features(pipeline, X_explain_raw)

    if X_explain.shape[1] != len(feature_names):
        raise ValueError(
            f"Transformed width {X_explain.shape[1]} != "
            f"feature count {len(feature_names)}."
        )

    X_explain_df = pd.DataFrame(X_explain, columns=feature_names)
    X_background_df = pd.DataFrame(X_background, columns=feature_names)

    print("Model loaded successfully.")
    print(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")
    print(f"Features: {len(feature_names)}")
    print(f"Rows available: {len(X_raw)}")
    print(f"Background rows: {len(X_background_df)}")
    print(f"Rows to explain: {len(X_explain_df)}")

    # Path-dependent TreeSHAP matches IsolationForest path-length semantics.
    explainer = shap.TreeExplainer(iforest)
    shap_values = explainer(X_explain)
    values = np.asarray(shap_values.values)

    if values.ndim != 2 or values.shape != (len(X_explain_df), len(feature_names)):
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}")

    additivity = check_additivity(iforest, X_explain, shap_values)
    print(f"Maximum SHAP additivity error: {additivity['max_additivity_error']}")
    print(f"Mean SHAP additivity error: {additivity['mean_additivity_error']}")
    if not additivity["passed"]:
        raise RuntimeError(
            "SHAP additivity check failed "
            f"(max error {additivity['max_additivity_error']} "
            f"> {ADDITIVITY_TOL})."
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)

    print("\nTop 15 global SHAP features:")
    print(importance.head(15).to_string(index=False))

    # Prefer the most anomalous row in the sample (lowest decision score).
    scores = iforest.decision_function(X_explain)
    row_pos = int(np.argmin(scores))
    row = X_explain_df.iloc[[row_pos]]
    row_shap = explainer(row.to_numpy())
    row_values = np.asarray(row_shap.values)
    if row_values.ndim != 2 or row_values.shape != (1, len(feature_names)):
        raise ValueError(f"Unexpected individual SHAP shape: {row_values.shape}")

    individual = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value_scaled": row.iloc[0].to_numpy(),
            "shap_value": row_values[0],
            "absolute_shap": np.abs(row_values[0]),
        }
    ).sort_values("absolute_shap", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUTPUT_DIR / "global_shap_importance.csv", index=False)
    individual.to_csv(
        OUTPUT_DIR / "individual_shap_explanation.csv", index=False
    )

    shap.summary_plot(shap_values, X_explain_df, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.plots.waterfall(row_shap[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "individual_shap_waterfall.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    decision = float(scores[row_pos])
    path_len = float(expected_path_length(iforest, X_explain[[row_pos]])[0])
    print("\nIndividual example (most anomalous in sample):")
    print(f"decision_function: {decision:.6f}")
    print(f"expected path length E (explained): {path_len:.6f}")
    print("Top 10 contributing features:")
    print(individual.head(10).to_string(index=False))

    print("\nIMPORTANT:")
    print(
        "SHAP values explain IsolationForest expected path length on "
        "imputed+scaled features (not raw units, not decision_function)."
    )

    meta = {
        "model": "isolation_forest",
        "base_estimator": "IsolationForest",
        "pipeline_steps": [name for name, _ in pipeline.steps],
        "model_output": "expected_path_length",
        "features_are_scaled": True,
        "data_path": str(data_path),
        "n_features": len(feature_names),
        "n_rows_available": len(X_raw),
        "background_size": len(X_background_df),
        "explanation_size": len(X_explain_df),
        "random_state": random_state,
        "individual_decision_function": decision,
        "individual_expected_path_length": path_len,
        "additivity": additivity,
        "top_features": importance.head(15).to_dict(orient="records"),
        "artifacts": [
            "global_shap_importance.csv",
            "individual_shap_explanation.csv",
            "shap_summary.png",
            "individual_shap_waterfall.png",
            "shap_report.json",
        ],
    }

    with (OUTPUT_DIR / "shap_report.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Explanations saved to: {OUTPUT_DIR}")
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for Isolation Forest."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "features_train.csv",
    )
    parser.add_argument("--background-size", type=int, default=DEFAULT_BACKGROUND_SIZE)
    parser.add_argument("--explanation-size", type=int, default=DEFAULT_EXPLANATION_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(
            data_path=args.data,
            background_size=args.background_size,
            explanation_size=args.explanation_size,
            random_state=args.random_state,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
