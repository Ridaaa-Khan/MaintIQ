"""
SHAP explanations for the failure-within-24h XGBoost classifier.

Explains the underlying XGBClassifier (raw margin / log-odds), not the
sigmoid calibration layer. Calibrated probabilities are reported alongside
for context only.

Run:
    python explain_failure_shap.py
    python explain_failure_shap.py --background-size 100 --explanation-size 200
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
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "failure_classifier"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = MODEL_DIR / "shap_explanations"

CALIBRATED_MODEL_PATH = MODEL_DIR / "xgboost_failure_classifier.joblib"
RAW_MODEL_PATH = MODEL_DIR / "xgboost_failure_classifier_raw.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
REPORT_PATH = MODEL_DIR / "experiment_report.json"

TARGET = "failure_within_24h"
ARTIFACT_COL = "is_artifact"

DEFAULT_BACKGROUND_SIZE = 200
DEFAULT_EXPLANATION_SIZE = 500
DEFAULT_RANDOM_STATE = 42
ADDITIVITY_TOL = 1e-5


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


def unwrap_estimator(estimator):
    """Unwrap sklearn calibration wrappers to the fitted base model."""
    seen: set[int] = set()

    while id(estimator) not in seen:
        seen.add(id(estimator))

        if isinstance(estimator, FrozenEstimator):
            estimator = estimator.estimator
            continue

        if isinstance(estimator, CalibratedClassifierCV):
            calibrated = getattr(estimator, "calibrated_classifiers_", None)
            if not calibrated:
                raise ValueError("CalibratedClassifierCV is not fitted.")
            if len(calibrated) != 1:
                raise ValueError(
                    f"Found {len(calibrated)} calibrated estimators; "
                    "expected a single prefit calibrator."
                )
            estimator = calibrated[0]
            continue

        underlying = getattr(estimator, "estimator", None)
        if underlying is None:
            underlying = getattr(estimator, "base_estimator", None)
        if underlying is not None and underlying is not estimator:
            estimator = underlying
            continue

        break

    return estimator


def load_models() -> tuple[object, xgb.XGBClassifier, str]:
    """Return (calibrated_or_none, base_xgb, source_label)."""
    for path in (FEATURES_PATH, REPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    calibrated = None
    if CALIBRATED_MODEL_PATH.exists():
        calibrated = joblib.load(CALIBRATED_MODEL_PATH)

    if RAW_MODEL_PATH.exists():
        base = joblib.load(RAW_MODEL_PATH)
        source = "raw_joblib"
    elif calibrated is not None:
        base = unwrap_estimator(calibrated)
        source = "unwrapped_calibrated"
    else:
        raise FileNotFoundError(
            "Neither calibrated nor raw classifier joblib found. "
            "Run train_failure_classifier.py first."
        )

    if not isinstance(base, xgb.XGBClassifier):
        raise TypeError(
            "Expected XGBClassifier after load/unwrap, "
            f"got {type(base)}."
        )

    return calibrated, base, source


def prepare_matrix(
    data_path: Path,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not data_path.exists():
        raise FileNotFoundError(f"Required data not found: {data_path}")

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column missing: {TARGET}")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            f"Model features missing from data: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}"
        )

    target = pd.to_numeric(df[TARGET], errors="coerce")
    keep = target.isin([0, 1])
    if ARTIFACT_COL in df.columns:
        keep &= ~parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)

    labeled = df.loc[keep].copy()
    labeled[TARGET] = target.loc[keep].astype(int)

    if labeled.empty:
        raise ValueError("No labelled non-artifact rows available.")

    X = labeled[feature_names].replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Non-finite model features found. "
            "Use the same feature-cleaning rules as model training."
        )

    return X, labeled


def check_additivity(
    base_model: xgb.XGBClassifier,
    X: pd.DataFrame,
    shap_explanation,
) -> dict:
    base_value = np.asarray(shap_explanation.base_values)
    shap_sum = np.asarray(shap_explanation.values).sum(axis=1)
    raw_output = base_model.predict(X, output_margin=True)

    if base_value.ndim == 0:
        reconstructed = float(base_value) + shap_sum
    else:
        reconstructed = base_value.reshape(-1) + shap_sum

    errors = np.abs(np.asarray(raw_output, dtype=float) - reconstructed)
    report = {
        "max_additivity_error": float(errors.max()),
        "mean_additivity_error": float(errors.mean()),
        "tolerance": ADDITIVITY_TOL,
        "passed": bool(errors.max() <= ADDITIVITY_TOL),
    }
    return report


def pick_individual_index(labeled: pd.DataFrame, explain_index) -> int:
    """Prefer a positive-class row among the explanation sample."""
    sample_labels = labeled.loc[explain_index, TARGET]
    positive = sample_labels[sample_labels == 1]
    if not positive.empty:
        return int(positive.index[0])
    return int(explain_index[0])


def save_plots_and_tables(
    shap_values,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    row: pd.DataFrame,
    row_shap,
    individual: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    importance.to_csv(OUTPUT_DIR / "global_shap_importance.csv", index=False)
    individual.to_csv(
        OUTPUT_DIR / "individual_shap_explanation.csv", index=False
    )

    shap.summary_plot(shap_values, X_explain, max_display=20, show=False)
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


def run(
    data_path: Path,
    background_size: int,
    explanation_size: int,
    random_state: int,
) -> dict:
    calibrated, base_model, source = load_models()

    with FEATURES_PATH.open(encoding="utf-8") as f:
        feature_names = list(json.load(f))

    X, labeled = prepare_matrix(data_path, feature_names)

    rng = np.random.default_rng(random_state)
    background_n = min(background_size, len(X))
    explanation_n = min(explanation_size, len(X))

    background_idx = rng.choice(len(X), size=background_n, replace=False)

    # Stratify explanation sample so rare positives are represented when present.
    pos_positions = np.flatnonzero(labeled[TARGET].to_numpy() == 1)
    neg_positions = np.flatnonzero(labeled[TARGET].to_numpy() == 0)
    if len(pos_positions) > 0 and explanation_n >= 2:
        n_pos = min(max(1, explanation_n // 10), len(pos_positions), explanation_n)
        n_neg = min(explanation_n - n_pos, len(neg_positions))
        chosen_pos = rng.choice(pos_positions, size=n_pos, replace=False)
        chosen_neg = (
            rng.choice(neg_positions, size=n_neg, replace=False)
            if n_neg > 0
            else np.array([], dtype=int)
        )
        explanation_idx = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(explanation_idx)
    else:
        explanation_idx = rng.choice(len(X), size=explanation_n, replace=False)

    background = X.iloc[background_idx]
    X_explain = X.iloc[explanation_idx]
    labeled_explain = labeled.iloc[explanation_idx]

    print("Model loaded successfully.")
    print(f"Base estimator source: {source}")
    print(f"Unwrapped estimator: {type(base_model)}")
    print(f"Features: {len(feature_names)}")
    print(f"Training rows available: {len(X)}")
    print(f"Background rows: {len(background)}")
    print(f"Rows to explain: {len(X_explain)}")

    explainer = shap.TreeExplainer(
        base_model,
        data=background,
        model_output="raw",
        feature_perturbation="interventional",
    )
    shap_values = explainer(X_explain)
    values = np.asarray(shap_values.values)

    if values.ndim != 2 or values.shape != (len(X_explain), len(feature_names)):
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}")

    additivity = check_additivity(base_model, X_explain, shap_values)
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

    chosen_idx = pick_individual_index(labeled_explain, labeled_explain.index)
    row = X_explain.loc[[chosen_idx]]
    row_shap = explainer(row)
    row_values = np.asarray(row_shap.values)
    if row_values.ndim != 2 or row_values.shape != (1, len(feature_names)):
        raise ValueError(f"Unexpected individual SHAP shape: {row_values.shape}")

    individual = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": row.iloc[0].to_numpy(),
            "shap_value": row_values[0],
            "absolute_shap": np.abs(row_values[0]),
        }
    ).sort_values("absolute_shap", ascending=False)

    save_plots_and_tables(
        shap_values,
        X_explain,
        feature_names,
        row,
        row_shap,
        individual,
        importance,
    )

    actual_label = int(labeled_explain.loc[chosen_idx, TARGET])
    raw_margin = float(base_model.predict(row, output_margin=True)[0])
    calibrated_prob = None
    if calibrated is not None:
        calibrated_prob = float(calibrated.predict_proba(row)[0, 1])

    print("\nIndividual example:")
    print(f"Actual label: {actual_label}")
    print(f"Raw margin (explained): {raw_margin:.6f}")
    if calibrated_prob is not None:
        print(f"Calibrated failure probability: {calibrated_prob:.6f}")
    print("Top 10 contributing features:")
    print(individual.head(10).to_string(index=False))

    print("\nIMPORTANT:")
    print(
        "SHAP values explain the underlying XGBClassifier raw margin, "
        "not the calibration transformation."
    )

    meta = {
        "model": "failure_classifier",
        "base_estimator": "XGBClassifier",
        "base_source": source,
        "model_output": "raw_margin",
        "explains_calibration": False,
        "data_path": str(data_path),
        "n_features": len(feature_names),
        "n_rows_available": len(X),
        "background_size": len(background),
        "explanation_size": len(X_explain),
        "random_state": random_state,
        "individual_actual_label": actual_label,
        "individual_raw_margin": raw_margin,
        "individual_calibrated_probability": calibrated_prob,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "shap_report.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Explanations saved to: {OUTPUT_DIR}")
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for the failure classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "features_train.csv",
        help="Feature CSV used for background / explanation samples.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=DEFAULT_BACKGROUND_SIZE,
    )
    parser.add_argument(
        "--explanation-size",
        type=int,
        default=DEFAULT_EXPLANATION_SIZE,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )
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
