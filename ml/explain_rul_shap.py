"""
SHAP explanations for the XGBoost RUL regressor.

The model predicts log1p(RUL hours). SHAP values are therefore in log1p
space (contributions to the transformed target), not hours.

Run:
    python explain_rul_shap.py
    python explain_rul_shap.py --background-size 100 --explanation-size 200
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

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "rul_regressor"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = MODEL_DIR / "shap_explanations"

MODEL_PATH = MODEL_DIR / "xgboost_rul_regressor.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
SELECTED_PATH = MODEL_DIR / "selected_model.json"

TARGET = "time_to_operational_endpoint_hours"
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

    if ARTIFACT_COL not in df.columns:
        raise ValueError(f"Required column missing: {ARTIFACT_COL}")

    artifact = parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)
    target = pd.to_numeric(df[TARGET], errors="coerce")
    keep = (~artifact) & target.notna() & np.isfinite(target) & (target >= 0)

    labeled = df.loc[keep].copy()
    labeled[TARGET] = target.loc[keep].astype(float)

    if labeled.empty:
        raise ValueError("No eligible non-artifact RUL rows available.")

    X = labeled[feature_names].replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Non-finite model features found. "
            "Use the same feature-cleaning rules as model training."
        )

    return X, labeled


def check_additivity(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    shap_explanation,
) -> dict:
    base_value = np.asarray(shap_explanation.base_values)
    shap_sum = np.asarray(shap_explanation.values).sum(axis=1)
    model_output = np.asarray(model.predict(X), dtype=float)

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
    }


def pick_individual_index(labeled: pd.DataFrame, explain_index) -> int:
    """Prefer a near-failure row (low RUL) among the explanation sample."""
    sample_rul = labeled.loc[explain_index, TARGET]
    return int(sample_rul.idxmin())


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
                "Run train_rul_regressor.py first."
            )

    model = joblib.load(MODEL_PATH)
    if not isinstance(model, xgb.XGBRegressor):
        raise TypeError(f"Expected XGBRegressor, got {type(model)}.")

    with FEATURES_PATH.open(encoding="utf-8") as f:
        feature_names = list(json.load(f))

    target_transform = "log1p / expm1"
    if SELECTED_PATH.exists():
        with SELECTED_PATH.open(encoding="utf-8") as f:
            selected = json.load(f)
        target_transform = selected.get("target_transform", target_transform)

    X, labeled = prepare_matrix(data_path, feature_names)

    rng = np.random.default_rng(random_state)
    background_n = min(background_size, len(X))
    explanation_n = min(explanation_size, len(X))

    background_idx = rng.choice(len(X), size=background_n, replace=False)
    explanation_idx = rng.choice(len(X), size=explanation_n, replace=False)

    background = X.iloc[background_idx]
    X_explain = X.iloc[explanation_idx]
    labeled_explain = labeled.iloc[explanation_idx]

    print("Model loaded successfully.")
    print(f"Estimator: {type(model)}")
    print(f"Target transform: {target_transform}")
    print(f"Features: {len(feature_names)}")
    print(f"Eligible RUL rows: {len(X)}")
    print(f"Background rows: {len(background)}")
    print(f"Rows to explain: {len(X_explain)}")

    explainer = shap.TreeExplainer(
        model,
        data=background,
        model_output="raw",
        feature_perturbation="interventional",
    )
    shap_values = explainer(X_explain)
    values = np.asarray(shap_values.values)

    if values.ndim != 2 or values.shape != (len(X_explain), len(feature_names)):
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}")

    additivity = check_additivity(model, X_explain, shap_values)
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

    print("\nTop 15 global SHAP features (log1p space):")
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

    actual_rul = float(labeled_explain.loc[chosen_idx, TARGET])
    pred_log = float(model.predict(row)[0])
    pred_hours = float(max(np.expm1(pred_log), 0.0))

    print("\nIndividual example (near-failure preference):")
    print(f"Actual RUL hours: {actual_rul:.3f}")
    print(f"Predicted log1p(RUL) (explained): {pred_log:.6f}")
    print(f"Predicted RUL hours: {pred_hours:.3f}")
    print("Top 10 contributing features:")
    print(individual.head(10).to_string(index=False))

    print("\nIMPORTANT:")
    print(
        "SHAP values explain log1p(RUL). "
        "They are not additive in hours."
    )

    meta = {
        "model": "rul_regressor",
        "base_estimator": "XGBRegressor",
        "model_output": "raw_log1p_rul",
        "target_transform": target_transform,
        "data_path": str(data_path),
        "n_features": len(feature_names),
        "n_rows_available": len(X),
        "background_size": len(background),
        "explanation_size": len(X_explain),
        "random_state": random_state,
        "individual_actual_rul_hours": actual_rul,
        "individual_predicted_log1p": pred_log,
        "individual_predicted_rul_hours": pred_hours,
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
        description="Generate SHAP explanations for the RUL regressor."
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
