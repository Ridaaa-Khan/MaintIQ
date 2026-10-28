"""
Audit the trained RUL regressor against a train-mean baseline
and regenerate trajectory time-series plots.

Run after train_rul_regressor.py:
    ./.venv/Scripts/python.exe audit_rul_regressor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "rul_regressor"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "time_to_operational_endpoint_hours"
ARTIFACT_COL = "is_artifact"


def parse_boolean(series: pd.Series) -> pd.Series:
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
        raise ValueError("Unexpected boolean values in is_artifact.")
    return result.astype(bool)


def metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae_hours": float(mean_absolute_error(actual, predicted)),
        "rmse_hours": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def report(name: str, actual, predicted, baseline_value: float) -> dict:
    baseline = np.full(len(actual), baseline_value, dtype=float)
    model_metrics = metrics(actual, predicted)
    baseline_metrics = metrics(actual, baseline)

    print(f"\n{name}")
    print(
        "XGBoost: "
        f"MAE={model_metrics['mae_hours']:.3f} h, "
        f"RMSE={model_metrics['rmse_hours']:.3f} h, "
        f"R2={model_metrics['r2']:.4f}"
    )
    print(
        "Mean baseline: "
        f"MAE={baseline_metrics['mae_hours']:.3f} h, "
        f"RMSE={baseline_metrics['rmse_hours']:.3f} h, "
        f"R2={baseline_metrics['r2']:.4f}"
    )
    return {
        "split": name.lower(),
        "model": model_metrics,
        "mean_baseline": baseline_metrics,
        "mae_improvement_hours": (
            baseline_metrics["mae_hours"] - model_metrics["mae_hours"]
        ),
    }


def main() -> None:
    model_path = MODEL_DIR / "xgboost_rul_regressor.joblib"
    val_path = MODEL_DIR / "predictions_validation.csv"
    test_path = MODEL_DIR / "predictions_test.csv"
    train_path = DATA_DIR / "features_train.csv"

    for path in (model_path, val_path, test_path, train_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run train_rul_regressor.py first."
            )

    _ = joblib.load(model_path)  # ensure the artifact loads

    train = pd.read_csv(train_path)
    val_pred = pd.read_csv(val_path, parse_dates=["timestamp"])
    test_pred = pd.read_csv(test_path, parse_dates=["timestamp"])

    train_target = pd.to_numeric(train[TARGET], errors="coerce")
    artifact = parse_boolean(train[ARTIFACT_COL])
    eligible = (
        train_target.notna()
        & np.isfinite(train_target)
        & (~artifact)
        & (train_target >= 0)
    )
    train_y = train_target.loc[eligible]

    if train_y.empty:
        raise ValueError("No eligible training targets for baseline.")

    training_mean = float(train_y.mean())
    print(f"Training-target mean baseline: {training_mean:.3f} hours")

    summaries = [
        report(
            "Validation",
            val_pred["actual_rul_hours"],
            val_pred["predicted_rul_hours"],
            training_mean,
        ),
        report(
            "Test",
            test_pred["actual_rul_hours"],
            test_pred["predicted_rul_hours"],
            training_mean,
        ),
    ]

    with (MODEL_DIR / "audit_baseline_comparison.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "train_mean_baseline_hours": training_mean,
                "comparisons": summaries,
            },
            f,
            indent=2,
        )

    # Plot actual and predicted RUL against time for each trajectory.
    for split, df in [("validation", val_pred), ("test", test_pred)]:
        required = {"test_id", "bearing_id", "timestamp"}
        if not required.issubset(df.columns):
            print(f"Skipping {split} plot: trajectory metadata missing.")
            continue

        for (test_id, bearing_id), group in df.groupby(
            ["test_id", "bearing_id"]
        ):
            group = group.dropna(subset=["timestamp"]).sort_values(
                "timestamp"
            )
            if group.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                group["timestamp"],
                group["actual_rul_hours"],
                label="Actual RUL",
            )
            ax.plot(
                group["timestamp"],
                group["predicted_rul_hours"],
                label="Predicted RUL",
            )
            ax.set_title(f"{split}: {test_id} — bearing {bearing_id}")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Hours until operational endpoint")
            ax.legend()
            fig.tight_layout()

            filename = f"{split}_{test_id}_{bearing_id}.png"
            fig.savefig(MODEL_DIR / filename, dpi=150)
            plt.close(fig)

    print(f"\nAudit outputs saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
