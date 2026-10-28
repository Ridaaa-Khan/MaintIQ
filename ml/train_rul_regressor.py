"""
XGBoost Remaining Useful Life (RUL) regressor.

Target:
    time_to_operational_endpoint_hours
    (failed trajectories / non-artifact rows only)

Input:
    data/processed/features_train.csv
    data/processed/features_val.csv
    data/processed/features_test.csv

Outputs:
    models/rul_regressor/
        xgboost_rul_regressor.joblib
        feature_columns.json
        selected_model.json
        experiment_report.json
        metrics.csv
        feature_importance.csv
        trajectory_metrics.csv
        hyperparameter_search.csv
        predictions_train.csv
        predictions_validation.csv
        predictions_test.csv
        predicted_vs_actual.png

Run:
    ./.venv/Scripts/python.exe train_rul_regressor.py
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from pipeline_config import TARGET_COLUMNS, feature_columns


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "rul_regressor"

TRAIN_FILE = DATA_DIR / "features_train.csv"
VAL_FILE = DATA_DIR / "features_val.csv"
TEST_FILE = DATA_DIR / "features_test.csv"

TARGET = "time_to_operational_endpoint_hours"
ARTIFACT_COL = "is_artifact"

RANDOM_STATE = 42
N_ESTIMATORS = 800
EARLY_STOPPING_ROUNDS = 40

METADATA_COLUMNS = {
    "test_id",
    "bearing_id",
    "timestamp",
    "source_file",
}

FORBIDDEN_FEATURES = (
    METADATA_COLUMNS
    | set(TARGET_COLUMNS)
    | {
        TARGET,
        ARTIFACT_COL,
        "failure_within_24h",
        "isolation_forest_train",
        "_split",
        "RUL_hours",
        "rul_hours",
        "anomaly_score",
        "anomaly_flag",
        "decision_score",
        "anomaly_score_auto",
        "anomaly_flag_auto",
        "decision_score_auto",
        "anomaly_score_1pct",
        "anomaly_flag_1pct",
        "decision_score_1pct",
        "failure_probability",
        "predicted_failure_within_24h",
    }
)


# ============================================================
# UTILITIES
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def make_json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {TARGET, ARTIFACT_COL, "test_id", "bearing_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    params = list(inspect.signature(feature_columns).parameters.values())
    if not params:
        selected = feature_columns()
    elif "column" in params[0].name.lower():
        selected = feature_columns(df.columns)
    else:
        selected = feature_columns(df)

    features = [
        col for col in list(selected)
        if col in df.columns
        and col not in FORBIDDEN_FEATURES
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not features:
        raise ValueError("No eligible numeric model features found.")
    if len(features) != len(set(features)):
        raise ValueError("Duplicate features returned by feature selector.")
    return features


def prepare_rul_data(
    df: pd.DataFrame,
    features: list[str],
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Keep failed-run RUL rows only: non-artifact + finite non-negative target."""
    artifact = parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)
    target = pd.to_numeric(df[TARGET], errors="coerce")
    keep = (~artifact) & target.notna() & np.isfinite(target) & (target >= 0)

    labeled = df.loc[keep].copy()
    labeled[TARGET] = target.loc[keep].astype(float)

    if labeled.empty:
        raise ValueError(f"No eligible RUL rows in {split_name}.")

    X = labeled[features].replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{split_name} features contain NaN/Inf. "
            "Audit the feature pipeline before training."
        )

    y = labeled[TARGET]
    return X, y, labeled


def to_model_target(y: pd.Series | np.ndarray) -> np.ndarray:
    """log1p compresses the long-tailed RUL distribution."""
    return np.log1p(np.asarray(y, dtype=float))


def from_model_target(y_log: np.ndarray) -> np.ndarray:
    """Inverse transform + clip tiny negatives from numeric noise."""
    return np.maximum(np.expm1(np.asarray(y_log, dtype=float)), 0.0)


def rul_sample_weights(
    y: pd.Series | np.ndarray,
    near_failure_hours: float = 72.0,
) -> np.ndarray:
    """
    Up-weight rows closer to the endpoint so late-life errors
    matter more than early-life hundreds of hours.
    """
    hours = np.asarray(y, dtype=float)
    return 1.0 + (near_failure_hours / (1.0 + np.maximum(hours, 0.0)))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    payload = {
        "mae_hours": float(mean_absolute_error(y_true, y_pred)),
        "rmse_hours": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae_hours": float(np.median(np.abs(y_true - y_pred))),
    }
    for horizon, key in ((24.0, "mae_last_24h"), (72.0, "mae_last_72h")):
        mask = y_true <= horizon
        payload[key] = (
            float(mean_absolute_error(y_true[mask], y_pred[mask]))
            if int(mask.sum()) >= 5
            else None
        )
    return payload


def build_param_grid() -> list[dict]:
    grid = []
    for max_depth, min_child_weight, learning_rate, reg_lambda in (
        (2, 10, 0.03, 3.0),
        (2, 20, 0.05, 5.0),
        (3, 5, 0.03, 1.0),
        (3, 10, 0.03, 3.0),
        (3, 20, 0.05, 5.0),
        (4, 10, 0.03, 3.0),
    ):
        grid.append({
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "learning_rate": learning_rate,
            "reg_lambda": reg_lambda,
            "reg_alpha": 0.2,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
        })
    return grid


def fit_regressor(
    params: dict,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_val: pd.DataFrame,
    y_val_log: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=N_ESTIMATORS,
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    fit_kwargs = {
        "eval_set": [(X_val, y_val_log)],
        "verbose": False,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train_log, **fit_kwargs)
    return model


def select_model_on_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[XGBRegressor, dict, list[dict]]:
    """Choose hyperparameters by validation MAE on the original hour scale."""
    y_train_log = to_model_target(y_train)
    y_val_log = to_model_target(y_val)
    sample_weight = rul_sample_weights(y_train)
    grid = build_param_grid()
    log(f"\nHyperparameter candidates: {len(grid)}")

    trials = []
    best_model = None
    best_trial = None

    for index, params in enumerate(grid, start=1):
        model = fit_regressor(
            params,
            X_train,
            y_train_log,
            X_val,
            y_val_log,
            sample_weight=sample_weight,
        )
        val_pred = from_model_target(model.predict(X_val))
        metrics = regression_metrics(y_val, val_pred)
        best_iteration = getattr(model, "best_iteration", None)
        selection_mae = metrics["mae_last_72h"]
        if selection_mae is None:
            selection_mae = metrics["mae_hours"]

        trial = {
            "trial": index,
            "params": params,
            "val_mae_hours": metrics["mae_hours"],
            "val_mae_last_72h": metrics["mae_last_72h"],
            "val_mae_last_24h": metrics["mae_last_24h"],
            "val_rmse_hours": metrics["rmse_hours"],
            "val_r2": metrics["r2"],
            "selection_mae_hours": float(selection_mae),
            "best_iteration": (
                int(best_iteration) if best_iteration is not None else None
            ),
        }
        trials.append(trial)

        if (
            best_trial is None
            or trial["selection_mae_hours"] < best_trial["selection_mae_hours"]
            or (
                trial["selection_mae_hours"] == best_trial["selection_mae_hours"]
                and trial["val_mae_hours"] < best_trial["val_mae_hours"]
            )
        ):
            best_trial = trial
            best_model = model

        if index % 8 == 0 or index == len(grid):
            log(
                f"  searched {index}/{len(grid)} | "
                f"best val last-72h MAE so far = "
                f"{best_trial['selection_mae_hours']:.3f} h"
            )

    assert best_model is not None and best_trial is not None
    return best_model, best_trial, trials


def predict_rul_hours(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return from_model_target(model.predict(X))


def evaluate_split(
    model: XGBRegressor,
    split_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_mean_baseline: float,
) -> dict:
    predictions = predict_rul_hours(model, X)
    metrics = regression_metrics(y, predictions)
    baseline_pred = np.full(len(y), train_mean_baseline, dtype=float)
    baseline = regression_metrics(y, baseline_pred)

    payload = {
        "split": split_name,
        "rows": int(len(y)),
        **metrics,
        "baseline_mae_hours": baseline["mae_hours"],
        "baseline_rmse_hours": baseline["rmse_hours"],
        "baseline_r2": baseline["r2"],
        "baseline_mae_last_72h": baseline["mae_last_72h"],
        "baseline_mae_last_24h": baseline["mae_last_24h"],
        "mae_improvement_vs_mean_baseline": (
            baseline["mae_hours"] - metrics["mae_hours"]
        ),
        "mae_last72_improvement_vs_mean_baseline": (
            None
            if baseline["mae_last_72h"] is None or metrics["mae_last_72h"] is None
            else baseline["mae_last_72h"] - metrics["mae_last_72h"]
        ),
    }

    log(f"\n{split_name.upper()} METRICS")
    log("-" * 50)
    log(json.dumps(make_json_safe(payload), indent=2))
    return payload, predictions


def export_predictions(
    labeled: pd.DataFrame,
    y_true: pd.Series,
    predictions: np.ndarray,
    split_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    cols = [
        col for col in (
            "test_id", "bearing_id", "timestamp", "source_file"
        )
        if col in labeled.columns
    ]
    output = labeled[cols].copy()
    output["actual_rul_hours"] = y_true.to_numpy()
    output["predicted_rul_hours"] = predictions
    output["absolute_error_hours"] = np.abs(
        output["actual_rul_hours"] - output["predicted_rul_hours"]
    )
    output.to_csv(
        output_dir / f"predictions_{split_name}.csv",
        index=False,
    )
    return output


def trajectory_metrics(predictions: pd.DataFrame, split_name: str) -> pd.DataFrame:
    records = []
    for (test_id, bearing_id), group in predictions.groupby(
        ["test_id", "bearing_id"], sort=True
    ):
        metrics = regression_metrics(
            group["actual_rul_hours"],
            group["predicted_rul_hours"],
        )
        records.append({
            "split": split_name,
            "test_id": test_id,
            "bearing_id": int(bearing_id),
            "rows": int(len(group)),
            **metrics,
            "actual_rul_min": float(group["actual_rul_hours"].min()),
            "actual_rul_max": float(group["actual_rul_hours"].max()),
        })
    return pd.DataFrame(records)


def save_pred_vs_actual_plot(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, (split_name, (y_true, y_pred)) in zip(axes, datasets.items()):
        ax.scatter(y_true, y_pred, s=8, alpha=0.35)
        limit = max(float(np.max(y_true)), float(np.max(y_pred)), 1.0)
        ax.plot([0, limit], [0, limit], color="black", linewidth=1)
        ax.set_title(split_name)
        ax.set_xlabel("Actual RUL (hours)")
        ax.set_ylabel("Predicted RUL (hours)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("RUL: predicted vs actual")
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_actual.png", dpi=160)
    plt.close(fig)


def verify_regressor(
    val_metrics: dict,
    test_metrics: dict,
    traj_df: pd.DataFrame,
) -> dict:
    checks = {
        "val_last72_beats_mean_baseline": (
            val_metrics.get("mae_last_72h") is not None
            and val_metrics.get("baseline_mae_last_72h") is not None
            and val_metrics["mae_last_72h"]
            < val_metrics["baseline_mae_last_72h"]
        ),
        "test_last72_beats_mean_baseline": (
            test_metrics.get("mae_last_72h") is not None
            and test_metrics.get("baseline_mae_last_72h") is not None
            and test_metrics["mae_last_72h"]
            < test_metrics["baseline_mae_last_72h"]
        ),
        "val_last72_mae_under_40h": (
            val_metrics.get("mae_last_72h") is not None
            and val_metrics["mae_last_72h"] < 40.0
        ),
        "test_last72_mae_under_80h": (
            test_metrics.get("mae_last_72h") is not None
            and test_metrics["mae_last_72h"] < 80.0
        ),
        "test_last24_mae_under_20h": (
            test_metrics.get("mae_last_24h") is not None
            and test_metrics["mae_last_24h"] < 20.0
        ),
        "all_trajectories_reported": (
            not traj_df.empty and traj_df["rows"].gt(0).all()
        ),
    }
    checks["all_critical_passed"] = all(checks.values())
    return checks


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("XGBOOST RUL REGRESSOR")
    log("=" * 70)
    log(f"XGBoost version: {xgboost.__version__}")
    log(f"Target: {TARGET}")

    train_df = read_dataset(TRAIN_FILE)
    val_df = read_dataset(VAL_FILE)
    test_df = read_dataset(TEST_FILE)

    features = get_feature_columns(train_df)
    for name, df in (("validation", val_df), ("test", test_df)):
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} is missing features: {sorted(missing)}"
            )

    X_train, y_train, labeled_train = prepare_rul_data(
        train_df, features, "train"
    )
    X_val, y_val, labeled_val = prepare_rul_data(
        val_df, features, "validation"
    )
    X_test, y_test, labeled_test = prepare_rul_data(
        test_df, features, "test"
    )

    log("\nELIGIBLE RUL ROWS")
    log("-" * 50)
    for name, labeled, y in (
        ("Train", labeled_train, y_train),
        ("Validation", labeled_val, y_val),
        ("Test", labeled_test, y_test),
    ):
        trajs = (
            labeled[["test_id", "bearing_id"]]
            .drop_duplicates()
            .to_dict("records")
        )
        log(
            f"{name}: {len(labeled):,} rows | "
            f"RUL {y.min():.2f}-{y.max():.2f} h | "
            f"trajectories={trajs}"
        )

    train_mean_baseline = float(y_train.mean())
    log(f"\nFeatures: {len(features)}")
    log(f"Train-mean baseline: {train_mean_baseline:.3f} h")

    model, best_trial, trials = select_model_on_validation(
        X_train, y_train, X_val, y_val
    )

    log("\nBEST HYPERPARAMETERS (VALIDATION MAE)")
    log("-" * 50)
    log(json.dumps(make_json_safe(best_trial), indent=2))

    train_metrics, train_pred = evaluate_split(
        model, "train", X_train, y_train, train_mean_baseline
    )
    val_metrics, val_pred = evaluate_split(
        model, "validation", X_val, y_val, train_mean_baseline
    )
    test_metrics, test_pred = evaluate_split(
        model, "test", X_test, y_test, train_mean_baseline
    )

    pred_frames = {
        "train": export_predictions(
            labeled_train, y_train, train_pred, "train", OUTPUT_DIR
        ),
        "validation": export_predictions(
            labeled_val, y_val, val_pred, "validation", OUTPUT_DIR
        ),
        "test": export_predictions(
            labeled_test, y_test, test_pred, "test", OUTPUT_DIR
        ),
    }

    traj_df = pd.concat(
        [
            trajectory_metrics(pred_frames["train"], "train"),
            trajectory_metrics(pred_frames["validation"], "validation"),
            trajectory_metrics(pred_frames["test"], "test"),
        ],
        ignore_index=True,
    )
    traj_df.to_csv(OUTPUT_DIR / "trajectory_metrics.csv", index=False)

    log("\nTRAJECTORY METRICS")
    log("-" * 50)
    log(traj_df.to_string(index=False))

    save_pred_vs_actual_plot(
        {
            "train": (y_train.to_numpy(), train_pred),
            "validation": (y_val.to_numpy(), val_pred),
            "test": (y_test.to_numpy(), test_pred),
        },
        OUTPUT_DIR,
    )

    verification = verify_regressor(val_metrics, test_metrics, traj_df)

    joblib.dump(model, OUTPUT_DIR / "xgboost_rul_regressor.joblib")

    with (OUTPUT_DIR / "feature_columns.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(features, f, indent=2)

    selected_payload = {
        "model_path": "models/rul_regressor/xgboost_rul_regressor.joblib",
        "feature_columns_path": "models/rul_regressor/feature_columns.json",
        "target": TARGET,
        "target_transform": "log1p / expm1",
        "unit": "hours",
        "best_trial": best_trial,
        "train_mean_baseline_hours": train_mean_baseline,
        "verification": verification,
    }

    with (OUTPUT_DIR / "selected_model.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            make_json_safe(selected_payload),
            f,
            indent=2,
            allow_nan=False,
        )

    metrics_df = pd.DataFrame(
        [train_metrics, val_metrics, test_metrics]
    )
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "trial": trial["trial"],
                "val_mae_hours": trial["val_mae_hours"],
                "val_rmse_hours": trial["val_rmse_hours"],
                "val_r2": trial["val_r2"],
                "best_iteration": trial["best_iteration"],
                **trial["params"],
            }
            for trial in trials
        ]
    ).sort_values("val_mae_hours").to_csv(
        OUTPUT_DIR / "hyperparameter_search.csv",
        index=False,
    )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    report = {
        "algorithm": "XGBRegressor",
        "xgboost_version": xgboost.__version__,
        "target": TARGET,
        "target_transform": "log1p while training; metrics on original hours",
        "unit": "hours",
        "feature_count": len(features),
        "features": features,
        "best_trial": best_trial,
        "train_mean_baseline_hours": train_mean_baseline,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "verification": verification,
        "test_used_for_tuning": False,
        "mitigations_applied": [
            "Train only on finite non-artifact RUL labels",
            "log1p target transform for long-tailed RUL",
            "Near-failure sample weights (higher weight when RUL is small)",
            "Two failed train trajectories (T1-B3 and T2-B1)",
            "Validation MAE hyperparameter search",
            "Early stopping on validation",
            "Trajectory-level MAE/RMSE/R2 plus last-24h/72h MAE",
            "Compare against train-mean baseline",
        ],
        "limitations": [
            "RUL exists only for the four documented failed trajectories.",
            "Train contains two failed trajectories "
            "(1st_test bearing 3 and 2nd_test bearing 1); "
            "absolute hour error on a held-out long run can still be large.",
            "Endpoint is an operational last-non-artifact timestamp, "
            "not a separate physical failure clock.",
            "Validation is used for early stopping and hyperparameter search.",
            "Test metrics are held-out and were not used for tuning.",
        ],
    }

    with (OUTPUT_DIR / "experiment_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            make_json_safe(report),
            f,
            indent=2,
            allow_nan=False,
        )

    log("\n" + "=" * 70)
    log("VERIFICATION")
    log("=" * 70)
    for key, value in verification.items():
        log(f"  {key}: {value}")

    if not verification["all_critical_passed"]:
        log(
            "\nWARNING: some verification checks failed. "
            "Inspect trajectory_metrics.csv and predicted_vs_actual.png."
        )
    else:
        log("\nAll critical verification checks passed.")

    log(f"\nOutputs saved to: {OUTPUT_DIR}")
    log(f"Elapsed: {time.perf_counter() - started:.1f} seconds")
    log("RUL REGRESSOR EXPERIMENT FINISHED")


if __name__ == "__main__":
    main()
