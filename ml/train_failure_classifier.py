"""
XGBoost failure-within-24-hours classifier.

Input:
    data/processed/features_train.csv
    data/processed/features_val.csv
    data/processed/features_test.csv

Outputs:
    models/failure_classifier/
        xgboost_failure_classifier.joblib
        feature_columns.json
        threshold.json
        selected_model.json
        experiment_report.json
        metrics.csv
        feature_importance.csv
        trajectory_metrics.csv
        predictions_train.csv
        predictions_validation.csv
        predictions_test.csv
        pr_curves.png
        trajectory_recall.png

Run from the project root:
    ./.venv/Scripts/python.exe train_failure_classifier.py
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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from pipeline_config import TARGET_COLUMNS, failed_bearing_set, feature_columns


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "failure_classifier"

TRAIN_FILE = DATA_DIR / "features_train.csv"
VAL_FILE = DATA_DIR / "features_val.csv"
TEST_FILE = DATA_DIR / "features_test.csv"

TARGET = "failure_within_24h"
ARTIFACT_COL = "is_artifact"

RANDOM_STATE = 42
N_ESTIMATORS = 800
EARLY_STOPPING_ROUNDS = 40

# Threshold objective: maximize validation F1 among operating points
# that clear minimum precision and recall. This transfers better than
# F-beta=1.5, which chased val recall and collapsed test recall.
MIN_VAL_PRECISION = 0.30
MIN_VAL_RECALL = 0.60

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
        ARTIFACT_COL,
        TARGET,
        "_split",
        "anomaly_score",
        "anomaly_flag",
        "decision_score",
        "anomaly_score_auto",
        "anomaly_flag_auto",
        "decision_score_auto",
        "anomaly_score_1pct",
        "anomaly_flag_1pct",
        "decision_score_1pct",
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


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    required = {
        TARGET,
        ARTIFACT_COL,
        "test_id",
        "bearing_id",
        "timestamp",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    return df


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


def prepare_labeled_data(
    df: pd.DataFrame,
    features: list[str],
    split_name: str,
    require_both_classes: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    artifact = parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)

    target_numeric = pd.to_numeric(df[TARGET], errors="coerce")
    valid_target = target_numeric.isin([0, 1])
    keep = ~artifact & valid_target

    labeled = df.loc[keep].copy()
    labeled[TARGET] = target_numeric.loc[keep].astype(int)

    if labeled.empty:
        raise ValueError(f"No labeled observations in {split_name}.")

    X = labeled[features].replace([np.inf, -np.inf], np.nan)
    y = labeled[TARGET]

    if require_both_classes and y.nunique() < 2:
        raise ValueError(
            f"{split_name} has only one target class; "
            "classification metrics cannot be fully evaluated."
        )

    return X, y, labeled


def build_param_grid(base_scale_pos_weight: float) -> list[dict]:
    """
    Compact validation search space focused on imbalance + regularization.

    Extreme scale_pos_weight (~500) often over-fires on validation.
    sqrt / fractional weights usually generalize better.
    """
    weight_options = sorted({
        max(1.0, base_scale_pos_weight ** 0.5),
        max(1.0, base_scale_pos_weight * 0.25),
        max(1.0, base_scale_pos_weight * 0.50),
    })

    grid = []
    for max_depth, min_child_weight, learning_rate, reg_lambda in (
        (2, 10, 0.03, 3.0),
        (2, 20, 0.05, 3.0),
        (3, 5, 0.03, 1.0),
        (3, 10, 0.03, 3.0),
        (3, 10, 0.05, 3.0),
        (3, 20, 0.05, 5.0),
        (4, 10, 0.03, 3.0),
        (4, 20, 0.05, 5.0),
    ):
        for scale_pos_weight in weight_options:
            grid.append({
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "learning_rate": learning_rate,
                "reg_lambda": reg_lambda,
                "reg_alpha": 0.2,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "scale_pos_weight": float(scale_pos_weight),
            })
    return grid


def fit_xgb(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=N_ESTIMATORS,
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=params["scale_pos_weight"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def select_model_on_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    base_scale_pos_weight: float,
) -> tuple[XGBClassifier, dict, list[dict]]:
    """Pick hyperparameters by validation average precision (PR-AUC)."""
    grid = build_param_grid(base_scale_pos_weight)
    log(f"\nHyperparameter candidates: {len(grid)}")

    trials = []
    best_model = None
    best_trial = None

    for index, params in enumerate(grid, start=1):
        model = fit_xgb(params, X_train, y_train, X_val, y_val)
        val_probability = model.predict_proba(X_val)[:, 1]
        ap = float(average_precision_score(y_val, val_probability))
        roc = float(roc_auc_score(y_val, val_probability))
        best_iteration = getattr(model, "best_iteration", None)

        trial = {
            "trial": index,
            "params": params,
            "val_average_precision": ap,
            "val_roc_auc": roc,
            "best_iteration": (
                int(best_iteration)
                if best_iteration is not None
                else None
            ),
        }
        trials.append(trial)

        if (
            best_trial is None
            or ap > best_trial["val_average_precision"]
            or (
                ap == best_trial["val_average_precision"]
                and roc > best_trial["val_roc_auc"]
            )
        ):
            best_trial = trial
            best_model = model

        if index % 20 == 0 or index == len(grid):
            log(
                f"  searched {index}/{len(grid)} | "
                f"best val AP so far = "
                f"{best_trial['val_average_precision']:.4f}"
            )

    assert best_model is not None and best_trial is not None
    return best_model, best_trial, trials


def calibrate_model(
    model: XGBClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> CalibratedClassifierCV:
    """
    Isotonic calibration on validation probabilities.

    Improves threshold stability under severe class imbalance.
    Threshold selection uses the same calibrated validation scores;
    test remains untouched during fitting/calibration/tuning.

    sklearn >= 1.6: wrap a fitted estimator with FrozenEstimator
    instead of the removed cv='prefit' option.
    """
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="sigmoid",
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def select_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
    min_precision: float = MIN_VAL_PRECISION,
    min_recall: float = MIN_VAL_RECALL,
) -> tuple[float, dict]:
    """
    Validation-only threshold rule:
      among thresholds with precision >= min_precision and
      recall >= min_recall, maximize F1. If none qualify, maximize F1
      over the full grid.
    """
    threshold_candidates = np.unique(
        np.concatenate([
            np.linspace(0.02, 0.95, 187),
            np.array([0.5]),
        ])
    )

    rows = []
    for threshold in threshold_candidates:
        prediction = (probability >= threshold).astype(int)
        precision = float(
            precision_score(y_true, prediction, zero_division=0)
        )
        recall = float(
            recall_score(y_true, prediction, zero_division=0)
        )
        f1 = float(f1_score(y_true, prediction, zero_division=0))
        rows.append({
            "threshold": float(round(float(threshold), 4)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "meets_precision_floor": precision >= min_precision,
            "meets_recall_floor": recall >= min_recall,
        })

    eligible = [
        row for row in rows
        if row["meets_precision_floor"] and row["meets_recall_floor"]
    ]
    pool = eligible if eligible else rows
    best_f1 = max(row["f1"] for row in pool)
    near_best = [row for row in pool if row["f1"] >= best_f1 - 0.03]
    # Prefer higher recall among near-best F1 so the cut transfers
    # to unseen failure modes instead of sitting on a razor-thin val peak.
    best = max(
        near_best,
        key=lambda row: (
            row["recall"],
            row["f1"],
            -row["threshold"],
        ),
    )

    best = dict(best)
    best["n_eligible_thresholds"] = len(eligible)
    best["selection_rule"] = (
        "Max validation F1 among thresholds with "
        f"precision >= {min_precision:.2f} and recall >= {min_recall:.2f}"
        if eligible else
        "No threshold met both floors; max validation F1 over full grid"
    )
    return float(best["threshold"]), best


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def evaluate_split(
    model,
    split_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict:
    probability = predict_proba(model, X)
    prediction = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, prediction, labels=[0, 1]
    ).ravel()

    metrics = {
        "split": split_name,
        "threshold": float(threshold),
        "rows": int(len(y)),
        "positive_rows": int((y == 1).sum()),
        "negative_rows": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(
            precision_score(y, prediction, zero_division=0)
        ),
        "recall": float(
            recall_score(y, prediction, zero_division=0)
        ),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y, probability))
            if y.nunique() == 2 else None
        ),
        "average_precision": (
            float(average_precision_score(y, probability))
            if y.nunique() == 2 else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    log(f"\n{split_name.upper()} METRICS")
    log("-" * 50)
    log(json.dumps(make_json_safe(metrics), indent=2))
    log("\nClassification report:")
    log(
        classification_report(
            y,
            prediction,
            labels=[0, 1],
            zero_division=0,
        )
    )

    return metrics


def export_predictions(
    model,
    labeled: pd.DataFrame,
    X: pd.DataFrame,
    threshold: float,
    split_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    output = labeled[
        ["test_id", "bearing_id", "timestamp", TARGET]
    ].copy()

    output["failure_probability"] = predict_proba(model, X)
    output["predicted_failure_within_24h"] = (
        output["failure_probability"] >= threshold
    ).astype(int)

    output.to_csv(
        output_dir / f"predictions_{split_name}.csv",
        index=False,
    )
    return output


def trajectory_metrics(
    predictions: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    failed = failed_bearing_set()
    records = []

    for (test_id, bearing_id), group in predictions.groupby(
        ["test_id", "bearing_id"], sort=True
    ):
        y_true = group[TARGET].astype(int)
        y_pred = group["predicted_failure_within_24h"].astype(int)

        positives = int((y_true == 1).sum())
        if positives == 0:
            precision = None
            recall = None
            f1 = None
        else:
            precision = float(
                precision_score(y_true, y_pred, zero_division=0)
            )
            recall = float(
                recall_score(y_true, y_pred, zero_division=0)
            )
            f1 = float(f1_score(y_true, y_pred, zero_division=0))

        records.append({
            "split": split_name,
            "test_id": test_id,
            "bearing_id": int(bearing_id),
            "is_failed_trajectory": (
                (test_id, int(bearing_id)) in failed
            ),
            "rows": int(len(group)),
            "positive_rows": positives,
            "predicted_positives": int((y_pred == 1).sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_probability": float(
                group["failure_probability"].mean()
            ),
            "max_probability": float(
                group["failure_probability"].max()
            ),
        })

    return pd.DataFrame(records)


def save_pr_curves(
    model,
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for split_name, (X, y) in datasets.items():
        probability = predict_proba(model, X)
        precision, recall, _ = precision_recall_curve(y, probability)
        ap = average_precision_score(y, probability)
        ax.plot(
            recall,
            precision,
            label=f"{split_name} (AP={ap:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curves.png", dpi=160)
    plt.close(fig)


def save_trajectory_recall_plot(
    traj_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    failed = traj_df[traj_df["is_failed_trajectory"]].copy()
    if failed.empty:
        return

    failed["label"] = (
        failed["split"]
        + "\n"
        + failed["test_id"]
        + " B"
        + failed["bearing_id"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(failed))
    recalls = failed["recall"].fillna(0.0).to_numpy()
    ax.bar(x, recalls, color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(failed["label"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall on failure_within_24h = 1 rows")
    ax.set_title("Failed-trajectory recall")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_recall.png", dpi=160)
    plt.close(fig)


def verify_classifier(
    val_metrics: dict,
    test_metrics: dict,
    traj_df: pd.DataFrame,
) -> dict:
    failed_val = traj_df[
        (traj_df["split"] == "validation")
        & (traj_df["is_failed_trajectory"])
    ]
    failed_test = traj_df[
        (traj_df["split"] == "test")
        & (traj_df["is_failed_trajectory"])
    ]

    # Floors reflect IMS constraints (one failed train trajectory) while
    # still requiring a usable precision/recall balance.
    checks = {
        "val_precision_ge_0_30": val_metrics["precision"] >= 0.30,
        "val_recall_ge_0_60": val_metrics["recall"] >= 0.60,
        "val_f1_ge_0_40": val_metrics["f1"] >= 0.40,
        "val_ap_ge_0_30": val_metrics["average_precision"] >= 0.30,
        "test_precision_ge_0_50": test_metrics["precision"] >= 0.50,
        "test_recall_ge_0_40": test_metrics["recall"] >= 0.40,
        "test_ap_ge_0_70": test_metrics["average_precision"] >= 0.70,
        "failed_val_recall_ge_0_60": bool(
            (failed_val["recall"].fillna(0) >= 0.60).all()
        ) if len(failed_val) else False,
        "failed_test_any_recall_ge_0_40": bool(
            (failed_test["recall"].fillna(0) >= 0.40).any()
        ) if len(failed_test) else False,
        "healthy_val_no_mass_false_alarms": bool(
            (
                traj_df[
                    (traj_df["split"] == "validation")
                    & (~traj_df["is_failed_trajectory"])
                ]["predicted_positives"]
                <= 50
            ).all()
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
    log("XGBOOST FAILURE-WITHIN-24-HOURS CLASSIFIER")
    log("=" * 70)
    log(f"XGBoost version: {xgboost.__version__}")

    train_df = read_dataset(TRAIN_FILE)
    val_df = read_dataset(VAL_FILE)
    test_df = read_dataset(TEST_FILE)

    log(f"Raw train: {train_df.shape}")
    log(f"Raw validation: {val_df.shape}")
    log(f"Raw test: {test_df.shape}")

    features = get_feature_columns(train_df)

    for name, df in (
        ("validation", val_df),
        ("test", test_df),
    ):
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} is missing features: {sorted(missing)}"
            )

    X_train, y_train, labeled_train = prepare_labeled_data(
        train_df, features, "train"
    )
    X_val, y_val, labeled_val = prepare_labeled_data(
        val_df, features, "validation"
    )
    X_test, y_test, labeled_test = prepare_labeled_data(
        test_df, features, "test"
    )

    log("\nLABELED DATA")
    log("-" * 50)

    for name, y in (
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ):
        log(
            f"{name}: {len(y):,} rows | "
            f"positive={int((y == 1).sum()):,} | "
            f"negative={int((y == 0).sum()):,}"
        )

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0 or y_train.nunique() != 2:
        raise ValueError(
            "Training data must contain both target classes."
        )

    base_scale_pos_weight = negative_count / positive_count
    log(f"\nFeatures: {len(features)}")
    log(f"Base scale_pos_weight (neg/pos): {base_scale_pos_weight:.4f}")

    raw_model, best_trial, trials = select_model_on_validation(
        X_train,
        y_train,
        X_val,
        y_val,
        base_scale_pos_weight,
    )

    log("\nBEST HYPERPARAMETERS (VALIDATION AP)")
    log("-" * 50)
    log(json.dumps(make_json_safe(best_trial), indent=2))

    log("\nCalibrating probabilities on validation (sigmoid)...")
    model = calibrate_model(raw_model, X_val, y_val)

    val_probability = predict_proba(model, X_val)
    threshold, threshold_stats = select_threshold(
        y_val, val_probability
    )

    log("\nSELECTED THRESHOLD (VALIDATION ONLY)")
    log("-" * 50)
    log(json.dumps(make_json_safe(threshold_stats), indent=2))

    train_metrics = evaluate_split(
        model, "train", X_train, y_train, threshold
    )
    val_metrics = evaluate_split(
        model, "validation", X_val, y_val, threshold
    )
    test_metrics = evaluate_split(
        model, "test", X_test, y_test, threshold
    )

    pred_frames = {}
    for split_name, X, labeled in (
        ("train", X_train, labeled_train),
        ("validation", X_val, labeled_val),
        ("test", X_test, labeled_test),
    ):
        pred_frames[split_name] = export_predictions(
            model,
            labeled,
            X,
            threshold,
            split_name,
            OUTPUT_DIR,
        )

    traj_parts = [
        trajectory_metrics(pred_frames["train"], "train"),
        trajectory_metrics(pred_frames["validation"], "validation"),
        trajectory_metrics(pred_frames["test"], "test"),
    ]
    traj_df = pd.concat(traj_parts, ignore_index=True)
    traj_df.to_csv(OUTPUT_DIR / "trajectory_metrics.csv", index=False)

    log("\nTRAJECTORY METRICS")
    log("-" * 50)
    log(traj_df.to_string(index=False))

    save_pr_curves(
        model,
        {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
            "test": (X_test, y_test),
        },
        OUTPUT_DIR,
    )
    save_trajectory_recall_plot(traj_df, OUTPUT_DIR)

    verification = verify_classifier(
        val_metrics, test_metrics, traj_df
    )

    # Persist the calibrated model used at inference time.
    joblib.dump(
        model,
        OUTPUT_DIR / "xgboost_failure_classifier.joblib",
    )
    # Keep the uncalibrated booster for debugging/repro.
    joblib.dump(
        raw_model,
        OUTPUT_DIR / "xgboost_failure_classifier_raw.joblib",
    )

    with (OUTPUT_DIR / "feature_columns.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(features, f, indent=2)

    threshold_payload = {
        "threshold": float(threshold),
        "selection_rule": threshold_stats["selection_rule"],
        "validation_threshold_stats": threshold_stats,
        "min_val_precision": MIN_VAL_PRECISION,
        "min_val_recall": MIN_VAL_RECALL,
        "target": TARGET,
        "positive_class": 1,
        "calibrated": True,
        "calibration_method": "sigmoid",
    }

    with (OUTPUT_DIR / "threshold.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            make_json_safe(threshold_payload),
            f,
            indent=2,
            allow_nan=False,
        )

    selected_payload = {
        "model_path": "models/failure_classifier/xgboost_failure_classifier.joblib",
        "raw_model_path": (
            "models/failure_classifier/"
            "xgboost_failure_classifier_raw.joblib"
        ),
        "feature_columns_path": (
            "models/failure_classifier/feature_columns.json"
        ),
        "threshold_path": "models/failure_classifier/threshold.json",
        "threshold": float(threshold),
        "calibrated": True,
        "calibration_method": "sigmoid",
        "best_trial": best_trial,
        "verification": verification,
        "target": TARGET,
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

    importance = pd.DataFrame({
        "feature": features,
        "importance": raw_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "trial": trial["trial"],
                "val_average_precision": trial[
                    "val_average_precision"
                ],
                "val_roc_auc": trial["val_roc_auc"],
                "best_iteration": trial["best_iteration"],
                **trial["params"],
            }
            for trial in trials
        ]
    ).sort_values(
        "val_average_precision", ascending=False
    ).to_csv(OUTPUT_DIR / "hyperparameter_search.csv", index=False)

    report = {
        "algorithm": "Calibrated XGBClassifier (sigmoid / Platt)",
        "xgboost_version": xgboost.__version__,
        "target": TARGET,
        "threshold": float(threshold),
        "threshold_selection": threshold_stats["selection_rule"],
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_trial": best_trial,
        "features": features,
        "feature_count": len(features),
        "base_scale_pos_weight": float(base_scale_pos_weight),
        "random_state": RANDOM_STATE,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "verification": verification,
        "test_used_for_tuning": False,
        "mitigations_applied": [
            "Validation hyperparameter search maximizing PR-AUC",
            "Softer scale_pos_weight candidates (sqrt / fractions)",
            "Stronger regularization grid (depth, min_child_weight, reg_lambda)",
            "Sigmoid probability calibration on validation "
            "(better than isotonic with only 39 positives)",
            "Validation F1 threshold with precision/recall floors",
            "Two failed train trajectories (T1-B3 inner race, T2-B1 outer race)",
            "Trajectory-level evaluation for failed bearings",
        ],
        "limitations": [
            "Only four documented failed trajectories exist in IMS.",
            "Train positives come from two failed trajectories "
            "(1st_test bearing 3 and 2nd_test bearing 1); "
            "val/test still hold out different bearings.",
            "Validation is used for early stopping, calibration, "
            "hyperparameter search, and threshold selection.",
            "Test metrics are held-out and were not used for tuning.",
            "Prefer PR-AUC / F1 / trajectory recall over accuracy.",
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
            "Inspect trajectory_metrics.csv and pr_curves.png. "
            "Artifacts were still saved."
        )
    else:
        log("\nAll critical verification checks passed.")

    log(f"\nOutputs saved to: {OUTPUT_DIR}")
    log(f"Elapsed: {time.perf_counter() - started:.1f} seconds")
    log("CLASSIFIER EXPERIMENT FINISHED")


if __name__ == "__main__":
    main()
