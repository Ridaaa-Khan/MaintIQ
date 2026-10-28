"""
Isolation Forest training and evaluation (anomaly detection).

Input:
    data/processed/features_train.csv
    data/processed/features_val.csv
    data/processed/features_test.csv

Outputs:
    models/isolation_forest/
        isolation_forest_auto.joblib
        isolation_forest_1pct.joblib
        isolation_forest_selected.joblib
        selected_model.json
        feature_columns.json
        experiment_config.json
        metrics.json / metrics.csv
        failure_trajectory_analysis.csv
        scores_train.csv / scores_val.csv / scores_test.csv
        anomaly_score_distributions.png
        anomaly_scores_over_time.png
        baseline_vs_training_scores.png
        failure_score_lift.png

Inference helper (load selected model and score rows):
    isolation_forest_infer.py
"""

from __future__ import annotations

import inspect
import json
import shutil
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_config import (
    FAILED_BEARINGS,
    TARGET_COLUMNS,
    failed_bearing_set,
    feature_columns,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "isolation_forest"

TRAIN_FILE = DATA_DIR / "features_train.csv"
VAL_FILE = DATA_DIR / "features_val.csv"
TEST_FILE = DATA_DIR / "features_test.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 300
CONTAMINATION_OPTIONS = ("auto", 0.01)

ARTIFACT_COL = "is_artifact"
BASELINE_COL = "isolation_forest_train"
CLASS_TARGET = "failure_within_24h"

METADATA_COLUMNS = {
    "test_id",
    "bearing_id",
    "timestamp",
    "source_file",
}

EXCLUDED_COLUMNS = (
    METADATA_COLUMNS
    | set(TARGET_COLUMNS)
    | {ARTIFACT_COL, BASELINE_COL, "_split"}
)

# Experimental + selected score columns (never used as model inputs).
SCORE_COLUMNS = {
    "anomaly_score_auto",
    "decision_score_auto",
    "anomaly_flag_auto",
    "anomaly_score_1pct",
    "decision_score_1pct",
    "anomaly_flag_1pct",
    "anomaly_score",
    "decision_score",
    "anomaly_flag",
}


# ============================================================
# UTILITIES
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run target engineering first and verify the CSV paths."
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])

    required = {
        "test_id",
        "bearing_id",
        "timestamp",
        ARTIFACT_COL,
        BASELINE_COL,
        CLASS_TARGET,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if df["timestamp"].isna().any():
        raise ValueError(f"{path.name} contains missing timestamps.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Use the project's shared feature selector, then enforce
    exclusions and numeric eligibility.

    Supports a helper that accepts either a DataFrame or
    its column index.
    """
    signature = inspect.signature(feature_columns)
    params = list(signature.parameters.values())

    if not params:
        selected = feature_columns()
    else:
        first_name = params[0].name.lower()

        if "column" in first_name:
            selected = feature_columns(df.columns)
        else:
            selected = feature_columns(df)

    selected = list(selected)

    selected = [
        col for col in selected
        if col in df.columns
        and col not in EXCLUDED_COLUMNS
        and col not in SCORE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not selected:
        raise ValueError(
            "No numeric features were selected by "
            "pipeline_config.feature_columns()."
        )

    if len(selected) != len(set(selected)):
        raise ValueError("The feature selector returned duplicate columns.")

    return selected


def parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    """Safely parse booleans loaded from CSV."""
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
        bad_values = series[result.isna()].unique()[:10]
        raise ValueError(
            f"Unexpected values in {column_name}: {bad_values}"
        )

    return result.astype(bool)


def eligible_metric_rows(df: pd.DataFrame) -> pd.Series:
    """
    Rows eligible for exploratory comparison against the
    separate supervised 24-hour target.
    """
    artifact = parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)

    return (
        ~artifact
        & df[CLASS_TARGET].isin([0, 1])
    )


def classification_metrics(
    df: pd.DataFrame,
    anomaly_score: pd.Series,
    anomaly_flag: pd.Series,
    split_name: str,
    contamination_name: str,
) -> dict:
    """
    Compare anomaly scores against failure_within_24h as a
    proxy reference. These are NOT ground-truth anomaly metrics.
    """
    mask = eligible_metric_rows(df)

    result = {
        "split": split_name,
        "contamination": contamination_name,
        "metric_interpretation": (
            "Proxy alignment with failure_within_24h; "
            "not ground-truth anomaly-detection accuracy."
        ),
        "eligible_rows": int(mask.sum()),
        "positive_rows": 0,
        "negative_rows": 0,
        "flagged_rows": int(anomaly_flag.sum()),
        "flag_rate_all_rows": float(anomaly_flag.mean()),
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "average_precision": None,
        "confusion_matrix_tn_fp_fn_tp": None,
    }

    if not mask.any():
        return result

    y_true = df.loc[mask, CLASS_TARGET].astype(int).to_numpy()
    scores = anomaly_score.loc[mask].to_numpy()
    predictions = anomaly_flag.loc[mask].astype(int).to_numpy()

    result["positive_rows"] = int((y_true == 1).sum())
    result["negative_rows"] = int((y_true == 0).sum())
    result["flagged_rows_eligible"] = int(predictions.sum())

    result["precision"] = float(
        precision_score(y_true, predictions, zero_division=0)
    )
    result["recall"] = float(
        recall_score(y_true, predictions, zero_division=0)
    )
    result["f1"] = float(
        f1_score(y_true, predictions, zero_division=0)
    )

    result["accuracy"] = float(
        accuracy_score(y_true, predictions)
    )

    if np.unique(y_true).size == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["average_precision"] = float(
            average_precision_score(y_true, scores)
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    result["confusion_matrix_tn_fp_fn_tp"] = [
        int(tn), int(fp), int(fn), int(tp)
    ]

    return result


# ============================================================
# FEATURE AUDIT
# ============================================================

def audit_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_features: list[str],
    baseline_mask: pd.Series,
) -> tuple[list[str], list[str]]:
    """
    Remove features that are non-finite or constant on the
    baseline training subset.

    No feature filtering uses validation/test outcomes.
    """
    baseline = train_df.loc[baseline_mask, candidate_features]

    if baseline.empty:
        raise ValueError(
            "No Isolation Forest baseline rows found in training data."
        )

    finite = np.isfinite(baseline.to_numpy(dtype=np.float64))
    valid_columns = finite.all(axis=0)

    # A feature is unsuitable if it has non-finite values
    # on the training baseline.
    non_finite_features = [
        col for col, is_valid in zip(candidate_features, valid_columns)
        if not is_valid
    ]

    kept = [
        col for col, is_valid in zip(candidate_features, valid_columns)
        if is_valid
    ]

    constant_features = [
        col for col in kept
        if baseline[col].nunique(dropna=True) <= 1
    ]

    kept = [
        col for col in kept
        if col not in set(constant_features)
    ]

    if not kept:
        raise ValueError(
            "No features remain after the training-baseline audit."
        )

    log("\nFEATURE AUDIT")
    log("-" * 50)
    log(f"Candidate features: {len(candidate_features)}")
    log(f"Non-finite on baseline: {len(non_finite_features)}")
    log(f"Constant on baseline: {len(constant_features)}")
    log(f"Final features: {len(kept)}")

    if non_finite_features:
        log(f"Excluded non-finite: {non_finite_features}")

    if constant_features:
        log(f"Excluded constant: {constant_features}")

    # Record whether held-out splits contain missing/non-finite
    # values; the imputer will use training-baseline medians.
    for name, df in (("validation", val_df), ("test", test_df)):
        values = df[kept].to_numpy(dtype=np.float64)
        bad_count = int((~np.isfinite(values)).sum())
        if bad_count:
            warnings.warn(
                f"{name} contains {bad_count} non-finite feature "
                "values. The training-fitted imputer will handle them."
            )

    return kept, non_finite_features + constant_features


# ============================================================
# MODEL TRAINING AND SCORING
# ============================================================

def make_model(contamination: str | float) -> IsolationForest:
    return IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def pipeline_feature_names(model: Pipeline) -> list[str]:
    """Feature names recorded by the first fitted pipeline step."""
    first_step = model.steps[0][1]
    return list(first_step.feature_names_in_)


def transform_features(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply every preprocessing step before the Isolation Forest."""
    Xt = X
    for name, step in model.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def add_scores(
    df: pd.DataFrame,
    model: Pipeline,
    contamination_label: str,
) -> pd.DataFrame:
    """
    IsolationForest score_samples / decision_function:
    lower means more anomalous.

    We invert score_samples so higher anomaly_score means
    more anomalous, which is easier to interpret.
    """
    result = df.copy()

    features = pipeline_feature_names(model)
    X = result[list(features)]

    estimator = model.named_steps["isolation_forest"]
    X_prepared = transform_features(model, X)

    raw_score = estimator.score_samples(X_prepared)
    decision_score = estimator.decision_function(X_prepared)
    prediction = estimator.predict(X_prepared)

    suffix = "auto" if contamination_label == "auto" else "1pct"

    result[f"anomaly_score_{suffix}"] = -raw_score
    result[f"decision_score_{suffix}"] = decision_score
    result[f"anomaly_flag_{suffix}"] = prediction == -1

    return result


def train_pipeline(
    train_df: pd.DataFrame,
    features: list[str],
    baseline_mask: pd.Series,
    contamination: str | float,
) -> Pipeline:
    """
    Fit imputer + scaler + Isolation Forest on baseline
    training observations only.
    """
    X_baseline = train_df.loc[baseline_mask, features].copy()

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=False,
            ),
        ),
        ("scaler", StandardScaler()),
        ("isolation_forest", make_model(contamination)),
    ])

    pipeline.fit(X_baseline)

    return pipeline


# ============================================================
# PLOTS
# ============================================================

def save_score_distribution(
    scored: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for split_name in ("train", "val", "test"):
        df = scored[split_name]
        mask = ~parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)

        ax.hist(
            df.loc[mask, "anomaly_score_auto"],
            bins=60,
            alpha=0.45,
            density=True,
            label=split_name,
        )

    ax.set_title("Isolation Forest anomaly-score distributions")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_score_distributions.png", dpi=160)
    plt.close(fig)


def save_scores_over_time(
    scored: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Plot scores without connecting across gaps greater than 2 hours."""
    trajectory_groups = []

    for split_name in ("train", "val", "test"):
        df = scored[split_name]
        artifact = parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)

        for (test_id, bearing_id), group in df.loc[~artifact].groupby(
            ["test_id", "bearing_id"], sort=True
        ):
            group = group.sort_values("timestamp").copy()
            trajectory_groups.append(
                (split_name, test_id, bearing_id, group)
            )

    fig, axes = plt.subplots(
        len(trajectory_groups),
        1,
        figsize=(13, max(5, 2.1 * len(trajectory_groups))),
        squeeze=False,
    )

    for ax, (split_name, test_id, bearing_id, group) in zip(
        axes.ravel(), trajectory_groups
    ):
        scores = group["anomaly_score_auto"].astype(float).copy()

        # Break the line at the first observation after a long gap.
        gaps = group["timestamp"].diff().dt.total_seconds() / 3600
        scores.loc[gaps > 2.0] = np.nan

        ax.plot(group["timestamp"], scores, linewidth=0.8)
        ax.set_title(
            f"{split_name}: {test_id}, Bearing {bearing_id}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Anomaly score")
        ax.tick_params(axis="x", labelrotation=25)

    axes[-1, 0].set_xlabel("Timestamp")
    fig.suptitle(
        "Isolation Forest anomaly scores over time (auto threshold)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "anomaly_scores_over_time.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_score_diagnostics(
    scored: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Diagnose baseline separation and potential time-gap effects."""
    train = scored["train"].copy()
    artifact = parse_boolean(train[ARTIFACT_COL], ARTIFACT_COL)
    baseline = parse_boolean(train[BASELINE_COL], BASELINE_COL)
    train = train.loc[~artifact].copy()
    train["baseline_group"] = np.where(
        baseline.loc[train.index],
        "baseline",
        "other_training_rows",
    )

    summary = (
        train.groupby("baseline_group")["anomaly_score_auto"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(
        output_dir / "baseline_score_summary.csv",
        index=False,
    )

    # Compare training-baseline scores against the full validation
    # and test distributions.
    distribution_records = []

    for split_name in ("train", "val", "test"):
        df = scored[split_name]
        is_artifact = parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)
        is_baseline = parse_boolean(df[BASELINE_COL], BASELINE_COL)

        if split_name == "train":
            groups = {
                "baseline": is_baseline & ~is_artifact,
                "other_training_rows": ~is_baseline & ~is_artifact,
            }
        else:
            groups = {"all_operational_rows": ~is_artifact}

        for group_name, mask in groups.items():
            values = df.loc[mask, "anomaly_score_auto"].dropna()

            distribution_records.append({
                "split": split_name,
                "group": group_name,
                "count": int(values.size),
                "mean": float(values.mean()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
                "p90": float(values.quantile(0.90)) if len(values) else None,
                "p95": float(values.quantile(0.95)) if len(values) else None,
                "p99": float(values.quantile(0.99)) if len(values) else None,
            })

    pd.DataFrame(distribution_records).to_csv(
        output_dir / "score_distribution_summary.csv",
        index=False,
    )

    # Plot baseline against the other training observations.
    fig, ax = plt.subplots(figsize=(10, 6))

    for group_name, mask in (
        ("Training baseline", baseline.loc[train.index]),
        ("Other training rows", ~baseline.loc[train.index]),
    ):
        values = train.loc[mask, "anomaly_score_auto"]

        if len(values):
            ax.hist(
                values,
                bins=60,
                alpha=0.5,
                density=True,
                label=group_name,
            )

    ax.set_title("Baseline versus other training observations")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "baseline_vs_training_scores.png",
        dpi=160,
    )
    plt.close(fig)

    # Associations are diagnostic, not proof of causation.
    candidate_diagnostics = [
        "large_gap_flag",
        "time_since_previous_hours",
    ]

    correlation_records = []

    for split_name, df in scored.items():
        operational = df.loc[
            ~parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)
        ]

        for feature in candidate_diagnostics:
            if feature not in operational.columns:
                continue

            pair = operational[
                [feature, "anomaly_score_auto"]
            ].apply(pd.to_numeric, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            correlation = (
                pair[feature].corr(
                    pair["anomaly_score_auto"],
                    method="spearman",
                )
                if len(pair) >= 3 and pair[feature].nunique() > 1
                else np.nan
            )

            correlation_records.append({
                "split": split_name,
                "feature": feature,
                "valid_rows": len(pair),
                "spearman_correlation": (
                    float(correlation)
                    if pd.notna(correlation)
                    else None
                ),
            })

    pd.DataFrame(correlation_records).to_csv(
        output_dir / "temporal_feature_score_correlations.csv",
        index=False,
    )

    log("\nBASELINE SCORE SUMMARY")
    log(summary.to_string(index=False))
    log("\nSaved baseline and temporal-feature diagnostic reports.")


def compute_trajectory_analysis(
    scored: dict[str, pd.DataFrame],
    score_col: str,
    flag_col: str,
) -> pd.DataFrame:
    """Early-life vs late-life anomaly scores for every trajectory."""
    failed = failed_bearing_set()
    records = []

    for split_name, df in scored.items():
        operational = df.loc[
            ~parse_boolean(df[ARTIFACT_COL], ARTIFACT_COL)
        ].copy()

        for (test_id, bearing_id), group in operational.groupby(
            ["test_id", "bearing_id"], sort=True
        ):
            group = group.sort_values("timestamp")
            n = len(group)
            window = max(1, n // 5)
            early = group.iloc[:window]
            late = group.iloc[-window:]
            is_failed = (test_id, int(bearing_id)) in failed

            early_mean = float(early[score_col].mean())
            late_mean = float(late[score_col].mean())

            records.append({
                "split": split_name,
                "test_id": test_id,
                "bearing_id": int(bearing_id),
                "is_failed": is_failed,
                "n_operational": n,
                "early_mean_score": early_mean,
                "late_mean_score": late_mean,
                "score_lift_late_minus_early": late_mean - early_mean,
                "late_flag_rate": float(late[flag_col].mean()),
                "overall_flag_rate": float(group[flag_col].mean()),
            })

    return pd.DataFrame(records)


def save_failure_trajectory_report(
    analysis: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Persist trajectory lift table + plot.

    Late-life lift on failed bearings is the main qualitative check
    that Isolation Forest tracks degradation (anomaly detection).
    """
    analysis.to_csv(
        output_dir / "failure_trajectory_analysis.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    failed_df = analysis[analysis["is_failed"]]
    healthy_df = analysis[~analysis["is_failed"]]

    x_failed = np.arange(len(failed_df))
    x_healthy = np.arange(len(healthy_df)) + len(failed_df) + 1

    ax.bar(
        x_failed,
        failed_df["score_lift_late_minus_early"],
        color="#c0392b",
        label="Failed trajectories",
    )
    ax.bar(
        x_healthy,
        healthy_df["score_lift_late_minus_early"],
        color="#2980b9",
        label="Healthy trajectories",
    )

    labels = (
        [
            f"{r.split}\n{r.test_id} B{r.bearing_id}"
            for r in failed_df.itertuples()
        ]
        + [
            f"{r.split}\n{r.test_id} B{r.bearing_id}"
            for r in healthy_df.itertuples()
        ]
    )
    ax.set_xticks(list(x_failed) + list(x_healthy))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Late − early mean anomaly score")
    ax.set_title("Anomaly-score lift (last 20% − first 20% of run)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "failure_score_lift.png", dpi=160)
    plt.close(fig)

    log("\nFAILURE / TRAJECTORY SCORE LIFT")
    log("-" * 50)
    log(
        analysis.sort_values(
            ["is_failed", "score_lift_late_minus_early"],
            ascending=[False, False],
        ).to_string(index=False)
    )


def select_production_model(
    scored: dict[str, pd.DataFrame],
    analysis_by_label: dict[str, pd.DataFrame],
) -> dict:
    """
    Choose contamination on validation only.

    Prefer the setting with the lower validation operational flag
    rate, provided failed validation trajectories still show a
    positive late-life score lift.
    """
    val = scored["val"]
    operational = ~parse_boolean(val[ARTIFACT_COL], ARTIFACT_COL)

    candidates = []

    for label in ("auto", "1pct"):
        flag_col = f"anomaly_flag_{label}"
        flag_rate = float(val.loc[operational, flag_col].mean())

        failed_val = analysis_by_label[label]
        failed_val = failed_val[
            (failed_val["split"] == "val")
            & (failed_val["is_failed"])
        ]
        mean_lift = float(
            failed_val["score_lift_late_minus_early"].mean()
        ) if len(failed_val) else float("nan")
        late_flag = float(
            failed_val["late_flag_rate"].mean()
        ) if len(failed_val) else float("nan")

        candidates.append({
            "label": label,
            "contamination": "auto" if label == "auto" else 0.01,
            "val_operational_flag_rate": flag_rate,
            "val_failed_mean_score_lift": mean_lift,
            "val_failed_late_flag_rate": late_flag,
            "detects_failed_val_degradation": (
                pd.notna(mean_lift) and mean_lift > 0
            ),
        })

    # Eligible: positive lift on failed val trajectories.
    eligible = [
        c for c in candidates
        if c["detects_failed_val_degradation"]
    ]
    pool = eligible if eligible else candidates

    chosen = min(
        pool,
        key=lambda c: c["val_operational_flag_rate"],
    )

    # Return a fresh dict so we never nest `candidates` inside an
    # object that already lives in that same list (circular JSON).
    return {
        "label": chosen["label"],
        "contamination": chosen["contamination"],
        "val_operational_flag_rate": chosen[
            "val_operational_flag_rate"
        ],
        "val_failed_mean_score_lift": chosen[
            "val_failed_mean_score_lift"
        ],
        "val_failed_late_flag_rate": chosen[
            "val_failed_late_flag_rate"
        ],
        "detects_failed_val_degradation": chosen[
            "detects_failed_val_degradation"
        ],
        "selection_rule": (
            "Among contamination settings that show positive late-life "
            "score lift on failed validation trajectories, choose the "
            "lowest validation operational flag rate. Test metrics are "
            "not used for selection."
        ),
        "candidates": candidates,
    }


def attach_selected_score_columns(
    scored: dict[str, pd.DataFrame],
    selected_label: str,
) -> None:
    """Copy the selected contamination columns to generic names."""
    for df in scored.values():
        df["anomaly_score"] = df[f"anomaly_score_{selected_label}"]
        df["decision_score"] = df[f"decision_score_{selected_label}"]
        df["anomaly_flag"] = df[f"anomaly_flag_{selected_label}"]


def verify_anomaly_detection(
    scored: dict[str, pd.DataFrame],
    analysis: pd.DataFrame,
    selected_label: str,
) -> dict:
    """
    Sanity checks that the selected model behaves like an
    anomaly / degradation detector on held-out trajectories.
    """
    checks = {}

    # 1) Failed validation bearings should show positive lift.
    failed_val = analysis[
        (analysis["split"] == "val") & (analysis["is_failed"])
    ]
    checks["val_failed_positive_lift"] = bool(
        (failed_val["score_lift_late_minus_early"] > 0).all()
    )

    # 2) Baseline training rows should mostly be inliers.
    train = scored["train"]
    baseline = parse_boolean(train[BASELINE_COL], BASELINE_COL)
    artifacts = parse_boolean(train[ARTIFACT_COL], ARTIFACT_COL)
    baseline_rows = train.loc[baseline & ~artifacts]
    baseline_flag_rate = float(
        baseline_rows[f"anomaly_flag_{selected_label}"].mean()
    )
    checks["baseline_flag_rate"] = baseline_flag_rate
    checks["baseline_mostly_inliers"] = baseline_flag_rate < 0.10

    # 3) Validation operational flag rate should stay below 35%.
    val = scored["val"]
    val_op = ~parse_boolean(val[ARTIFACT_COL], ARTIFACT_COL)
    val_flag_rate = float(
        val.loc[val_op, f"anomaly_flag_{selected_label}"].mean()
    )
    checks["val_operational_flag_rate"] = val_flag_rate
    checks["val_flag_rate_reasonable"] = val_flag_rate < 0.35

    # 4) At least one failed trajectory (any split) has late flag rate
    #    clearly above its early half for qualitative detection.
    failed_any = analysis[analysis["is_failed"]]
    checks["any_failed_late_flag_gt_0.5"] = bool(
        (failed_any["late_flag_rate"] > 0.5).any()
    )

    checks["all_critical_passed"] = all(
        [
            checks["val_failed_positive_lift"],
            checks["baseline_mostly_inliers"],
            checks["val_flag_rate_reasonable"],
            checks["any_failed_late_flag_gt_0.5"],
        ]
    )

    return checks


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("ISOLATION FOREST TRAINING")
    log("=" * 70)
    log(f"Project root: {PROJECT_ROOT}")
    log(f"scikit-learn version: {sklearn.__version__}")

    train_df = read_dataset(TRAIN_FILE)
    val_df = read_dataset(VAL_FILE)
    test_df = read_dataset(TEST_FILE)

    log("\nDATASETS")
    log("-" * 50)
    log(f"Train:      {train_df.shape}")
    log(f"Validation: {val_df.shape}")
    log(f"Test:       {test_df.shape}")

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more train/validation/test datasets are empty.")

    # The shared helper selects the project-defined feature set.
    candidate_features = get_feature_columns(train_df)

    # Training baseline: selected chronological baseline rows,
    # restricted to the training split and excluding artifacts.
    artifacts = parse_boolean(train_df[ARTIFACT_COL], ARTIFACT_COL)
    baseline_selected = parse_boolean(
        train_df[BASELINE_COL], BASELINE_COL
    )

    baseline_mask = baseline_selected & ~artifacts

    if baseline_mask.sum() < 100:
        raise ValueError(
            f"Unexpectedly small baseline: {baseline_mask.sum()} rows."
        )

    log(f"\nTraining baseline rows: {int(baseline_mask.sum()):,}")
    log(
        "All baseline rows come from features_train.csv; "
        "validation/test rows are not used for fitting."
    )

    features, excluded_features = audit_features(
        train_df,
        val_df,
        test_df,
        candidate_features,
        baseline_mask,
    )

    # Explicitly verify all features are present in each split.
    for name, df in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} is missing model features: {sorted(missing)}"
            )

    scored: dict[str, pd.DataFrame] = {}
    metric_records: list[dict] = []
    model_metadata = {}

    # Compare only the two predeclared settings. No test results
    # are used to choose the threshold.
    for contamination in CONTAMINATION_OPTIONS:
        label = (
            "auto" if contamination == "auto"
            else "1pct"
        )

        log("\n" + "=" * 70)
        log(f"FITTING CONTAMINATION = {contamination}")
        log("=" * 70)

        pipeline = train_pipeline(
            train_df,
            features,
            baseline_mask,
            contamination,
        )

        model_path = OUTPUT_DIR / f"isolation_forest_{label}.joblib"
        joblib.dump(pipeline, model_path)

        model_metadata[label] = {
            "contamination": contamination,
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "training_rows": int(baseline_mask.sum()),
            "feature_count": len(features),
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
        }

        # Score all three splits independently. Validation and
        # test observations are never used for fit.
        split_frames = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        for split_name, df in split_frames.items():
            current = add_scores(df, pipeline, label)

            if split_name not in scored:
                scored[split_name] = current
            else:
                score_cols = [
                    f"anomaly_score_{label}",
                    f"decision_score_{label}",
                    f"anomaly_flag_{label}",
                ]
                scored[split_name][score_cols] = current[score_cols]

            anomaly_col = f"anomaly_flag_{label}"
            score_col = f"anomaly_score_{label}"

            metric_records.append(
                classification_metrics(
                    current,
                    current[score_col],
                    current[anomaly_col],
                    split_name,
                    label,
                )
            )

            log(
                f"{split_name}: "
                f"{int(current[anomaly_col].sum()):,} flagged / "
                f"{len(current):,} rows "
                f"({100 * current[anomaly_col].mean():.2f}%)"
            )

        # Persist the imputer alongside each pipeline already.
        # Export the shared feature list separately below.

    # Trajectory degradation analysis for both contamination settings.
    analysis_by_label = {
        label: compute_trajectory_analysis(
            scored,
            score_col=f"anomaly_score_{label}",
            flag_col=f"anomaly_flag_{label}",
        )
        for label in ("auto", "1pct")
    }

    selection = select_production_model(scored, analysis_by_label)
    selected_label = selection["label"]
    analysis = analysis_by_label[selected_label]
    save_failure_trajectory_report(analysis, OUTPUT_DIR)

    attach_selected_score_columns(scored, selected_label)

    selected_src = OUTPUT_DIR / f"isolation_forest_{selected_label}.joblib"
    selected_dst = OUTPUT_DIR / "isolation_forest_selected.joblib"
    shutil.copy2(selected_src, selected_dst)

    verification = verify_anomaly_detection(
        scored, analysis, selected_label
    )

    selection_payload = {
        "selected_label": selected_label,
        "selected_contamination": selection["contamination"],
        "model_path": str(
            selected_dst.relative_to(PROJECT_ROOT)
        ),
        "feature_columns_path": "models/isolation_forest/feature_columns.json",
        "preprocessing": [
            "SimpleImputer(strategy='median')",
            "StandardScaler()",
            "IsolationForest",
        ],
        "selection_rule": selection["selection_rule"],
        "validation_stats": {
            "val_operational_flag_rate": selection[
                "val_operational_flag_rate"
            ],
            "val_failed_mean_score_lift": selection[
                "val_failed_mean_score_lift"
            ],
            "val_failed_late_flag_rate": selection[
                "val_failed_late_flag_rate"
            ],
        },
        "candidates": selection["candidates"],
        "verification": verification,
        "feature_policy": (
            "All 190 eligible numeric vibration/temporal features "
            "after excluding metadata and target/mask columns. "
            "Gap features (large_gap_flag, time_since_previous_hours) "
            "are retained as ordinary features."
        ),
    }

    with (OUTPUT_DIR / "selected_model.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(selection_payload, f, indent=2, allow_nan=False)

    # Save scores while preserving original columns and targets.
    for split_name, df in scored.items():
        path = OUTPUT_DIR / f"scores_{split_name}.csv"
        df.to_csv(path, index=False)
        log(f"Saved {path.name}: {df.shape}")

    # Metrics are exploratory comparisons against the existing
    # failure-within-24h target, not verified anomaly labels.
    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metric_records, f, indent=2, allow_nan=False)

    with (OUTPUT_DIR / "feature_columns.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(features, f, indent=2)

    config = {
        "project_root": str(PROJECT_ROOT),
        "algorithm": "IsolationForest",
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "contamination_options": list(CONTAMINATION_OPTIONS),
        "selected_label": selected_label,
        "selected_contamination": selection["contamination"],
        "preprocessing": selection_payload["preprocessing"],
        "training_baseline_rows": int(baseline_mask.sum()),
        "candidate_feature_count": len(candidate_features),
        "final_feature_count": len(features),
        "excluded_after_audit": excluded_features,
        "features": features,
        "target_columns_excluded": list(TARGET_COLUMNS),
        "metadata_columns_excluded": sorted(METADATA_COLUMNS),
        "artifact_rule": "is_artifact == False",
        "baseline_rule": (
            "isolation_forest_train == True AND is_artifact == False "
            "within features_train.csv only"
        ),
        "score_interpretation": (
            "Higher anomaly_score = more anomalous; "
            "threshold flags come from IsolationForest.predict."
        ),
        "metric_caveat": (
            "Precision/recall/F1/ROC-AUC/AP compare anomaly outputs "
            "with failure_within_24h as a proxy target. They do not "
            "measure accuracy against verified anomaly ground truth."
        ),
        "test_used_for_tuning": False,
        "models": model_metadata,
        "verification": verification,
        "failed_bearings": {
            test_id: bearing_ids
            for test_id, bearing_ids in FAILED_BEARINGS.items()
        },
    }

    with (OUTPUT_DIR / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2, allow_nan=False)

    save_score_distribution(scored, OUTPUT_DIR)
    save_scores_over_time(scored, OUTPUT_DIR)
    run_score_diagnostics(scored, OUTPUT_DIR)

    # Compact report for the terminal.
    log("\n" + "=" * 70)
    log("PROXY CLASSIFICATION METRICS")
    log("=" * 70)
    log(
        metrics_df[
            [
                "split",
                "contamination",
                "eligible_rows",
                "flagged_rows_eligible",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
            ]
        ].to_string(index=False)
    )

    log("\nIMPORTANT INTERPRETATION")
    log(
        "These metrics compare anomaly flags/scores with the "
        "failure_within_24h target. They are proxy alignment metrics, "
        "not ground-truth anomaly-detection accuracy."
    )
    log(
        "Contamination was selected on validation only. "
        "Treat test metrics as final reporting only."
    )

    log("\n" + "=" * 70)
    log("SELECTED PRODUCTION MODEL")
    log("=" * 70)
    log(f"Selected label        : {selected_label}")
    log(f"Contamination         : {selection['contamination']}")
    log(
        "Val operational flags : "
        f"{selection['val_operational_flag_rate']:.3%}"
    )
    log(
        "Val failed score lift : "
        f"{selection['val_failed_mean_score_lift']:.4f}"
    )
    log(f"Model file            : {selected_dst}")

    log("\nANOMALY-DETECTION VERIFICATION")
    for key, value in verification.items():
        log(f"  {key}: {value}")

    if not verification["all_critical_passed"]:
        raise RuntimeError(
            "Isolation Forest verification checks failed. "
            "Inspect failure_trajectory_analysis.csv and plots "
            "before using the model."
        )

    log(f"\nOutputs: {OUTPUT_DIR}")
    log(f"Elapsed: {time.perf_counter() - started:.1f} seconds")
    log("ISOLATION FOREST EXPERIMENT FINISHED")


if __name__ == "__main__":
    main()