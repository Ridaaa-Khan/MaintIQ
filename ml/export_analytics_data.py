"""
Rebuild fsd/web/public/js/analytics-data.js from scored pipeline artifacts.

Uses only values present in:
  - models/isolation_forest/scores_{train,val,test}.csv
  - models/failure_classifier/predictions_{train,validation,test}.csv
  - models/rul_regressor/predictions_{train,validation,test}.csv
  - model metric / threshold / baseline JSON+CSV files

No invented sensor values. Missing classifier/RUL scores stay null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUT = ROOT.parent / "fsd" / "web" / "public" / "js" / "analytics-data.js"

# Documented IMS damage labels (README / IMS documentation) — not model outputs.
DOCUMENTED_DAMAGE = {
    ("1st_test", 3): "Documented inner-race damage (T1-B3)",
    ("1st_test", 4): "Documented rolling-element damage (T1-B4)",
    ("2nd_test", 1): "Documented outer-race damage (T2-B1)",
    ("3rd_test", 3): "Documented outer-race damage (T3-B3)",
}

TRAJECTORY_SPLIT = {
    ("1st_test", 1): "train",
    ("1st_test", 2): "train",
    ("1st_test", 3): "train",
    ("2nd_test", 1): "train",
    ("2nd_test", 2): "train",
    ("2nd_test", 3): "train",
    ("3rd_test", 1): "train",
    ("3rd_test", 2): "train",
    ("1st_test", 4): "val",
    ("2nd_test", 4): "val",
    ("3rd_test", 3): "test",
    ("3rd_test", 4): "test",
}

BAND_COLS = [
    "band_energy_0_500_mean",
    "band_energy_500_1000_mean",
    "band_energy_1000_2000_mean",
    "band_energy_2000_4000_mean",
]


def r(value, digits=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def machine_key(test_id: str, bearing_id: int) -> str:
    # Preserve existing UI/report keys for Test 3 held-out bearings.
    if test_id == "3rd_test" and bearing_id == 3:
        return "brg-003"
    if test_id == "3rd_test" and bearing_id == 4:
        return "brg-004"
    label = {"1st_test": "t1", "2nd_test": "t2", "3rd_test": "t3"}[test_id]
    return f"{label}-b{bearing_id}"


def machine_name(test_id: str, bearing_id: int) -> str:
    # Keep legacy display names for the original Test 3 UI keys.
    if test_id == "3rd_test" and bearing_id in (3, 4):
        return f"IMS Bearing {bearing_id}"
    return f"IMS {test_id} Bearing {bearing_id}"


def machine_asset(test_id: str, bearing_id: int) -> str:
    if test_id == "3rd_test" and bearing_id in (3, 4):
        return f"BRG-{bearing_id:03d}"
    label = {"1st_test": "T1", "2nd_test": "T2", "3rd_test": "T3"}[test_id]
    return f"{label}-B{bearing_id}"


def load_scores() -> pd.DataFrame:
    frames = []
    for split, name in [
        ("train", "scores_train.csv"),
        ("val", "scores_val.csv"),
        ("test", "scores_test.csv"),
    ]:
        path = MODELS / "isolation_forest" / name
        df = pd.read_csv(path)
        df["artifact_split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out[out["is_artifact"] == False].copy()  # noqa: E712
    return out


def load_classifier() -> pd.DataFrame:
    frames = []
    for name in [
        "predictions_train.csv",
        "predictions_validation.csv",
        "predictions_test.csv",
    ]:
        frames.append(pd.read_csv(MODELS / "failure_classifier" / name))
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out[
        [
            "test_id",
            "bearing_id",
            "timestamp",
            "failure_probability",
            "predicted_failure_within_24h",
        ]
    ]


def load_rul() -> pd.DataFrame:
    frames = []
    for name in [
        "predictions_train.csv",
        "predictions_validation.csv",
        "predictions_test.csv",
    ]:
        frames.append(pd.read_csv(MODELS / "rul_regressor" / name))
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out[["test_id", "bearing_id", "timestamp", "predicted_rul_hours"]]


def mean_or_none(series: pd.Series, digits=4):
    series = series.dropna()
    if series.empty:
        return None
    return r(series.mean(), digits)


def build_machine(group: pd.DataFrame) -> dict:
    group = group.sort_values("timestamp").reset_index(drop=True)
    test_id = group["test_id"].iloc[0]
    bearing_id = int(group["bearing_id"].iloc[0])
    n = len(group)
    early_n = max(1, int(n * 0.2))
    late_n = max(1, int(n * 0.1))
    early = group.iloc[:early_n]
    late = group.iloc[-late_n:]
    latest_row = group.iloc[-1]

    # 12-hour calendar bins (same approach as existing Test 3 export).
    indexed = group.set_index("timestamp")
    resampled = indexed.resample("12h").agg(
        {
            "rms_mean": "mean",
            "peak_mean": "mean",
            "kurtosis_mean": "mean",
            "skewness_mean": "mean",
            "crest_factor_mean": "mean",
            "anomaly_score": "mean",
            "anomaly_flag": "max",
            "bpfo_magnitude_mean": "mean",
            "bpfi_magnitude_mean": "mean",
            "failure_probability": "mean",
            "predicted_failure_within_24h": "max",
            "predicted_rul_hours": "mean",
        }
    )
    resampled = resampled.dropna(subset=["rms_mean"])

    series = []
    for ts, row in resampled.iterrows():
        fail_prob = row["failure_probability"]
        fail_pred = row["predicted_failure_within_24h"]
        rul = row["predicted_rul_hours"]
        series.append(
            {
                "t": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "rms": r(row["rms_mean"]),
                "peak": r(row["peak_mean"]),
                "kurtosis": r(row["kurtosis_mean"], 3),
                "skew": r(row["skewness_mean"], 3),
                "crest": r(row["crest_factor_mean"], 3),
                "anomaly": r(row["anomaly_score"], 3),
                "flag": bool(row["anomaly_flag"]) if pd.notna(row["anomaly_flag"]) else False,
                "failProb": r(fail_prob, 4) if pd.notna(fail_prob) else None,
                "failPred": int(fail_pred) if pd.notna(fail_pred) else None,
                "rul": r(rul, 1) if pd.notna(rul) else None,
                "bpfo": r(row["bpfo_magnitude_mean"], 5),
                "bpfi": r(row["bpfi_magnitude_mean"], 5),
            }
        )

    recent = []
    for _, row in group.tail(6).iterrows():
        recent.append(
            {
                "t": row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "rms": r(row["rms_mean"]),
                "peak": r(row["peak_mean"]),
                "anomaly": r(row["anomaly_score"], 3),
                "failProb": r(row["failure_probability"], 4)
                if pd.notna(row["failure_probability"])
                else None,
                "rul": r(row["predicted_rul_hours"], 1)
                if pd.notna(row["predicted_rul_hours"])
                else None,
                "flag": bool(row["anomaly_flag"]),
                "failPred": int(row["predicted_failure_within_24h"])
                if pd.notna(row["predicted_failure_within_24h"])
                else None,
            }
        )

    latest_fail = latest_row["failure_probability"]
    latest_pred = latest_row["predicted_failure_within_24h"]
    latest_rul = latest_row["predicted_rul_hours"]

    return {
        "name": machine_name(test_id, bearing_id),
        "assetId": machine_asset(test_id, bearing_id),
        "testId": test_id,
        "bearingId": bearing_id,
        "split": TRAJECTORY_SPLIT.get((test_id, bearing_id)),
        "documentedDamage": DOCUMENTED_DAMAGE.get((test_id, bearing_id)),
        "start": group["timestamp"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S"),
        "end": group["timestamp"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S"),
        "nWindows": int(n),
        "earlyMean": {
            "rms": r(early["rms_mean"].mean()),
            "peak": r(early["peak_mean"].mean()),
            "kurtosis": r(early["kurtosis_mean"].mean(), 3),
            "anomalyScore": r(early["anomaly_score"].mean(), 3),
            "bandEnergy": [r(early[c].mean(), 6) for c in BAND_COLS],
            "bpfo": r(early["bpfo_magnitude_mean"].mean(), 5),
            "bpfi": r(early["bpfi_magnitude_mean"].mean(), 5),
        },
        "lateMean": {
            "rms": r(late["rms_mean"].mean()),
            "peak": r(late["peak_mean"].mean()),
            "anomalyScore": r(late["anomaly_score"].mean(), 3),
            "failureProbability": mean_or_none(late["failure_probability"], 4),
        },
        "latest": {
            "timestamp": latest_row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
            "rms": r(latest_row["rms_mean"]),
            "peak": r(latest_row["peak_mean"]),
            "kurtosis": r(latest_row["kurtosis_mean"], 3),
            "anomalyScore": r(latest_row["anomaly_score"], 3),
            "anomalyFlag": bool(latest_row["anomaly_flag"]),
            "failureProbability": r(latest_fail, 4) if pd.notna(latest_fail) else None,
            "predictedFailure24h": int(latest_pred) if pd.notna(latest_pred) else None,
            "predictedRulHours": r(latest_rul, 1) if pd.notna(latest_rul) else None,
        },
        "latestBands": [r(latest_row[c], 6) for c in BAND_COLS],
        "latestFaultFreq": {
            "ftf": r(latest_row["ftf_magnitude_mean"], 5),
            "bsf": r(latest_row["bsf_magnitude_mean"], 5),
            "bpfo": r(latest_row["bpfo_magnitude_mean"], 5),
            "bpfi": r(latest_row["bpfi_magnitude_mean"], 5),
        },
        "recentWindows": recent,
        "series": series,
    }


def load_metrics() -> dict:
    clf = pd.read_csv(MODELS / "failure_classifier" / "metrics.csv")
    thr = json.loads((MODELS / "failure_classifier" / "threshold.json").read_text(encoding="utf-8"))
    selected = json.loads((MODELS / "isolation_forest" / "selected_model.json").read_text(encoding="utf-8"))
    if_metrics = json.loads((MODELS / "isolation_forest" / "metrics.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(MODELS / "isolation_forest" / "baseline_score_summary.csv")
    rul = pd.read_csv(MODELS / "rul_regressor" / "metrics.csv")

    def clf_split(name: str) -> dict:
        row = clf.loc[clf["split"] == name].iloc[0]
        return {
            "precision": r(row["precision"], 3) if name == "validation" else r(row["precision"], 1)
            if float(row["precision"]) == 1.0
            else r(row["precision"], 3),
            "recall": r(row["recall"], 3),
            "f1": r(row["f1"], 3),
            "rocAuc": r(row["roc_auc"], 3),
            "prAuc": r(row["average_precision"], 3),
            "tn": int(row["tn"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "tp": int(row["tp"]),
        }

    # Match previously published rounded classifier metrics in the UI payload.
    val = clf.loc[clf["split"] == "validation"].iloc[0]
    test = clf.loc[clf["split"] == "test"].iloc[0]
    clf_val = {
        "precision": r(val["precision"], 3),
        "recall": r(val["recall"], 3),
        "f1": r(val["f1"], 3),
        "rocAuc": r(val["roc_auc"], 3),
        "prAuc": r(val["average_precision"], 3),
        "tn": int(val["tn"]),
        "fp": int(val["fp"]),
        "fn": int(val["fn"]),
        "tp": int(val["tp"]),
    }
    clf_test = {
        "precision": float(test["precision"]),
        "recall": r(test["recall"], 3),
        "f1": r(test["f1"], 3),
        "rocAuc": float(test["roc_auc"]) if float(test["roc_auc"]) == 1.0 else r(test["roc_auc"], 3),
        "prAuc": r(test["average_precision"], 3),
        "tn": int(test["tn"]),
        "fp": int(test["fp"]),
        "fn": int(test["fn"]),
        "tp": int(test["tp"]),
    }

    if_test = next(
        m
        for m in if_metrics
        if m["split"] == "test" and str(m["contamination"]) == "1pct"
    )
    rul_val = rul.loc[rul["split"] == "validation"].iloc[0]
    rul_test = rul.loc[rul["split"] == "test"].iloc[0]
    baseline_mean = float(baseline.loc[baseline["baseline_group"] == "baseline", "mean"].iloc[0])

    return {
        "classifierThreshold": float(thr["threshold"]),
        "ifContamination": float(selected["selected_contamination"]),
        "ifBaselineMean": r(baseline_mean, 3),
        "metrics": {
            "classifier": {
                "threshold": float(thr["threshold"]),
                "val": clf_val,
                "test": clf_test,
            },
            "isolationForest": {
                "label": selected["selected_label"],
                "valLateLift": r(selected["validation_stats"]["val_failed_mean_score_lift"], 3),
                "valLateFlagRate": float(selected["validation_stats"]["val_failed_late_flag_rate"]),
                "testProxyF1": r(if_test["f1"], 2),
                "testProxyRecall": r(if_test["recall"], 3),
                "testProxyPrecision": float(if_test["precision"]),
            },
            "rul": {
                "valMae24h": int(round(float(rul_val["mae_last_24h"]))),
                "valMae72h": int(round(float(rul_val["mae_last_72h"]))),
                "testMae24h": int(round(float(rul_test["mae_last_24h"]))),
                "testMae72h": int(round(float(rul_test["mae_last_72h"]))),
                "valMaeFull": int(round(float(rul_val["mae_hours"]))),
                "testMaeFull": int(round(float(rul_test["mae_hours"]))),
            },
        },
    }


def main() -> None:
    scores = load_scores()
    clf = load_classifier()
    rul = load_rul()

    merged = scores.merge(
        clf,
        on=["test_id", "bearing_id", "timestamp"],
        how="left",
    ).merge(
        rul,
        on=["test_id", "bearing_id", "timestamp"],
        how="left",
    )

    machines = {}
    for (test_id, bearing_id), group in merged.groupby(["test_id", "bearing_id"], sort=True):
        key = machine_key(str(test_id), int(bearing_id))
        machines[key] = build_machine(group)

    meta = load_metrics()
    payload = {
        "source": (
            "NASA IMS operational windows (1st/2nd/3rd_test), "
            "12-hour means from scored pipeline artifacts "
            "(Isolation Forest + failure classifier + RUL where available)"
        ),
        "classifierThreshold": meta["classifierThreshold"],
        "ifContamination": meta["ifContamination"],
        "ifBaselineMean": meta["ifBaselineMean"],
        "metrics": meta["metrics"],
        "machines": machines,
    }

    OUT.write_text(
        "window.ANALYTICS_DATA = "
        + json.dumps(payload, separators=(",", ": "))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(machines)} machines -> {OUT}")
    for key, m in machines.items():
        print(
            f"  {key}: {m['assetId']} windows={m['nWindows']} bins={len(m['series'])} "
            f"split={m['split']} damage={m['documentedDamage']!r}"
        )


if __name__ == "__main__":
    main()
