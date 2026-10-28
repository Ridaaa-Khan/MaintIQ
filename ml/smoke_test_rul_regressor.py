"""
Smoke test for the XGBoost RUL regressor.

Run:
    python smoke_test_rul_regressor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from rul_regressor_infer import load_regressor, score_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "rul_regressor"
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "features_val.csv"


def main() -> int:
    print("=" * 60)
    print("RUL REGRESSOR SMOKE TEST")
    print("=" * 60)

    required = [
        MODEL_DIR / "xgboost_rul_regressor.joblib",
        MODEL_DIR / "feature_columns.json",
        MODEL_DIR / "selected_model.json",
        MODEL_DIR / "experiment_report.json",
        MODEL_DIR / "metrics.csv",
        MODEL_DIR / "trajectory_metrics.csv",
    ]
    for path in required:
        if not path.exists():
            print(f"FAIL missing artifact: {path}")
            return 1
        print(f"OK  artifact: {path.name}")

    regressor = load_regressor(MODEL_DIR)
    print(f"OK  loaded model (features={len(regressor['features'])})")

    with (MODEL_DIR / "selected_model.json").open(encoding="utf-8") as f:
        selected = json.load(f)

    verification = selected.get("verification", {})
    if not verification:
        print("FAIL selected_model.json missing verification")
        return 1

    for key, value in verification.items():
        print(f"    {key}: {value}")

    metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
    val = metrics.loc[metrics["split"] == "validation"].iloc[0]
    test = metrics.loc[metrics["split"] == "test"].iloc[0]

    print(
        f"OK  val  MAE/RMSE/R2 = "
        f"{val['mae_hours']:.2f}/"
        f"{val['rmse_hours']:.2f}/"
        f"{val['r2']:.3f}"
        f"  last72={val.get('mae_last_72h', float('nan'))}"
    )
    print(
        f"OK  test MAE/RMSE/R2 = "
        f"{test['mae_hours']:.2f}/"
        f"{test['rmse_hours']:.2f}/"
        f"{test['r2']:.3f}"
        f"  last72={test.get('mae_last_72h', float('nan'))}"
    )

    if pd.isna(val.get("mae_last_72h")) or float(val["mae_last_72h"]) >= 40.0:
        print("FAIL validation last-72h MAE >= 40 hours")
        return 1
    if pd.isna(test.get("mae_last_72h")) or float(test["mae_last_72h"]) >= 80.0:
        print("FAIL test last-72h MAE >= 80 hours")
        return 1
    if pd.isna(test.get("mae_last_24h")) or float(test["mae_last_24h"]) >= 20.0:
        print("FAIL test last-24h MAE >= 20 hours")
        return 1

    val_df = pd.read_csv(VAL_FILE)
    scored = score_dataframe(val_df.head(50), regressor)
    if "predicted_rul_hours" not in scored.columns:
        print("FAIL inference helper missing predicted_rul_hours")
        return 1
    if scored["predicted_rul_hours"].isna().any():
        print("FAIL NaN RUL predictions from inference helper")
        return 1
    if (scored["predicted_rul_hours"] < 0).any():
        print("FAIL negative RUL predictions from inference helper")
        return 1

    print("OK  inference helper scored a validation sample")

    if not verification.get("all_critical_passed", False):
        print("WARN verification.all_critical_passed is False")
    else:
        print("OK  all critical verification checks passed")

    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
