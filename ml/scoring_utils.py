"""
Shared scoring / data-quality helpers for inference.

All three model infer helpers must:
  1. Require the exact trained feature list
  2. Reject empty tables
  3. Replace +/- inf before predict
  4. Report NaN counts (XGBoost can treat NaN as missing; Isolation
     Forest imputes)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


class FeatureSchemaError(ValueError):
    """Input table cannot be scored with the saved model features."""


@dataclass
class SchemaReport:
    n_rows: int
    n_required_features: int
    n_missing_features: int
    n_extra_columns: int
    n_inf_cells: int
    n_nan_cells: int
    missing_features: list[str]
    extra_columns: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def validate_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    *,
    allow_nan: bool = True,
    replace_inf: bool = True,
) -> tuple[pd.DataFrame, SchemaReport]:
    """
    Check that `df` contains every required feature.

    Returns a copy with inf replaced by NaN when replace_inf=True.
    Extra columns are kept (they are not fed to the model).
    """
    if df is None or len(df) == 0:
        raise FeatureSchemaError("Input table is empty.")

    missing = [col for col in features if col not in df.columns]
    extra = [col for col in df.columns if col not in features]
    if missing:
        preview = missing[:12]
        suffix = "..." if len(missing) > 12 else ""
        raise FeatureSchemaError(
            "Input is missing required model features: "
            f"{preview}{suffix} ({len(missing)} missing, "
            f"{len(features)} required)."
        )

    result = df.copy()
    X = result[features]
    numeric = X.apply(pd.to_numeric, errors="coerce")

    inf_mask = ~np.isfinite(numeric.to_numpy(dtype=float)) & numeric.notna().to_numpy()
    n_inf = int(inf_mask.sum())
    if n_inf and replace_inf:
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        result[features] = numeric

    n_nan = int(numeric.isna().sum().sum())
    if n_nan and not allow_nan:
        raise FeatureSchemaError(
            f"Input has {n_nan} NaN feature cells. "
            "This model was trained without missing values."
        )

    report = SchemaReport(
        n_rows=len(result),
        n_required_features=len(features),
        n_missing_features=0,
        n_extra_columns=len(extra),
        n_inf_cells=n_inf,
        n_nan_cells=n_nan,
        missing_features=[],
        extra_columns=extra[:20],
    )
    return result, report
