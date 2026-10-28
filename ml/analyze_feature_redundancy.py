import pandas as pd
import numpy as np

FEATURE_FILE = "data/processed/features.csv"

METADATA_COLS = [
    "test_id",
    "bearing_id",
    "timestamp",
    "source_file"
]

print("Loading feature dataset...")
df = pd.read_csv(FEATURE_FILE)

features = df.drop(columns=METADATA_COLS)

print(f"Rows: {len(df)}")
print(f"Features: {len(features.columns)}")


# ============================================================
# 1. CORRELATION MATRIX
# ============================================================

print("\n========== CALCULATING CORRELATIONS ==========")

corr = features.corr()

# Only look at upper triangle so we don't report
# A-B and B-A separately.
upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)


# ============================================================
# 2. VERY HIGH CORRELATION
# ============================================================

threshold = 0.95

high_corr = (
    upper
    .stack()
    .reset_index()
)

high_corr.columns = [
    "feature_1",
    "feature_2",
    "correlation"
]

high_corr["abs_correlation"] = (
    high_corr["correlation"].abs()
)

high_corr = high_corr.sort_values(
    "abs_correlation",
    ascending=False
)

high_corr = high_corr[
    high_corr["abs_correlation"] >= threshold
]

print(
    f"\nFeature pairs with |correlation| >= {threshold}: "
    f"{len(high_corr)}"
)

print("\nTop highly correlated pairs:")

if len(high_corr) > 0:
    print(
        high_corr[
            [
                "feature_1",
                "feature_2",
                "correlation"
            ]
        ]
        .head(100)
        .to_string(index=False)
    )
else:
    print("None found.")


# ============================================================
# 3. EXTREMELY HIGH CORRELATION
# ============================================================

threshold_extreme = 0.99

extreme_corr = high_corr[
    high_corr["abs_correlation"] >= threshold_extreme
]

print(
    f"\n========== |CORRELATION| >= {threshold_extreme} =========="
)

print(
    f"Number of pairs: {len(extreme_corr)}"
)

if len(extreme_corr) > 0:
    print(
        extreme_corr[
            [
                "feature_1",
                "feature_2",
                "correlation"
            ]
        ]
        .to_string(index=False)
    )
else:
    print("None found.")


# ============================================================
# 4. FEATURE GROUP CORRELATION
# ============================================================

print("\n========== FEATURE GROUPS ==========")

groups = {
    "time_domain": [
        c for c in features.columns
        if any(
            c.startswith(prefix)
            for prefix in [
                "mean_",
                "std_",
                "variance_",
                "rms_",
                "min_",
                "max_",
                "peak_",
                "peak_to_peak_",
                "skewness_",
                "kurtosis_",
                "crest_factor_",
                "shape_factor_",
                "impulse_factor_",
                "clearance_factor_",
                "energy_"
            ]
        )
    ],

    "raw_fft": [
        c for c in features.columns
        if any(
            c.startswith(prefix)
            for prefix in [
                "spectral_",
                "dominant_",
                "band_energy_"
            ]
        )
    ],

    "envelope": [
        c for c in features.columns
        if any(
            c.startswith(prefix)
            for prefix in [
                "envelope_",
                "ftf_",
                "bsf_",
                "bpfo_",
                "bpfi_"
            ]
        )
    ]
}

for group_name, columns in groups.items():
    print(
        f"{group_name}: {len(columns)} features"
    )


# ============================================================
# 5. FEATURES WITH MANY STRONG CORRELATIONS
# ============================================================

print("\n========== FEATURES WITH MANY STRONG CORRELATIONS ==========")

corr_abs = corr.abs()

strong_counts = (
    (corr_abs >= 0.95)
    .sum()
    - 1
)

strong_counts = strong_counts.sort_values(
    ascending=False
)

print(
    strong_counts.head(30).to_string()
)


# ============================================================
# 6. SAVE CORRELATION MATRIX
# ============================================================

corr.to_csv(
    "data/processed/feature_correlations.csv"
)

high_corr.to_csv(
    "data/processed/high_correlation_pairs.csv",
    index=False
)

print("\n========== COMPLETE ==========")

print(
    "Saved:"
)

print(
    "data/processed/feature_correlations.csv"
)

print(
    "data/processed/high_correlation_pairs.csv"
)