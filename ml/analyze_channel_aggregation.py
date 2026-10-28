import pandas as pd
import numpy as np

FEATURE_FILE = "data/processed/features.csv"

df = pd.read_csv(FEATURE_FILE)

print("Rows:", len(df))


# ============================================================
# FIND MEAN/MAX PAIRS
# ============================================================

mean_cols = [
    c for c in df.columns
    if c.endswith("_mean")
]

pairs = []

for mean_col in mean_cols:
    base = mean_col[:-5]
    max_col = base + "_max"

    if max_col in df.columns:
        pairs.append((mean_col, max_col))


print("\n========== MEAN/MAX PAIRS ==========")
print("Pairs found:", len(pairs))


# ============================================================
# ANALYZE DIFFERENCE
# ============================================================

results = []

for mean_col, max_col in pairs:

    difference = (
        df[mean_col] - df[max_col]
    ).abs()

    nonzero = difference > 1e-12

    results.append({
        "feature": mean_col[:-5],
        "different_rows": nonzero.sum(),
        "different_percent": nonzero.mean() * 100,
        "mean_absolute_difference": difference.mean(),
        "max_absolute_difference": difference.max()
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "different_percent",
    ascending=False
)


print("\n========== ALL DATA ==========")

print(
    results_df.to_string(index=False)
)


# ============================================================
# TEST-SPECIFIC ANALYSIS
# ============================================================

print("\n========== TEST-SPECIFIC DIFFERENCES ==========")

for test_id in df["test_id"].unique():

    subset = df[df["test_id"] == test_id]

    print(f"\n--- {test_id} ---")

    test_results = []

    for mean_col, max_col in pairs:

        difference = (
            subset[mean_col] -
            subset[max_col]
        ).abs()

        nonzero = difference > 1e-12

        test_results.append({
            "feature": mean_col[:-5],
            "different_rows": nonzero.sum(),
            "different_percent": nonzero.mean() * 100
        })

    test_results_df = pd.DataFrame(
        test_results
    )

    print(
        test_results_df
        .sort_values(
            "different_percent",
            ascending=False
        )
        .to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "data/processed/channel_aggregation_analysis.csv",
    index=False
)

print(
    "\nSaved:"
    "\n"
    "data/processed/channel_aggregation_analysis.csv"
)