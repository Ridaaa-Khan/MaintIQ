import pandas as pd
import numpy as np

FEATURE_FILE = "data/processed/features_temporal.csv"

df = pd.read_csv(FEATURE_FILE)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== COLUMNS ==========")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3}. {col}")

metadata_cols = [
    "test_id",
    "bearing_id",
    "timestamp",
    "source_file"
]

feature_cols = [
    c for c in df.columns
    if c not in metadata_cols
]

print("\n========== FEATURE COUNT ==========")
print("Metadata columns:", len(metadata_cols))
print("Feature columns:", len(feature_cols))

print("\n========== DATA TYPES ==========")
print(df.dtypes.value_counts())

print("\n========== MISSING VALUES ==========")
nan_counts = df.isna().sum()
nan_features = nan_counts[nan_counts > 0]

if len(nan_features) == 0:
    print("No NaN values.")
else:
    print(nan_features)

print("\n========== INFINITE VALUES ==========")
numeric_df = df[feature_cols].select_dtypes(include=np.number)

inf_counts = np.isinf(numeric_df).sum()
inf_features = inf_counts[inf_counts > 0]

if len(inf_features) == 0:
    print("No infinite values.")
else:
    print(inf_features)

print("\n========== CONSTANT FEATURES ==========")
constant_features = []

for col in feature_cols:
    if df[col].nunique(dropna=False) <= 1:
        constant_features.append(col)

if constant_features:
    print(constant_features)
else:
    print("No constant features.")

print("\n========== NEAR-CONSTANT FEATURES ==========")
near_constant = []

for col in feature_cols:
    counts = df[col].value_counts(normalize=True, dropna=False)

    if counts.iloc[0] >= 0.999:
        near_constant.append(
            (col, counts.iloc[0])
        )

if near_constant:
    for col, ratio in near_constant:
        print(f"{col}: {ratio:.4%}")
else:
    print("No near-constant features.")

print("\n========== DUPLICATE FEATURES ==========")
duplicate_features = []

for i, col1 in enumerate(feature_cols):
    for col2 in feature_cols[i + 1:]:
        if df[col1].equals(df[col2]):
            duplicate_features.append((col1, col2))

if duplicate_features:
    for pair in duplicate_features:
        print(pair)
else:
    print("No exact duplicate features.")

print("\n========== FEATURE STATISTICS ==========")
stats = df[feature_cols].describe().T

print(
    stats[
        [
            "mean",
            "std",
            "min",
            "max"
        ]
    ].to_string()
)

print("\n========== TEST COVERAGE ==========")
print(df["test_id"].value_counts().sort_index())

print("\n========== BEARING COVERAGE ==========")
print(
    df.groupby(
        ["test_id", "bearing_id"]
    ).size()
)

print("\n========== TIMESTAMP CHECK ==========")

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Earliest:", df["timestamp"].min())
print("Latest:", df["timestamp"].max())

print("\n========== ROW ORDER CHECK ==========")

ordered = True

for (test_id, bearing_id), group in df.groupby(
    ["test_id", "bearing_id"]
):
    if not group["timestamp"].is_monotonic_increasing:
        print(
            f"NOT SORTED: Test {test_id}, Bearing {bearing_id}"
        )
        ordered = False

if ordered:
    print("All test/bearing groups are chronologically ordered.")

print("\n========== SAMPLE ROWS ==========")
print(df.head(3).to_string())

print("\n========== INSPECTION COMPLETE ==========")