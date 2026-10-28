from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "features_refined.csv"

METADATA_COLS = [
    "test_id",
    "bearing_id",
    "timestamp",
    "source_file"
]

REMOVE_FEATURES_CANDIDATES = [
    # Signal mean
    "mean_mean",
    "mean_max",

    # RMS already represents signal magnitude.
    # Because signal mean ≈ 0:
    # RMS ≈ standard deviation.
    "std_mean",
    "std_max",

    # Variance / energy / spectral energy are mathematically
    # related for the current feature extraction.
    "variance_mean",
    "variance_max",
    "energy_mean",
    "energy_max",
    "spectral_energy_mean",
    "spectral_energy_max",

    # Dominant magnitude is less useful than the explicit
    # frequency-domain and bearing-frequency features.
    "dominant_magnitude_mean",
    "dominant_magnitude_max",
]


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    print("Original shape:", df.shape)

    remove_features = [
        col for col in REMOVE_FEATURES_CANDIDATES
        if col in df.columns
    ]

    refined_df = df.drop(columns=remove_features)

    feature_count_before = len(df.columns) - len(METADATA_COLS)
    feature_count_after = len(refined_df.columns) - len(METADATA_COLS)

    print("\n========== FEATURE REFINEMENT ==========")
    print("Features before:", feature_count_before)
    print("Features removed:", len(remove_features))
    print("Features after:", feature_count_after)
    print("\nRemoved:")
    for feature in remove_features:
        print(" -", feature)
    print("\nOriginal shape:", df.shape)
    print("Refined shape:", refined_df.shape)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    refined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved refined dataset:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()