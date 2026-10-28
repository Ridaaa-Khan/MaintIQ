from pathlib import Path
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, hilbert


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
IMS_ROOT = PROJECT_ROOT / "data" / "raw" / "IMS"

SAMPLING_RATE = 20_000
FILTER_ORDER = 4

CANDIDATE_BANDS = [
    (1000, 3000),
    (3000, 5000),
    (5000, 7000),
    (7000, 9000),
]

BPFI = 296.93
BPFO = 236.40
FREQUENCY_TOLERANCE = 5

NUMBER_OF_SAMPLES = 25

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$"
)

TEST_CONFIGS = [
    ("1st_test", 4, BPFI, "Test 1 Bearing 3 (BPFI - Inner Race)", 0),
    ("2nd_test", 0, BPFO, "Test 2 Bearing 1 (BPFO - Outer Race)", -3),  # Exclude last 2 artifact files
    ("3rd_test", 2, BPFO, "Test 3 Bearing 3 (BPFO - Outer Race)", -2),  # Exclude last 1 artifact file
]


# ============================================================
# DATASET FUNCTIONS
# ============================================================

def find_data_files(test_folder):
    files = []
    for path in test_folder.rglob("*"):
        if path.is_file() and TIMESTAMP_PATTERN.match(path.name):
            files.append(path)
    return sorted(files)


def parse_timestamp(filename):
    return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")


def load_signal(file_path, channel_index):
    data = np.loadtxt(file_path)
    return data[:, channel_index]


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def bandpass_filter(signal, low_cutoff, high_cutoff):
    nyquist = SAMPLING_RATE / 2
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    sos = butter(FILTER_ORDER, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def calculate_envelope(signal):
    analytic_signal = hilbert(signal)
    return np.abs(analytic_signal)


def calculate_fft(signal):
    signal = signal - np.mean(signal)
    n = len(signal)
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1 / SAMPLING_RATE)
    magnitude = np.abs(fft_values) / n
    if n > 1:
        magnitude[1:-1] *= 2
    return frequencies, magnitude


# ============================================================
# FAULT RATIO METRIC
# ============================================================

def calculate_fault_ratio(frequencies, magnitude, target_freq):
    target_mask = (
        (frequencies >= target_freq - FREQUENCY_TOLERANCE)
        & (frequencies <= target_freq + FREQUENCY_TOLERANCE)
    )

    target_indices = np.where(target_mask)[0]
    if len(target_indices) == 0:
        return np.nan

    target_magnitude = np.max(magnitude[target_indices])

    neighborhood_mask = (
        (frequencies >= target_freq - 25)
        & (frequencies <= target_freq + 25)
        & ~target_mask
    )

    neighborhood = magnitude[neighborhood_mask]
    if len(neighborhood) == 0:
        return np.nan

    local_median = np.median(neighborhood)
    if local_median <= 0:
        return np.nan

    return target_magnitude / local_median


def analyze_file(file_path, channel_index, target_freq):
    signal = load_signal(file_path, channel_index)
    signal = signal - np.mean(signal)

    results = {}
    for low, high in CANDIDATE_BANDS:
        filtered = bandpass_filter(signal, low, high)
        envelope = calculate_envelope(filtered)
        frequencies, magnitude = calculate_fft(envelope)
        ratio = calculate_fault_ratio(frequencies, magnitude, target_freq)
        results[f"{low}-{high}"] = ratio

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("RESONANCE BAND TRAJECTORY VALIDATION (BPFI & BPFO)")
    print("=" * 70)

    fig, axes = plt.subplots(len(TEST_CONFIGS), 1, figsize=(14, 4 * len(TEST_CONFIGS)))
    if len(TEST_CONFIGS) == 1:
        axes = [axes]

    for idx, (test_name, ch_idx, target_freq, desc, offset) in enumerate(TEST_CONFIGS):
        print("\n" + "=" * 70)
        print(f"VALIDATING: {desc}")
        print("=" * 70)

        test_folder = IMS_ROOT / test_name
        files = find_data_files(test_folder)

        if offset < 0:
            files = files[:offset]

        total_files = len(files)
        print(f"Validating across {total_files} operational files (artifacts excluded).")

        sample_indices = np.linspace(0, total_files - 1, NUMBER_OF_SAMPLES, dtype=int)
        sampled_files = [files[i] for i in sample_indices]

        timestamps = []
        band_results = {f"{low}-{high}": [] for low, high in CANDIDATE_BANDS}

        for file_path in sampled_files:
            timestamp = parse_timestamp(file_path.name)
            res = analyze_file(file_path, ch_idx, target_freq)
            timestamps.append(timestamp)
            for band in band_results:
                band_results[band].append(res[band])

        print("\nSTATISTICAL SUMMARY:")
        print(f"{'Band (Hz)':<15} {'Median Ratio':<15} {'Mean Ratio':<15} {'Max Ratio':<15}")
        print("-" * 60)

        for band, vals in band_results.items():
            arr = np.array(vals)
            print(
                f"{band:<15} "
                f"{np.nanmedian(arr):<15.2f} "
                f"{np.nanmean(arr):<15.2f} "
                f"{np.nanmax(arr):<15.2f}"
            )

        ax = axes[idx]
        for band, vals in band_results.items():
            ax.plot(timestamps, vals, marker="o", markersize=3, label=f"{band} Hz")

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Fault / Local Noise Ratio")
        ax.set_title(f"{desc}\nFault Frequency Progression Across Degradation Run")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "resonance_band_trajectory_validation.png"
    plt.savefig(plot_path)
    print(f"\nSaved trajectory validation plot to: {plot_path}")
    print("\nResonance band validation complete.")


if __name__ == "__main__":
    main()