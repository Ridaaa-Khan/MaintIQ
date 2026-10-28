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

# Candidate resonance bands
CANDIDATE_BANDS = [
    (1000, 3000),
    (3000, 5000),
    (5000, 7000),
    (7000, 9000),
]

# Theoretical bearing frequencies
FTF = 14.78
BSF = 139.92
BPFO = 236.40
BPFI = 296.93
SHAFT = 33.33

FREQUENCY_TOLERANCE = 5

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$"
)

# Test cases to evaluate: (test_folder, channel_index, target_freq, description, offset_from_end)
TEST_CONFIGS = [
    ("1st_test", 4, BPFI, "Test 1 Bearing 3 (BPFI - Inner Race)", 0),
    ("2nd_test", 0, BPFO, "Test 2 Bearing 1 (BPFO - Outer Race)", -3),  # Skip last 2 shutdown artifact files
    ("3rd_test", 2, BPFO, "Test 3 Bearing 3 (BPFO - Outer Race)", -2),  # Skip last 1 shutdown artifact file
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
# CHARACTERISTIC FAULT METRICS (BPFI & BPFO)
# ============================================================

def calculate_fault_metrics(frequencies, magnitude, target_freq):
    target_mask = (
        (frequencies >= target_freq - FREQUENCY_TOLERANCE)
        & (frequencies <= target_freq + FREQUENCY_TOLERANCE)
    )

    target_indices = np.where(target_mask)[0]
    if len(target_indices) == 0:
        return None

    target_index = target_indices[np.argmax(magnitude[target_indices])]
    target_frequency = frequencies[target_index]
    target_magnitude = magnitude[target_index]

    neighborhood_mask = (
        (frequencies >= target_freq - 25)
        & (frequencies <= target_freq + 25)
        & ~target_mask
    )

    neighborhood_values = magnitude[neighborhood_mask]
    if len(neighborhood_values) == 0:
        return None

    local_noise = np.median(neighborhood_values)
    ratio = (target_magnitude / local_noise) if local_noise > 0 else np.inf

    return {
        "frequency": target_frequency,
        "magnitude": target_magnitude,
        "local_noise": local_noise,
        "ratio": ratio
    }


def analyze_band(signal, low_cutoff, high_cutoff, target_freq):
    filtered = bandpass_filter(signal, low_cutoff, high_cutoff)
    envelope = calculate_envelope(filtered)
    frequencies, magnitude = calculate_fft(envelope)
    metrics = calculate_fault_metrics(frequencies, magnitude, target_freq)
    return frequencies, magnitude, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("RESONANCE BAND SELECTION ACROSS BPFI AND BPFO FAILURES")
    print("=" * 70)

    fig, axes = plt.subplots(len(TEST_CONFIGS), 1, figsize=(14, 4 * len(TEST_CONFIGS)))
    if len(TEST_CONFIGS) == 1:
        axes = [axes]

    for idx, (test_name, ch_idx, target_freq, desc, offset) in enumerate(TEST_CONFIGS):
        print("\n" + "=" * 70)
        print(f"ANALYZING: {desc}")
        print("=" * 70)

        test_folder = IMS_ROOT / test_name
        files = find_data_files(test_folder)

        target_file_idx = len(files) - 1 + offset
        target_file = files[target_file_idx]

        print(f"Total files in {test_name}: {len(files)}")
        print(f"Selected file for analysis: {target_file.name} (index {target_file_idx})")

        signal = load_signal(target_file, ch_idx)
        signal = signal - np.mean(signal)

        results = []
        for low, high in CANDIDATE_BANDS:
            freqs, mag, metrics = analyze_band(signal, low, high, target_freq)
            if metrics is not None:
                results.append({
                    "band": f"{low}-{high} Hz",
                    "low": low,
                    "high": high,
                    "frequency": metrics["frequency"],
                    "magnitude": metrics["magnitude"],
                    "noise": metrics["local_noise"],
                    "ratio": metrics["ratio"],
                    "frequencies": freqs,
                    "magnitude_array": mag
                })

        print(f"\nSUMMARY FOR {desc}:")
        print(f"{'Band':<15} {'Fault Freq (Hz)':<18} {'Magnitude':<14} {'Local Noise':<14} {'Ratio':<10}")
        print("-" * 71)
        for r in results:
            print(
                f"{r['band']:<15} "
                f"{r['frequency']:<18.2f} "
                f"{r['magnitude']:<14.6f} "
                f"{r['noise']:<14.6f} "
                f"{r['ratio']:<10.2f}"
            )

        # Plot envelope spectrum
        ax = axes[idx]
        for r in results:
            freqs = r["frequencies"]
            mag = r["magnitude_array"]
            mask = freqs <= 500
            ax.plot(freqs[mask], mag[mask], label=r["band"])

        ax.axvline(target_freq, linestyle="--", color="black", label=f"Target ({target_freq} Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Envelope FFT Magnitude")
        ax.set_title(f"{desc}\nEnvelope Spectra Across Resonance Candidate Bands")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "resonance_band_selection_comparison.png"
    plt.savefig(plot_path)
    print(f"\nSaved comparison plot to: {plot_path}")
    print("\nResonance band analysis complete.")


if __name__ == "__main__":
    main()