"""
envelope_analysis.py  [NON-AUTHORITATIVE — EXPLORATORY ONLY]
=============================================================
This script was used to explore resonance bands and envelope
spectra during the early phase of the project.  Its parameters
(band 3-6 kHz, BSF=139.92 Hz) are NOT the values used by the
production pipeline.

Authoritative source of truth
------------------------------
  extract_features.py
    ENVELOPE_LOW  = 1000 Hz
    ENVELOPE_HIGH = 3000 Hz
    BSF           = ~140 Hz  (computed from geometry)
    SAMPLING_RATE = 20000 Hz (official IMS)

Do NOT use this script to generate features or drive any
modelling decision.  It is kept for reference only.
"""
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMS_ROOT = PROJECT_ROOT / "data" / "raw" / "IMS"

SAMPLING_RATE = 20_000

TEST_NAME = "1st_test"

# Test 1:
# Bearing 3 = Channels 5 and 6
#
# Python index:
# Channel 5 -> index 4
CHANNEL_INDEX = 4

# Exploratory resonance band.
#
# From our previous raw FFT investigation, Test 1 Bearing 3
# showed substantial high-frequency activity around several kHz.
#
# We start with 3-6 kHz as an exploratory band.
LOW_CUTOFF = 3000
HIGH_CUTOFF = 6000

FILTER_ORDER = 4

# Frequencies we want to investigate in the envelope spectrum
SHAFT_FREQUENCY = 33.33
FTF = 14.78
BPFO = 236.40
BPFI = 296.93
BSF = 139.92


TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$"
)


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
    return datetime.strptime(
        filename,
        "%Y.%m.%d.%H.%M.%S"
    )


def load_signal(file_path, channel_index):
    data = np.loadtxt(file_path)

    signal = data[:, channel_index]

    return signal


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
    nyquist = sampling_rate / 2

    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    sos = butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


def calculate_envelope(signal):
    analytic_signal = hilbert(signal)

    envelope = np.abs(analytic_signal)

    return envelope


def calculate_fft(signal, sampling_rate):
    # Remove DC component
    signal = signal - np.mean(signal)

    n = len(signal)

    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / sampling_rate
    )

    magnitude = np.abs(fft_values) / n

    if n > 1:
        magnitude[1:-1] *= 2

    return frequencies, magnitude


# ============================================================
# ENVELOPE SPECTRUM ANALYSIS
# ============================================================

def find_near_frequency(
    frequencies,
    magnitude,
    target_frequency,
    tolerance=5
):

    mask = (
        (frequencies >= target_frequency - tolerance)
        &
        (frequencies <= target_frequency + tolerance)
    )

    if not np.any(mask):
        return None

    local_indices = np.where(mask)[0]

    best_index = local_indices[
        np.argmax(magnitude[local_indices])
    ]

    return (
        frequencies[best_index],
        magnitude[best_index]
    )


def print_frequency_comparison(
    frequencies,
    magnitude,
    label
):

    print("\n" + "=" * 70)
    print(f"{label} ENVELOPE FREQUENCY CHECK")
    print("=" * 70)

    reference_frequencies = {
        "FTF": FTF,
        "BSF": BSF,
        "BPFO": BPFO,
        "BPFI": BPFI,
        "Shaft": SHAFT_FREQUENCY
    }

    for name, target in reference_frequencies.items():

        result = find_near_frequency(
            frequencies,
            magnitude,
            target,
            tolerance=5
        )

        if result is None:
            print(
                f"{name:8s} "
                f"expected = {target:7.2f} Hz   "
                f"no peak found"
            )

        else:

            observed_frequency, observed_magnitude = result

            print(
                f"{name:8s} "
                f"expected = {target:7.2f} Hz   "
                f"observed = {observed_frequency:7.2f} Hz   "
                f"magnitude = {observed_magnitude:.6f}"
            )


def print_top_peaks(
    frequencies,
    magnitude,
    label,
    max_frequency=500
):

    mask = frequencies <= max_frequency

    f = frequencies[mask]
    m = magnitude[mask]

    peaks, properties = find_peaks(
        m,
        prominence=0.0001
    )

    sorted_peaks = peaks[
        np.argsort(m[peaks])[::-1]
    ]

    print("\n" + "=" * 70)
    print(f"TOP {label.upper()} ENVELOPE PEAKS")
    print("=" * 70)

    for peak_index in sorted_peaks[:20]:

        print(
            f"{f[peak_index]:8.2f} Hz   "
            f"magnitude = {m[peak_index]:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

test_folder = IMS_ROOT / TEST_NAME

files = find_data_files(test_folder)

print(f"Total files: {len(files)}")


# ------------------------------------------------------------
# Select earliest and latest vibration snapshots
# ------------------------------------------------------------

early_file = files[0]
late_file = files[-1]

print("\nEarly:")
print(early_file.name)
print(parse_timestamp(early_file.name))

print("\nLate:")
print(late_file.name)
print(parse_timestamp(late_file.name))


# ------------------------------------------------------------
# Load signals
# ------------------------------------------------------------

early_signal = load_signal(
    early_file,
    CHANNEL_INDEX
)

late_signal = load_signal(
    late_file,
    CHANNEL_INDEX
)


# ------------------------------------------------------------
# Remove mean before filtering
# ------------------------------------------------------------

early_signal = early_signal - np.mean(early_signal)
late_signal = late_signal - np.mean(late_signal)


# ------------------------------------------------------------
# Band-pass filtering
# ------------------------------------------------------------

early_filtered = bandpass_filter(
    early_signal,
    LOW_CUTOFF,
    HIGH_CUTOFF,
    SAMPLING_RATE
)

late_filtered = bandpass_filter(
    late_signal,
    LOW_CUTOFF,
    HIGH_CUTOFF,
    SAMPLING_RATE
)


# ------------------------------------------------------------
# Hilbert envelope
# ------------------------------------------------------------

early_envelope = calculate_envelope(
    early_filtered
)

late_envelope = calculate_envelope(
    late_filtered
)


# ------------------------------------------------------------
# Envelope FFT
# ------------------------------------------------------------

early_freq, early_mag = calculate_fft(
    early_envelope,
    SAMPLING_RATE
)

late_freq, late_mag = calculate_fft(
    late_envelope,
    SAMPLING_RATE
)


# ------------------------------------------------------------
# Print expected-frequency comparisons
# ------------------------------------------------------------

print_frequency_comparison(
    early_freq,
    early_mag,
    "EARLY"
)

print_frequency_comparison(
    late_freq,
    late_mag,
    "LATE"
)


# ------------------------------------------------------------
# Print top envelope peaks
# ------------------------------------------------------------

print_top_peaks(
    early_freq,
    early_mag,
    "Early",
    max_frequency=500
)

print_top_peaks(
    late_freq,
    late_mag,
    "Late",
    max_frequency=500
)


# ============================================================
# PLOTS
# ============================================================

# ------------------------------------------------------------
# 1. Filtered vibration
# ------------------------------------------------------------

time = np.arange(len(early_filtered)) / SAMPLING_RATE

plt.figure(figsize=(14, 6))

plt.plot(
    time,
    early_filtered,
    label="Early"
)

plt.plot(
    time,
    late_filtered,
    label="Late"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.title(
    "Test 1 - Bearing 3 - Channel 5\n"
    "Band-pass Filtered Vibration"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 2. Envelope
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    time,
    early_envelope,
    label="Early"
)

plt.plot(
    time,
    late_envelope,
    label="Late"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Envelope Amplitude")

plt.title(
    "Test 1 - Bearing 3 - Channel 5\n"
    "Hilbert Envelope"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 3. Envelope spectrum
# ------------------------------------------------------------

max_frequency = 500

early_mask = early_freq <= max_frequency
late_mask = late_freq <= max_frequency

plt.figure(figsize=(14, 6))

plt.plot(
    early_freq[early_mask],
    early_mag[early_mask],
    label="Early"
)

plt.plot(
    late_freq[late_mask],
    late_mag[late_mask],
    label="Late"
)

# Reference frequencies

plt.axvline(
    FTF,
    linestyle="--",
    label="FTF"
)

plt.axvline(
    BSF,
    linestyle="--",
    label="BSF"
)

plt.axvline(
    BPFO,
    linestyle="--",
    label="BPFO"
)

plt.axvline(
    BPFI,
    linestyle="--",
    label="BPFI"
)

plt.axvline(
    SHAFT_FREQUENCY,
    linestyle="--",
    label="Shaft"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Envelope FFT Magnitude")

plt.title(
    "Test 1 - Bearing 3 - Channel 5\n"
    "Envelope Spectrum"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()