"""
fft_analysis.py  [NON-AUTHORITATIVE — EXPLORATORY ONLY]
=======================================================
This script was used during early exploration to identify dominant
spectral peaks and estimate fault frequencies.  It is NOT part of
the production feature extraction pipeline.

Authoritative source of truth
------------------------------
  extract_features.py
    SAMPLING_RATE = 20000 Hz (official IMS)
    BSF  computed from bearing geometry (~140 Hz at 1x)
    BPFI, BPFO, FTF computed from bearing geometry

Do NOT use this script to generate features or drive any
modelling decision.  It is kept for reference only.
"""
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


PROJECT_ROOT = Path(__file__).resolve().parent

IMS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IMS"
)

SAMPLING_RATE = 20_000

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$"
)


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


def calculate_fft(signal):

    # Remove DC component
    signal = signal - np.mean(signal)

    n = len(signal)

    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / SAMPLING_RATE
    )

    magnitude = np.abs(fft_values) / n

    if n > 1:
        magnitude[1:-1] *= 2

    return frequencies, magnitude


# =========================================================
# FIND DATA
# =========================================================

test_folder = IMS_ROOT / "1st_test"

files = find_data_files(test_folder)

print(f"Total files: {len(files)}")


# =========================================================
# SELECT EARLY AND LATE FILE
# =========================================================

early_file = files[0]
late_file = files[-1]

print("\nEarly:")
print(early_file.name)
print(parse_timestamp(early_file.name))

print("\nLate:")
print(late_file.name)
print(parse_timestamp(late_file.name))


# =========================================================
# READ BEARING 3 - CHANNEL 5
# =========================================================

early_data = np.loadtxt(early_file)
late_data = np.loadtxt(late_file)

early_signal = early_data[:, 4]
late_signal = late_data[:, 4]


# =========================================================
# FFT
# =========================================================

early_freq, early_mag = calculate_fft(
    early_signal
)

late_freq, late_mag = calculate_fft(
    late_signal
)


# =========================================================
# LIMIT ANALYSIS TO 0-1000 Hz
# =========================================================

max_frequency = 1000

early_mask = early_freq <= max_frequency
late_mask = late_freq <= max_frequency


early_f = early_freq[early_mask]
early_m = early_mag[early_mask]

late_f = late_freq[late_mask]
late_m = late_mag[late_mask]


# =========================================================
# FIND PEAKS
# =========================================================

# Prominence prevents tiny random fluctuations
# from being treated as meaningful peaks.

early_peaks, early_properties = find_peaks(
    early_m,
    prominence=0.002
)

late_peaks, late_properties = find_peaks(
    late_m,
    prominence=0.002
)


# =========================================================
# SORT PEAKS BY MAGNITUDE
# =========================================================

early_sorted = early_peaks[
    np.argsort(
        early_m[early_peaks]
    )[::-1]
]

late_sorted = late_peaks[
    np.argsort(
        late_m[late_peaks]
    )[::-1]
]


# =========================================================
# PRINT TOP PEAKS
# =========================================================

print("\n" + "=" * 70)
print("TOP EARLY FREQUENCY PEAKS")
print("=" * 70)

for peak_index in early_sorted[:20]:

    print(
        f"{early_f[peak_index]:8.2f} Hz"
        f"   magnitude = "
        f"{early_m[peak_index]:.6f}"
    )


print("\n" + "=" * 70)
print("TOP LATE FREQUENCY PEAKS")
print("=" * 70)

for peak_index in late_sorted[:20]:

    print(
        f"{late_f[peak_index]:8.2f} Hz"
        f"   magnitude = "
        f"{late_m[peak_index]:.6f}"
    )


# =========================================================
# EXPECTED BEARING FREQUENCIES
# =========================================================

shaft_frequency = 2000 / 60

print("\n" + "=" * 70)
print("REFERENCE FREQUENCIES")
print("=" * 70)

print(
    f"Shaft frequency: {shaft_frequency:.2f} Hz"
)

print("Approximate published values:")
print("BPFO: ~236 Hz")
print("BPFI: ~297 Hz")
print("BSF-related: ~278 Hz")


# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(14, 6))

plt.plot(
    early_f,
    early_m,
    label="Early"
)

plt.plot(
    late_f,
    late_m,
    label="Late"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.title(
    "Test 1 - Bearing 3 - Channel 5\n"
    "FFT with Peak Detection"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()