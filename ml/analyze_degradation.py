from pathlib import Path
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt


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


def calculate_features(signal):

    rms = np.sqrt(np.mean(signal ** 2))

    peak = np.max(np.abs(signal))

    peak_to_peak = np.ptp(signal)

    std = np.std(signal)

    crest_factor = peak / rms if rms != 0 else 0

    # Pearson kurtosis
    centered = signal - np.mean(signal)

    fourth_moment = np.mean(centered ** 4)

    variance = np.mean(centered ** 2)

    if variance != 0:
        kurtosis = fourth_moment / (variance ** 2)
    else:
        kurtosis = 0

    return {
        "rms": rms,
        "std": std,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis
    }


# ---------------------------------------------------------
# Test 1
# ---------------------------------------------------------

test_folder = IMS_ROOT / "1st_test"

files = find_data_files(test_folder)

print(f"Total files: {len(files)}")


# Test 1:
# Bearing 3 = Channels 5 and 6
channel_indices = [4, 5]


# ---------------------------------------------------------
# Store results
# ---------------------------------------------------------

timestamps = []

channel_5_features = []
channel_6_features = []


# ---------------------------------------------------------
# Process every measurement
# ---------------------------------------------------------

for index, file in enumerate(files):

    data = np.loadtxt(file)

    timestamp = parse_timestamp(file.name)

    timestamps.append(timestamp)

    channel_5 = data[:, channel_indices[0]]
    channel_6 = data[:, channel_indices[1]]

    channel_5_features.append(
        calculate_features(channel_5)
    )

    channel_6_features.append(
        calculate_features(channel_6)
    )

    if (index + 1) % 200 == 0:
        print(f"Processed {index + 1}/{len(files)}")


# ---------------------------------------------------------
# Convert to numpy arrays
# ---------------------------------------------------------

rms_5 = np.array([
    x["rms"] for x in channel_5_features
])

rms_6 = np.array([
    x["rms"] for x in channel_6_features
])

peak_5 = np.array([
    x["peak"] for x in channel_5_features
])

peak_6 = np.array([
    x["peak"] for x in channel_6_features
])

kurtosis_5 = np.array([
    x["kurtosis"] for x in channel_5_features
])

kurtosis_6 = np.array([
    x["kurtosis"] for x in channel_6_features
])


# ---------------------------------------------------------
# Print selected measurements
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SELECTED FEATURE VALUES")
print("=" * 70)

selected_indices = [
    0,
    len(files) // 4,
    len(files) // 2,
    (3 * len(files)) // 4,
    len(files) - 1
]

for index in selected_indices:

    print("\nFile:", files[index].name)
    print("Time:", timestamps[index])

    print(
        f"Channel 5 → "
        f"RMS={rms_5[index]:.4f}, "
        f"Peak={peak_5[index]:.4f}, "
        f"Kurtosis={kurtosis_5[index]:.4f}"
    )

    print(
        f"Channel 6 → "
        f"RMS={rms_6[index]:.4f}, "
        f"Peak={peak_6[index]:.4f}, "
        f"Kurtosis={kurtosis_6[index]:.4f}"
    )


# ---------------------------------------------------------
# Plot RMS
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(timestamps, rms_5, label="Channel 5")
plt.plot(timestamps, rms_6, label="Channel 6")

plt.xlabel("Time")
plt.ylabel("RMS")

plt.title(
    "Test 1 - Bearing 3 - RMS over time"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# Plot Peak
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(timestamps, peak_5, label="Channel 5")
plt.plot(timestamps, peak_6, label="Channel 6")

plt.xlabel("Time")
plt.ylabel("Peak amplitude")

plt.title(
    "Test 1 - Bearing 3 - Peak amplitude over time"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# Plot Kurtosis
# ---------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(timestamps, kurtosis_5, label="Channel 5")
plt.plot(timestamps, kurtosis_6, label="Channel 6")

plt.xlabel("Time")
plt.ylabel("Kurtosis")

plt.title(
    "Test 1 - Bearing 3 - Kurtosis over time"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()