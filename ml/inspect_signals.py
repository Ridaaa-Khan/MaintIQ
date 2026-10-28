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


# ---------------------------------------------------------
# Choose Test 1
# ---------------------------------------------------------

test_folder = IMS_ROOT / "1st_test"

files = find_data_files(test_folder)

print(f"Total files: {len(files)}")


# ---------------------------------------------------------
# Select three points in the experiment
# ---------------------------------------------------------

selected_indices = [
    0,
    len(files) // 2,
    len(files) - 1
]


# Test 1:
# Bearing 3 = channels 5 and 6
CHANNEL_INDEX = 4       # Python index 4 = Channel 5


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(figsize=(14, 8))

for position, file_index in enumerate(selected_indices):

    file = files[file_index]

    data = np.loadtxt(file)

    signal = data[:, CHANNEL_INDEX]

    time = np.arange(len(signal)) / SAMPLING_RATE

    plt.subplot(3, 1, position + 1)

    plt.plot(time, signal)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    timestamp = parse_timestamp(file.name)

    plt.title(
        f"Test 1 - Bearing 3 - Channel 5\n"
        f"File {file_index + 1}/{len(files)} - {timestamp}"
    )

    plt.grid(True)


plt.tight_layout()

plt.show()