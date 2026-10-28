from pathlib import Path
from datetime import datetime
import re
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

IMS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IMS"
)

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$"
)

TEST_FOLDERS = [
    IMS_ROOT / "1st_test",
    IMS_ROOT / "2nd_test",
    IMS_ROOT / "3rd_test",
]


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


for test_folder in TEST_FOLDERS:

    print("\n" + "=" * 70)
    print(f"TEST: {test_folder.name}")
    print("=" * 70)

    data_files = find_data_files(test_folder)

    print(f"Number of files: {len(data_files)}")

    if not data_files:
        continue

    # --------------------------------------------------
    # First file
    # --------------------------------------------------

    first_file = data_files[0]

    print(f"\nFirst file:")
    print(first_file.name)

    print(f"Timestamp:")
    print(parse_timestamp(first_file.name))

    # --------------------------------------------------
    # Last file
    # --------------------------------------------------

    last_file = data_files[-1]

    print(f"\nLast file:")
    print(last_file.name)

    print(f"Timestamp:")
    print(parse_timestamp(last_file.name))

    # --------------------------------------------------
    # Read first file
    # --------------------------------------------------

    data = np.loadtxt(first_file)

    print("\nSignal information:")
    print(f"Shape:       {data.shape}")
    print(f"Data type:   {data.dtype}")
    print(f"Dimensions:  {data.ndim}")

    # --------------------------------------------------
    # First 5 rows
    # --------------------------------------------------

    print("\nFirst 5 rows:")

    print(data[:5])

    # --------------------------------------------------
    # Channel statistics
    # --------------------------------------------------

    print("\nChannel statistics:")

    for channel_index in range(data.shape[1]):

        channel = data[:, channel_index]

        print(
            f"Channel {channel_index + 1}: "
            f"min={channel.min():.6f}, "
            f"max={channel.max():.6f}, "
            f"mean={channel.mean():.6f}, "
            f"std={channel.std():.6f}"
        )

    # --------------------------------------------------
    # Check timestamps
    # --------------------------------------------------

    first_timestamp = parse_timestamp(data_files[0].name)
    second_timestamp = parse_timestamp(data_files[1].name)

    interval = second_timestamp - first_timestamp

    print("\nTime between first two measurements:")
    print(interval)

    # --------------------------------------------------
    # File size
    # --------------------------------------------------

    print("\nFile size:")
    print(f"{first_file.stat().st_size:,} bytes")