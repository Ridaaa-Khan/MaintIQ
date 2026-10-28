
from pathlib import Path
from datetime import datetime
import csv
import math
import re
import time

import numpy as np

from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import skew, kurtosis

from dataset_config import channel_map, load_dataset_spec


# ============================================================
# PATHS / CONFIGURATION
# ============================================================
#
# Dataset-specific constants (sampling rate, geometry, channel map)
# live in configs/datasets/nasa_ims.json. Waveform feature math stays
# in this file.

PROJECT_ROOT = Path(__file__).resolve().parent
_SPEC = load_dataset_spec()
_ACQ = _SPEC["acquisition"]
_BEARING = _SPEC["bearing"]
_ENVELOPE = _SPEC["envelope_band"]

# Use the exact on-disk directory name.  The IMS dataset ships as
# "IMS" (uppercase).  Lower-case aliases silently work on Windows but
# break on Linux / Docker where the filesystem is case-sensitive.
IMS_ROOT = PROJECT_ROOT / _SPEC["raw_root"]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "features.csv"

# Official IMS sampling rate per the dataset documentation (20 kHz).
# Some papers cite 20,480 Hz – that figure equals 20 kHz × 1.024 and
# is NOT the correct playback rate.  All characteristic-frequency
# calculations (BPFI, BPFO, BSF …) use this constant, so changing it
# will shift every fault-frequency window by ±2 %.
SAMPLING_RATE = int(_ACQ["sampling_rate_hz"])

FILTER_ORDER = int(_ENVELOPE["filter_order"])

# Selected through our resonance-band experiments
ENVELOPE_LOW = int(_ENVELOPE["low_hz"])
ENVELOPE_HIGH = int(_ENVELOPE["high_hz"])

# Print progress every N raw files
PROGRESS_INTERVAL = 25


# ============================================================
# BEARING GEOMETRY
# ============================================================

RPM = int(_BEARING["rpm"])

NUMBER_OF_ROLLING_ELEMENTS = int(_BEARING["number_of_rolling_elements"])

PITCH_DIAMETER = float(_BEARING["pitch_diameter_mm"])
ROLLING_ELEMENT_DIAMETER = float(_BEARING["rolling_element_diameter_mm"])
CONTACT_ANGLE_DEGREES = float(_BEARING["contact_angle_degrees"])

CONTACT_ANGLE = math.radians(
    CONTACT_ANGLE_DEGREES
)

SHAFT_FREQUENCY = RPM / 60

FTF = (
    0.5
    * (
        1
        - (
            ROLLING_ELEMENT_DIAMETER
            / PITCH_DIAMETER
        )
        * math.cos(CONTACT_ANGLE)
    )
    * SHAFT_FREQUENCY
)

BPFO = (
    NUMBER_OF_ROLLING_ELEMENTS
    / 2
    * (
        1
        - (
            ROLLING_ELEMENT_DIAMETER
            / PITCH_DIAMETER
        )
        * math.cos(CONTACT_ANGLE)
    )
    * SHAFT_FREQUENCY
)

BPFI = (
    NUMBER_OF_ROLLING_ELEMENTS
    / 2
    * (
        1
        + (
            ROLLING_ELEMENT_DIAMETER
            / PITCH_DIAMETER
        )
        * math.cos(CONTACT_ANGLE)
    )
    * SHAFT_FREQUENCY
)

BSF = (
    PITCH_DIAMETER
    / (2 * ROLLING_ELEMENT_DIAMETER)
    * (
        1
        - (
            (
                ROLLING_ELEMENT_DIAMETER
                / PITCH_DIAMETER
            )
            * math.cos(CONTACT_ANGLE)
        ) ** 2
    )
    * SHAFT_FREQUENCY
)


# ============================================================
# DATASET STRUCTURE
# ============================================================

BEARING_CHANNELS = channel_map(_SPEC)


TIMESTAMP_PATTERN = re.compile(_ACQ["timestamp_pattern"])


# ============================================================
# PRECOMPUTED FFT FREQUENCY AXIS
# ============================================================

# Every IMS vibration snapshot contains exactly 20,480 samples
# (1.024 s at 20 kHz).  The FFT frequency axis is therefore fixed and
# can be pre-computed once.  Any file that deviates from this length
# will produce a misaligned FFT magnitude vector – process_file()
# raises an AssertionError rather than silently producing bad features.
SIGNAL_LENGTH = int(_ACQ["signal_length"])

FFT_FREQUENCIES = np.fft.rfftfreq(
    SIGNAL_LENGTH,
    d=1 / SAMPLING_RATE
)


# ============================================================
# PRECOMPUTED FILTER
# ============================================================

NYQUIST = SAMPLING_RATE / 2

LOW_NORMALIZED = (
    ENVELOPE_LOW / NYQUIST
)

HIGH_NORMALIZED = (
    ENVELOPE_HIGH / NYQUIST
)

FILTER_SOS = butter(
    FILTER_ORDER,
    [LOW_NORMALIZED, HIGH_NORMALIZED],
    btype="bandpass",
    output="sos"
)


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_data_files(test_folder):

    files = []

    for path in test_folder.rglob("*"):

        if (
            path.is_file()
            and TIMESTAMP_PATTERN.match(path.name)
        ):
            files.append(path)

    return sorted(files)


def parse_timestamp(filename):

    return datetime.strptime(
        filename,
        "%Y.%m.%d.%H.%M.%S"
    )


# ============================================================
# TIME-DOMAIN FEATURES
# ============================================================

def calculate_time_features(signal):

    mean_value = np.mean(signal)

    std_value = np.std(signal)

    variance_value = np.var(signal)

    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    minimum = np.min(signal)

    maximum = np.max(signal)

    peak = np.max(
        np.abs(signal)
    )

    peak_to_peak = (
        maximum - minimum
    )

    skewness = skew(
        signal,
        bias=False
    )

    kurt = kurtosis(
        signal,
        fisher=False,
        bias=False
    )

    mean_absolute = np.mean(
        np.abs(signal)
    )

    # Crest factor
    if rms_value > 0:
        crest_factor = (
            peak / rms_value
        )
    else:
        crest_factor = 0.0

    # Shape factor
    if mean_absolute > 0:
        shape_factor = (
            rms_value / mean_absolute
        )
    else:
        shape_factor = 0.0

    # Impulse factor
    if mean_absolute > 0:
        impulse_factor = (
            peak / mean_absolute
        )
    else:
        impulse_factor = 0.0

    # Clearance factor
    mean_sqrt = np.mean(
        np.sqrt(
            np.abs(signal)
        )
    )

    if mean_sqrt > 0:
        clearance_factor = (
            peak / (mean_sqrt ** 2)
        )
    else:
        clearance_factor = 0.0

    energy = np.sum(
        signal ** 2
    )

    return {

        "mean": mean_value,

        "std": std_value,

        "variance": variance_value,

        "rms": rms_value,

        "min": minimum,

        "max": maximum,

        "peak": peak,

        "peak_to_peak": peak_to_peak,

        "skewness": skewness,

        "kurtosis": kurt,

        "crest_factor": crest_factor,

        "shape_factor": shape_factor,

        "impulse_factor": impulse_factor,

        "clearance_factor": clearance_factor,

        "energy": energy,
    }


# ============================================================
# FFT
# ============================================================

def calculate_fft(signal):

    signal = (
        signal
        - np.mean(signal)
    )

    fft_values = np.fft.rfft(
        signal
    )

    magnitude = (
        np.abs(fft_values)
        / SIGNAL_LENGTH
    )

    magnitude[1:-1] *= 2

    return FFT_FREQUENCIES, magnitude


# ============================================================
# FREQUENCY-DOMAIN FEATURES
# ============================================================

def calculate_frequency_features(
    frequencies,
    magnitude
):

    # Ignore DC
    mask = frequencies > 0

    frequencies_no_dc = frequencies[mask]
    magnitude_no_dc = magnitude[mask]

    magnitude_sum = np.sum(
        magnitude_no_dc
    )

    spectral_energy = np.sum(
        magnitude_no_dc ** 2
    )

    # Spectral centroid
    if magnitude_sum > 0:

        spectral_centroid = (
            np.sum(
                frequencies_no_dc
                * magnitude_no_dc
            )
            / magnitude_sum
        )

    else:

        spectral_centroid = 0.0

    # Spectral spread
    if magnitude_sum > 0:

        spectral_spread = np.sqrt(
            np.sum(
                (
                    (
                        frequencies_no_dc
                        - spectral_centroid
                    ) ** 2
                    * magnitude_no_dc
                )
            )
            / magnitude_sum
        )

    else:

        spectral_spread = 0.0

    # Dominant frequency
    dominant_index = np.argmax(
        magnitude_no_dc
    )

    dominant_frequency = (
        frequencies_no_dc[
            dominant_index
        ]
    )

    dominant_magnitude = (
        magnitude_no_dc[
            dominant_index
        ]
    )

    # Spectral entropy
    power = (
        magnitude_no_dc ** 2
    )

    power_sum = np.sum(power)

    if power_sum > 0:

        probability = (
            power / power_sum
        )

        probability = probability[
            probability > 0
        ]

        spectral_entropy = -np.sum(
            probability
            * np.log2(probability)
        )

    else:

        spectral_entropy = 0.0

    # Frequency band energy
    def band_energy(lower, upper):

        band_mask = (
            (frequencies_no_dc >= lower)
            &
            (frequencies_no_dc < upper)
        )

        return np.sum(
            magnitude_no_dc[band_mask] ** 2
        )

    return {

        "spectral_energy":
            spectral_energy,

        "spectral_centroid":
            spectral_centroid,

        "spectral_spread":
            spectral_spread,

        "dominant_frequency":
            dominant_frequency,

        "dominant_magnitude":
            dominant_magnitude,

        "spectral_entropy":
            spectral_entropy,

        "band_energy_0_500":
            band_energy(0, 500),

        "band_energy_500_1000":
            band_energy(500, 1000),

        "band_energy_1000_2000":
            band_energy(1000, 2000),

        "band_energy_2000_4000":
            band_energy(2000, 4000),

        "band_energy_4000_6000":
            band_energy(4000, 6000),

        "band_energy_6000_8000":
            band_energy(6000, 8000),

        "band_energy_8000_10000":
            band_energy(8000, 10000),
    }


# ============================================================
# ENVELOPE FEATURES
# ============================================================

def calculate_envelope_features(
    signal
):

    # Filter the signal
    filtered = sosfiltfilt(
        FILTER_SOS,
        signal
    )

    # Hilbert envelope
    envelope = np.abs(
        hilbert(filtered)
    )

    # --------------------------------------------------------
    # Envelope time-domain features
    # --------------------------------------------------------

    envelope_mean = np.mean(
        envelope
    )

    envelope_std = np.std(
        envelope
    )

    envelope_rms = np.sqrt(
        np.mean(
            envelope ** 2
        )
    )

    envelope_peak = np.max(
        envelope
    )

    envelope_kurtosis = kurtosis(
        envelope,
        fisher=False,
        bias=False
    )

    if envelope_rms > 0:

        envelope_crest_factor = (
            envelope_peak
            / envelope_rms
        )

    else:

        envelope_crest_factor = 0.0

    # --------------------------------------------------------
    # Envelope FFT
    # --------------------------------------------------------

    frequencies, magnitude = (
        calculate_fft(envelope)
    )

    # --------------------------------------------------------
    # Frequency extraction helper
    # --------------------------------------------------------

    def frequency_features(
        target_frequency,
        tolerance=5
    ):

        mask = (
            (frequencies >= target_frequency - tolerance)
            &
            (frequencies <= target_frequency + tolerance)
        )

        if not np.any(mask):

            return {
                "magnitude": 0.0,
                "energy": 0.0,
            }

        values = magnitude[mask]

        return {

            "magnitude":
                np.max(values),

            "energy":
                np.sum(values ** 2),
        }

    # Characteristic frequencies

    ftf = frequency_features(
        FTF
    )

    bsf = frequency_features(
        BSF
    )

    bpfo = frequency_features(
        BPFO
    )

    bpfi = frequency_features(
        BPFI
    )

    # Harmonics

    bsf_2x = frequency_features(
        2 * BSF
    )

    bpfi_2x = frequency_features(
        2 * BPFI
    )

    bpfi_3x = frequency_features(
        3 * BPFI
    )

    bpfo_2x = frequency_features(
        2 * BPFO
    )

    bpfo_3x = frequency_features(
        3 * BPFO
    )

    # --------------------------------------------------------
    # BPFI / local background ratio
    # --------------------------------------------------------

    target_mask_bpfi = (
        (frequencies >= BPFI - 5)
        &
        (frequencies <= BPFI + 5)
    )

    neighborhood_mask_bpfi = (
        (frequencies >= BPFI - 25)
        &
        (frequencies <= BPFI + 25)
        &
        ~target_mask_bpfi
    )

    target_values_bpfi = magnitude[
        target_mask_bpfi
    ]

    neighborhood_values_bpfi = magnitude[
        neighborhood_mask_bpfi
    ]

    if (
        len(target_values_bpfi) > 0
        and len(neighborhood_values_bpfi) > 0
    ):

        target_peak_bpfi = np.max(
            target_values_bpfi
        )

        local_background_bpfi = np.median(
            neighborhood_values_bpfi
        )

        if local_background_bpfi > 0:

            bpfi_ratio = (
                target_peak_bpfi
                / local_background_bpfi
            )

        else:

            bpfi_ratio = 0.0

    else:

        bpfi_ratio = 0.0

    # --------------------------------------------------------
    # BPFO / local background ratio
    # --------------------------------------------------------

    target_mask_bpfo = (
        (frequencies >= BPFO - 5)
        &
        (frequencies <= BPFO + 5)
    )

    neighborhood_mask_bpfo = (
        (frequencies >= BPFO - 25)
        &
        (frequencies <= BPFO + 25)
        &
        ~target_mask_bpfo
    )

    target_values_bpfo = magnitude[
        target_mask_bpfo
    ]

    neighborhood_values_bpfo = magnitude[
        neighborhood_mask_bpfo
    ]

    if (
        len(target_values_bpfo) > 0
        and len(neighborhood_values_bpfo) > 0
    ):

        target_peak_bpfo = np.max(
            target_values_bpfo
        )

        local_background_bpfo = np.median(
            neighborhood_values_bpfo
        )

        if local_background_bpfo > 0:

            bpfo_ratio = (
                target_peak_bpfo
                / local_background_bpfo
            )

        else:

            bpfo_ratio = 0.0

    else:

        bpfo_ratio = 0.0

    return {

        "envelope_mean":
            envelope_mean,

        "envelope_std":
            envelope_std,

        "envelope_rms":
            envelope_rms,

        "envelope_peak":
            envelope_peak,

        "envelope_kurtosis":
            envelope_kurtosis,

        "envelope_crest_factor":
            envelope_crest_factor,

        "ftf_magnitude":
            ftf["magnitude"],

        "ftf_energy":
            ftf["energy"],

        "bsf_magnitude":
            bsf["magnitude"],

        "bsf_energy":
            bsf["energy"],

        "bsf_2x_magnitude":
            bsf_2x["magnitude"],

        "bsf_2x_energy":
            bsf_2x["energy"],

        "bpfo_magnitude":
            bpfo["magnitude"],

        "bpfo_energy":
            bpfo["energy"],

        "bpfi_magnitude":
            bpfi["magnitude"],

        "bpfi_energy":
            bpfi["energy"],

        "bpfi_2x_magnitude":
            bpfi_2x["magnitude"],

        "bpfi_2x_energy":
            bpfi_2x["energy"],

        "bpfi_3x_magnitude":
            bpfi_3x["magnitude"],

        "bpfi_3x_energy":
            bpfi_3x["energy"],

        "bpfo_2x_magnitude":
            bpfo_2x["magnitude"],

        "bpfo_2x_energy":
            bpfo_2x["energy"],

        "bpfo_3x_magnitude":
            bpfo_3x["magnitude"],

        "bpfo_3x_energy":
            bpfo_3x["energy"],

        "bpfi_local_ratio":
            bpfi_ratio,

        "bpfo_local_ratio":
            bpfo_ratio,
    }


# ============================================================
# EXTRACT FEATURES FROM ONE CHANNEL
# ============================================================

def extract_channel_features(
    signal
):

    # Work with float64
    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    # Remove DC
    signal = (
        signal
        - np.mean(signal)
    )

    features = {}

    # Time domain
    features.update(
        calculate_time_features(
            signal
        )
    )

    # FFT
    frequencies, magnitude = (
        calculate_fft(signal)
    )

    features.update(
        calculate_frequency_features(
            frequencies,
            magnitude
        )
    )

    # Envelope
    features.update(
        calculate_envelope_features(
            signal
        )
    )

    return features


# ============================================================
# COMBINE CHANNEL FEATURES
# ============================================================

def combine_channel_features(
    channel_features
):
    """
    Aggregate per-channel feature dicts into a single row dict.

    For each raw feature ``f``, two columns are produced:
      * ``f_mean`` – mean across channels.
      * ``f_max``  – maximum across channels.

    DUPLICATION NOTE (Test2 / Test3)
    ---------------------------------
    Test 2 and Test 3 each have one accelerometer per bearing, so
    ``f_mean`` and ``f_max`` will always be identical for those
    tests.  Test 1 has two accelerometers per bearing (channels 0+1,
    2+3, 4+5, 6+7), so ``f_mean`` and ``f_max`` carry distinct
    information there.

    Both columns are kept for *all* tests to maintain a single,
    consistent schema across the three datasets.  Downstream
    selection steps (``select_model_features.py``) may drop
    ``*_max`` or ``*_mean`` variants for single-channel bearings
    if a variance / redundancy analysis justifies it, but that
    decision is deferred there—not here.
    """

    combined = {}

    feature_names = (
        channel_features[0].keys()
    )

    for feature_name in feature_names:

        values = np.array([
            channel[feature_name]
            for channel in channel_features
        ])

        combined[
            feature_name + "_mean"
        ] = np.mean(values)

        combined[
            feature_name + "_max"
        ] = np.max(values)

    return combined


# ============================================================
# PROCESS ONE RAW FILE
# ============================================================

def process_file(
    test_name,
    file_path
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Load the raw file ONLY ONCE.
    # --------------------------------------------------------

    data = np.loadtxt(
        file_path
    )

    # --------------------------------------------------------
    # INTEGRITY GUARDS
    #
    # Wrong-length or wrong-channel files must fail loudly.
    # A silent mismatch would mis-align FFT bins and corrupt
    # every frequency feature derived from that file.
    # --------------------------------------------------------

    expected_channels = (
        sum(
            len(ch)
            for ch in BEARING_CHANNELS[test_name].values()
        )
    )

    actual_rows, actual_cols = data.shape

    if actual_rows != SIGNAL_LENGTH:
        raise AssertionError(
            f"Signal length mismatch in {file_path.name}: "
            f"expected {SIGNAL_LENGTH} rows, got {actual_rows}. "
            f"File may be corrupt or from a different dataset."
        )

    if actual_cols != expected_channels:
        raise AssertionError(
            f"Channel count mismatch in {file_path.name} "
            f"(test={test_name}): "
            f"expected {expected_channels} columns, got {actual_cols}."
        )

    rows = []

    timestamp = parse_timestamp(
        file_path.name
    )

    bearing_mapping = (
        BEARING_CHANNELS[
            test_name
        ]
    )

    # --------------------------------------------------------
    # Process each bearing from the already-loaded data
    # --------------------------------------------------------

    for bearing_id, channel_indices in (
        bearing_mapping.items()
    ):

        channel_features = []

        for channel_index in channel_indices:

            signal = data[
                :,
                channel_index
            ]

            features = (
                extract_channel_features(
                    signal
                )
            )

            channel_features.append(
                features
            )

        combined = (
            combine_channel_features(
                channel_features
            )
        )

        row = {

            "test_id":
                test_name,

            "bearing_id":
                bearing_id,

            "timestamp":
                timestamp.isoformat(),

            "source_file":
                file_path.name,

        }

        row.update(
            combined
        )

        rows.append(row)

    return rows


# ============================================================
# DISCOVER ALL FILES
# ============================================================

def get_all_files():

    all_files = []

    for test_name in BEARING_CHANNELS:

        test_folder = (
            IMS_ROOT
            / test_name
        )

        files = find_data_files(
            test_folder
        )

        print(
            f"{test_name}: "
            f"{len(files):,} files"
        )

        for file_path in files:

            all_files.append(
                (
                    test_name,
                    file_path
                )
            )

    return all_files


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("INDUSTRIAL PREDICTIVE MAINTENANCE")
    print("FEATURE EXTRACTION")
    print("=" * 70)
    print()

    print(
        f"Sampling rate: "
        f"{SAMPLING_RATE:,} Hz"
    )

    print(
        f"Signal length: "
        f"{SIGNAL_LENGTH:,} samples"
    )

    print(
        f"Envelope band: "
        f"{ENVELOPE_LOW:,}-"
        f"{ENVELOPE_HIGH:,} Hz"
    )

    print()

    print(
        f"Shaft: "
        f"{SHAFT_FREQUENCY:.3f} Hz"
    )

    print(
        f"FTF: "
        f"{FTF:.3f} Hz"
    )

    print(
        f"BSF: "
        f"{BSF:.3f} Hz"
    )

    print(
        f"BPFO: "
        f"{BPFO:.3f} Hz"
    )

    print(
        f"BPFI: "
        f"{BPFI:.3f} Hz"
    )

    print()

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    all_files = get_all_files()

    total_files = len(
        all_files
    )

    print()

    print(
        f"Total raw files: "
        f"{total_files:,}"
    )

    print(
        f"Expected feature rows: "
        f"{total_files * 4:,}"
    )

    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process first file to establish CSV columns
    # --------------------------------------------------------

    first_test, first_file = (
        all_files[0]
    )

    print(
        "Processing first file to "
        "initialize output..."
    )

    first_rows = process_file(
        first_test,
        first_file
    )

    fieldnames = list(
        first_rows[0].keys()
    )

    # --------------------------------------------------------
    # Open CSV once
    # --------------------------------------------------------

    start_time = time.time()

    processed_files = 0
    processed_rows = 0

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        # Write first file
        for row in first_rows:

            writer.writerow(row)

            processed_rows += 1

        processed_files += 1

        # ----------------------------------------------------
        # Process remaining files
        # ----------------------------------------------------

        for test_name, file_path in (
            all_files[1:]
        ):

            rows = process_file(
                test_name,
                file_path
            )

            for row in rows:

                writer.writerow(row)

                processed_rows += 1

            processed_files += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                processed_files
                % PROGRESS_INTERVAL == 0
                or processed_files == total_files
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                files_per_second = (
                    processed_files
                    / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    total_files
                    - processed_files
                )

                estimated_remaining = (
                    remaining
                    / files_per_second
                    if files_per_second > 0
                    else 0
                )

                print(
                    f"[{processed_files:,}/"
                    f"{total_files:,}] "
                    f"{files_per_second:.2f} "
                    f"files/sec | "
                    f"ETA: "
                    f"{estimated_remaining / 60:.1f} min"
                )

    # ========================================================
    # FINISHED
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Output file: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Files processed: "
        f"{processed_files:,}"
    )

    print(
        f"Rows generated: "
        f"{processed_rows:,}"
    )

    print(
        f"Features: "
        f"{len(fieldnames) - 4}"
    )

    print(
        f"Columns: "
        f"{len(fieldnames)}"
    )

    print(
        f"Time taken: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()


if __name__ == "__main__":
    main()