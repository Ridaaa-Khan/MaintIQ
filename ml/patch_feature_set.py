from pathlib import Path
from datetime import datetime
import os
import re
import time
import math
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent
# Obsolete one-shot patcher. Production extraction is extract_features.py.
IMS_ROOT = PROJECT_ROOT / "data" / "raw" / "IMS"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLING_RATE = 20_000
FILTER_ORDER = 4
ENVELOPE_LOW = 1000
ENVELOPE_HIGH = 3000

RPM = 2000
NUMBER_OF_ROLLING_ELEMENTS = 16
PITCH_DIAMETER = 71.5
ROLLING_ELEMENT_DIAMETER = 8.4
CONTACT_ANGLE_DEGREES = 15.17
CONTACT_ANGLE = math.radians(CONTACT_ANGLE_DEGREES)
SHAFT_FREQUENCY = RPM / 60

BPFO = (
    NUMBER_OF_ROLLING_ELEMENTS / 2
    * (1 - (ROLLING_ELEMENT_DIAMETER / PITCH_DIAMETER) * math.cos(CONTACT_ANGLE))
    * SHAFT_FREQUENCY
)

BSF = (
    PITCH_DIAMETER / (2 * ROLLING_ELEMENT_DIAMETER)
    * (1 - ((ROLLING_ELEMENT_DIAMETER / PITCH_DIAMETER) * math.cos(CONTACT_ANGLE)) ** 2)
    * SHAFT_FREQUENCY
)

BEARING_CHANNELS = {
    "1st_test": {1: [0, 1], 2: [2, 3], 3: [4, 5], 4: [6, 7]},
    "2nd_test": {1: [0], 2: [1], 3: [2], 4: [3]},
    "3rd_test": {1: [0], 2: [1], 3: [2], 4: [3]},
}

TIMESTAMP_PATTERN = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
SIGNAL_LENGTH = 20_480
FFT_FREQUENCIES = np.fft.rfftfreq(SIGNAL_LENGTH, d=1 / SAMPLING_RATE)
NYQUIST = SAMPLING_RATE / 2
FILTER_SOS = butter(
    FILTER_ORDER,
    [ENVELOPE_LOW / NYQUIST, ENVELOPE_HIGH / NYQUIST],
    btype="bandpass",
    output="sos"
)


def process_single_file(file_args):
    test_name, file_path_str = file_args
    file_path = Path(file_path_str)
    data = np.loadtxt(file_path)
    timestamp = datetime.strptime(file_path.name, "%Y.%m.%d.%H.%M.%S")

    rows = []
    bearing_mapping = BEARING_CHANNELS[test_name]

    for bearing_id, channel_indices in bearing_mapping.items():
        ch_bsf_2x_mag = []
        ch_bsf_2x_eng = []
        ch_bpfo_ratio = []

        for ch_idx in channel_indices:
            signal = data[:, ch_idx]
            signal = signal - np.mean(signal)

            filtered = sosfiltfilt(FILTER_SOS, signal)
            envelope = np.abs(hilbert(filtered))

            env_mean_sub = envelope - np.mean(envelope)
            fft_vals = np.fft.rfft(env_mean_sub)
            mag = np.abs(fft_vals) / SIGNAL_LENGTH
            mag[1:-1] *= 2

            # 2*BSF
            mask_2bsf = (FFT_FREQUENCIES >= 2 * BSF - 5) & (FFT_FREQUENCIES <= 2 * BSF + 5)
            if np.any(mask_2bsf):
                vals = mag[mask_2bsf]
                ch_bsf_2x_mag.append(np.max(vals))
                ch_bsf_2x_eng.append(np.sum(vals ** 2))
            else:
                ch_bsf_2x_mag.append(0.0)
                ch_bsf_2x_eng.append(0.0)

            # BPFO local ratio
            t_mask_bpfo = (FFT_FREQUENCIES >= BPFO - 5) & (FFT_FREQUENCIES <= BPFO + 5)
            n_mask_bpfo = (FFT_FREQUENCIES >= BPFO - 25) & (FFT_FREQUENCIES <= BPFO + 25) & ~t_mask_bpfo
            t_vals = mag[t_mask_bpfo]
            n_vals = mag[n_mask_bpfo]
            if len(t_vals) > 0 and len(n_vals) > 0:
                t_peak = np.max(t_vals)
                n_med = np.median(n_vals)
                ch_bpfo_ratio.append(t_peak / n_med if n_med > 0 else 0.0)
            else:
                ch_bpfo_ratio.append(0.0)

        rows.append({
            "test_id": test_name,
            "bearing_id": bearing_id,
            "timestamp": timestamp.isoformat(),
            "source_file": file_path.name,
            "bsf_2x_magnitude_mean": float(np.mean(ch_bsf_2x_mag)),
            "bsf_2x_magnitude_max": float(np.max(ch_bsf_2x_mag)),
            "bsf_2x_energy_mean": float(np.mean(ch_bsf_2x_eng)),
            "bsf_2x_energy_max": float(np.max(ch_bsf_2x_eng)),
            "bpfo_local_ratio_mean": float(np.mean(ch_bpfo_ratio)),
            "bpfo_local_ratio_max": float(np.max(ch_bpfo_ratio)),
        })

    return rows


def get_all_raw_files():
    all_files = []
    for test_name in ["1st_test", "2nd_test", "3rd_test"]:
        test_folder = IMS_ROOT / test_name
        files = sorted([p for p in test_folder.rglob("*") if p.is_file() and TIMESTAMP_PATTERN.match(p.name)])
        for f in files:
            all_files.append((test_name, str(f)))
    return all_files


def main():
    start_time = time.time()
    print("=" * 70)
    print("PATCHING FEATURE SET WITH 2*BSF & BPFO LOCAL RATIO")
    print("=" * 70)

    raw_files = get_all_raw_files()
    print(f"Total raw files: {len(raw_files):,}")

    print("Extracting missing features in parallel...")
    extracted_rows = []
    max_workers = min(os.cpu_count() or 4, 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, f_args) for f_args in raw_files]
        done_cnt = 0
        for fut in as_completed(futures):
            extracted_rows.extend(fut.result())
            done_cnt += 1
            if done_cnt % 1000 == 0 or done_cnt == len(raw_files):
                print(f"[{done_cnt:,}/{len(raw_files):,}] raw files extracted ({done_cnt/len(raw_files)*100:.1f}%)")

    patch_df = pd.DataFrame(extracted_rows)
    patch_df["timestamp"] = pd.to_datetime(patch_df["timestamp"])
    patch_df.sort_values(["test_id", "bearing_id", "timestamp"], inplace=True)

    print(f"\nExtracted patch dataframe shape: {patch_df.shape}")

    new_cols = [
        "bsf_2x_magnitude_mean", "bsf_2x_magnitude_max",
        "bsf_2x_energy_mean", "bsf_2x_energy_max",
        "bpfo_local_ratio_mean", "bpfo_local_ratio_max"
    ]

    target_files = [
        "features.csv",
        "features_refined.csv",
        "features_temporal.csv",
        "features_selected.csv",
        "features_model_ready.csv"
    ]

    for fname in target_files:
        fpath = PROCESSED_DIR / fname
        if not fpath.exists():
            continue
        print(f"\nPatching: {fname}")
        df = pd.read_csv(fpath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Remove old cols if present to avoid dupes
        existing_patch_cols = [c for c in new_cols if c in df.columns]
        if existing_patch_cols:
            df.drop(columns=existing_patch_cols, inplace=True)

        merged = pd.merge(
            df,
            patch_df[["test_id", "bearing_id", "timestamp"] + new_cols],
            on=["test_id", "bearing_id", "timestamp"],
            how="left"
        )

        merged.to_csv(fpath, index=False)
        print(f"Patched {fname}: shape={merged.shape}")

    print(f"\nPatching complete in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
