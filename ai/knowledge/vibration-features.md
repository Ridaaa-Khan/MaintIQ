# Vibration features in MaintIQ

Raw IMS snapshots are 20,480 samples at 20 kHz. The UI never plots those waveforms. It shows processed 12-hour means and the latest stored snapshot.

## Time-domain features (stored on series bins)
- RMS (g)
- Peak (g)
- Kurtosis
- Crest factor (available as a processed-feature option on Analytics)

## Frequency features
- FFT band energy: 0–500, 500–1000, 1000–2000, 2000–4000 Hz (latest window vs early-life mean).
- Envelope design band used in the feature pipeline: 1–3 kHz.
- Fault-frequency magnitudes at 2000 RPM: FTF, BSF, BPFO, BPFI.
- Band energy is stored on the latest window and the early-life mean, not as a per-bin series. Changing the Analytics date filter does not invent new band-energy history.

## Rolling / model-ready columns
The models consume a wider processed table (about 190 features), including rolling means and standard deviations of RMS and envelope statistics, BPFO/BPFI harmonics, and spectral shape. Those names appear in SHAP reports. They are derived from the same snapshots — they are not extra sensors.

## What 20,480 is
20,480 is samples per snapshot, not the number of machines. The uploaded fleet shown in MaintIQ is 12 IMS trajectories (3 tests × 4 bearings).
