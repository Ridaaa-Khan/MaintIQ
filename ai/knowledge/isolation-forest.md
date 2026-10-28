# Isolation Forest in MaintIQ

Isolation Forest is MaintIQ's unsupervised anomaly detector. It looks for 12-hour windows whose processed vibration features are unlike the early-life baseline learned on training bearings.

## Why we use it
- The NASA IMS fleet has few labeled failures, so a fully supervised detector would overfit those four documented events.
- Isolation Forest does not need a failure label. It isolates unusual windows relative to early-life behaviour.
- In MaintIQ it is an investigation signal, not a diagnosis of a cracked race or a failed rolling element.

## How MaintIQ runs it
- Input: model-ready processed features from each operational snapshot (not raw 20,480-sample waveforms).
- Operating point: contamination 0.01 (1st percentile of the training score distribution).
- Train early-life mean anomaly score (chart reference line): 0.431. That dashed line is a reference, not an investigation cut.
- Validation late-life lift: +0.244 versus the early-life baseline (from selected_model.json / baseline summaries).
- Charts resample stored scores to 12-hour bins and drop rows marked `is_artifact`.

## How to read a flag
- `anomaly_flag = true` means that 12-hour bin looked unlike the early-life baseline.
- A higher score means more unlike the baseline, not a confirmed mechanical failure.
- MaintIQ health is derived from the latest scored window: Critical if the latest bin is IF-flagged, or the classifier predicted failure-within-24h, or calibrated probability is at or above 0.315.

## Limitations
- Flags can appear on bearings that have no documented IMS failure (for example T3-B4).
- End-of-run windows are often elevated because the run stops near degradation, not because every bearing is a labeled failure.
- Isolation Forest does not estimate remaining life and does not name a fault type.
