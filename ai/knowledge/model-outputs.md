# How to read MaintIQ model outputs

## Anomaly score (Isolation Forest)
- Higher = more unlike the early-life baseline.
- Chart reference: train early-life mean 0.431.
- A flag is an investigation cue.

## Failure probability (classifier)
- Calibrated P(last 24 hours of the recorded run).
- Threshold 0.315.
- Null means "not stored for that window," not zero.

## Predicted failure within 24 h
- Class label at the 0.315 operating point.
- Precision on Test 3 is high; recall is incomplete.

## Predicted RUL (hours)
- Experimental. Present only where the regressor stored a value.
- Quote last-24 h / last-72 h MAE, not full-run MAE.

## Health badge (derived, not a separate model)
From the latest scored window:
- Critical: latest IF flag, or predicted_failure_24h = 1, or probability ≥ 0.315.
- Warning: anomaly > 0.431 × 1.15, or RMS > 2× early-life mean.
- Healthy: none of the above.
- Unknown: uploaded machine with no scores yet.

Documented IMS damage is separate from health. Only four bearings have a documented failure label: T1-B3 inner race, T1-B4 rolling element, T2-B1 outer race, T3-B3 outer race. T3-B4 has no documented failure.
