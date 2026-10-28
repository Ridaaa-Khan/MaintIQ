# Failure classifier (XGBoost) in MaintIQ

The failure classifier is a supervised XGBoost model that estimates the probability that a bearing is in its last 24 hours of the recorded run.

## What it predicts
- Target: failure within 24 hours on the stored IMS trajectory (`predicted_failure_within_24h`).
- Output: a calibrated probability plus a class at the operating threshold **0.315**.
- This is a horizon flag on historical scores, not a shop-floor certificate that the machine will fail tomorrow.

## Why XGBoost
- The processed feature table is mixed (rolling RMS, band energy, envelope stats, fault-frequency magnitudes).
- Gradient-boosted trees handle that mix without inventing extra sensors.
- Calibration is applied so the stored probability can be compared to the 0.315 threshold.

## Held-out Test 3 metrics (do not paraphrase as fleet-wide proof)
- Classifier F1: 0.87 on validation / 0.61 on test.
- Test confusion at threshold 0.315:
  - Actual healthy: 12,355 predicted healthy, 0 predicted risk.
  - Actual last 24 h: 81 predicted healthy, 64 predicted risk.
- When it fires on Test 3 it is right (precision 1.00). It still misses part of the last-24-hour window (recall 0.44).

## How MaintIQ uses it
- A stored probability at or above 0.315 raises a Critical alert of type "Failure prediction".
- Missing classifier rows stay null. MaintIQ does not invent a probability for those windows.
- Some bearings (for example parts of T3-B4) do not have a probability on every window.

## Limitations
- Trained on IMS run-to-failure labels, not on a live plant with mixed machine types.
- Train bearings are not held-out performance. Quote validation / Test 3 numbers, not training scores.
- A probability of 0.52 is not "52% chance the bearing is broken right now."
