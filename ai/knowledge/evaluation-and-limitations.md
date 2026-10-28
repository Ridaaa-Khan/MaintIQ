# Evaluation, splits, and known limits

## Trajectory split
- Train: 1st_test B1–B3, 2nd_test B1–B3, 3rd_test B1–B2.
- Validation: 1st_test B4, 2nd_test B4.
- Test: entire 3rd_test B3 and B4 (UI keys brg-003 and brg-004).
- Train bearings are not held-out performance.

## What the numbers mean
- Classifier threshold 0.315; F1 0.87 val / 0.61 test; Test 3 precision 1.00, recall 0.44.
- Isolation Forest contamination 0.01; baseline mean 0.431; val late-life lift +0.24.
- RUL last-24 h test MAE 2 h.

## Known failure cases / caveats
- Classifier can miss part of the last 24 hours (81 false negatives on Test 3).
- IF flags fire on windows that look unlike early life, including bearings without a documented failure.
- Uploaded CSV/ZIP/TXT files are parsed and stored; they do not receive invented Isolation Forest, classifier, or RUL scores until a real scoring run exists.
- FastAPI inference is not required for the assistant to read already-scored IMS artifacts.

## Safety
MaintIQ does not certify that a machine is safe to keep operating. It reports stored evidence and uncertainty. A qualified engineer still owns the operating decision.
