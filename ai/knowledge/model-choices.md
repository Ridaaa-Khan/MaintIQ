# Why these models, and not others

## Isolation Forest
- Only four IMS bearings have documented failures, so a supervised anomaly detector would overfit them. Isolation Forest needs no failure labels.
- Versus One-Class SVM: scales better to ~190 features and thousands of windows, with less kernel/scale tuning.
- Versus autoencoders or LSTM: those need more healthy-variety data, more tuning and compute, and are harder to explain. With 12 trajectories they are a poor fit.
- Versus simple thresholds: thresholds on RMS miss multi-feature changes.

## XGBoost classifier
- Features are mixed tabular data (rolling RMS, band energy, envelope stats, fault-frequency magnitudes), where gradient-boosted trees work well.
- Versus Random Forest: boosting usually does better on imbalanced tabular data and supports probability calibration.
- Versus logistic regression: captures non-linear interactions between features.
- Versus LSTM/CNN: needs far more data and raw sequences; we use processed features and have few trajectories.
- Feature importance and SHAP explanations are straightforward with trees.

## XGBoost regressor (RUL)
- Same tabular reasoning. Experimental because only failed, scored trajectories exist. Not a service-life certificate.

## Why three models
They answer different questions: is this window unusual (Isolation Forest), is it in the last 24 h of the run (classifier), how many hours remain (regressor).
