# SHAP in MaintIQ

SHAP (SHapley Additive exPlanations) attributes a model output to processed features for a scored window.

## What we store
- Reports can show top features by mean |SHAP| from the Isolation Forest and failure-classifier explanation JSON.
- Feature names such as `rms_mean_rolling_mean_6h` or `bpfo_energy_mean_rolling_mean_6h` are processed columns, not extra plant sensors.
- A glossary in the report builder maps common names to plain language.

## How to talk about SHAP
- "This feature pushed the score up or down for that window" — not "this is the broken part."
- If a SHAP file or a feature glossary entry is missing, say the meaning is not stored.
- Do not invent SHAP values that are not in the report artifacts.
