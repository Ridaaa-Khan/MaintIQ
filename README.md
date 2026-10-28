# Industrial Predictive Maintenance – IMS Bearing Dataset

End-to-end feature pipeline for NASA / University of Cincinnati IMS
run-to-failure bearing vibration data.

## Repo layout

| Path | Contents |
|------|----------|
| `ml/` | Feature pipeline, training, inference, configs, data, model artifacts |
| `fsd/web/` | MaintIQ landing site (Express + EJS) |
| `ai/` | Deferred AI Assistant package (not wired into the app) |
| Root | Shared project files (`.gitignore`, this README, `.venv`) |

Authoritative feature code: `ml/extract_features.py`.

Dataset constants (IMS specs vs our choices):
`ml/configs/datasets/nasa_ims.json`, loaded by `dataset_config.py` and
`pipeline_config.py`.

**Demo / interview brief (metrics, why each model, how to prove it):**
[`ml/ML_DESIGN.md`](ml/ML_DESIGN.md).

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r ml/requirements.txt
cd ml
```

---

## Pipeline order

Run from `ml/`. Orchestrate steps 2–6 (skips slow extract unless you pass
`--with-extract`):

```bash
cd ml
python run_pipeline.py
python smoke_test_ml.py
```

Or run these in order. Each step reads the previous output.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `extract_features.py` | `data/raw/IMS/*/` | `features.csv` |
| 2 | `refine_features.py` | `features.csv` | `features_refined.csv` |
| 3 | `add_temporal_features.py` | `features_refined.csv` | `features_temporal.csv` |
| 4 | `select_temporal_features.py` | `features_temporal.csv` | `features_selected.csv` |
| 5 | `select_model_features.py` | `features_selected.csv` | `features_model_ready.csv`, `split_metadata.json` |
| 6 | `create_targets.py` | `features_model_ready.csv` | `features_model_ready_with_targets.csv` and labeled `features_train/val/test.csv` |
| 7 | `train_isolation_forest.py` | labeled train/val/test | `models/isolation_forest/` (selected anomaly model + reports) |

Important:

- Step 5 writes the **unlabeled** model-ready table only. It does **not**
  overwrite `features_train.csv` / `features_val.csv` / `features_test.csv`.
- Step 6 owns the labeled train/val/test files. Always re-run
  `create_targets.py` after re-running feature selection.
- Step 7 fits Isolation Forest on train baseline rows only, compares
  `contamination="auto"` vs `0.01` on validation, and writes
  `isolation_forest_selected.joblib`.

Do not run exploratory scripts (`envelope_analysis.py`, `fft_analysis.py`,
`select_resonance_band.py`, `validate_resonance_band.py`, `inspect_*.py`,
`analyze_*.py`, `patch_feature_set.py`) to generate production features.

`analyze_final_features.py` must be run on `features_train.csv` only.

---

## Dataset facts (folder is the source of truth)

| Test | Files | Window | Documented damage |
|------|-------|--------|-------------------|
| `1st_test` | 2156 | 2003-10-22 → 2003-11-25 | B3 inner race, B4 rolling element |
| `2nd_test` | 984 | 2004-02-12 → 2004-02-19 | B1 outer race |
| `3rd_test` | **6324** | 2004-03-04 → **2004-04-18** | B3 outer race |

Some papers list Test 3 as 4448 files ending 2004-04-04. This folder has
6324 files ending 2004-04-18.

---

## Key constants (`configs/datasets/nasa_ims.json`)

| Constant | Value | Provenance |
|----------|-------|------------|
| `sampling_rate_hz` | 20,000 | IMS documentation (not 20,480 Hz) |
| `signal_length` | 20,480 samples | IMS files; mismatches raise |
| Envelope band | 1–3 kHz | Our resonance-band experiments |
| RPM / ZA-2115 geometry | 2000 RPM, 16 balls, … | IMS documentation → FTF/BPFO/BPFI/BSF |
| Paths | `data/raw/IMS` | Exact on-disk casing (Linux/Docker safe) |

`extract_features.py` reads that JSON, then computes characteristic
frequencies. A second IMS-like dataset needs a new JSON, not a rewrite
of kurtosis/FFT code.

---

## Trajectory split (no row-level leakage)

Split by `(test_id, bearing_id)`, never by shuffled rows.
Defined once in `pipeline_config.TRAJECTORY_SPLIT`.

- **Train:** 1st_test B1, B2, B3 · 2nd_test B1, B2, B3 · 3rd_test B1, B2
  (two failed bearings: T1-B3 inner race, T2-B1 outer race)
- **Val:** 1st_test B4 · 2nd_test B4
- **Test:** 3rd_test B3, B4 (entire Test 3 held out)

Each partition contains at least one failed bearing.
Low-variance drops are fit on train only.

---

## Target columns

Created by `create_targets.py` from unlabeled `features_model_ready.csv`.
Column names and constants live in `pipeline_config.py`.

- `is_artifact` — `rms_mean <= 0.01` (Test 2 / Test 3 shutdown files)
- `time_to_operational_endpoint_hours` — hours to the last **non-artifact**
  timestamp on the four documented failed bearings only; NaN on healthy
  runs and artifacts. This is an operational endpoint proxy (IMS does not
  provide an exact physical failure timestamp).
- `failure_within_24h`
  - Failed operational rows: `1` if time-to-endpoint ≤ 24 h, else `0`
  - Healthy operational rows: `0` only if ≥ 24 h of follow-up was observed;
    otherwise `NaN` (horizon-censored / not eligible)
  - Artifact rows: `NaN`
- `isolation_forest_train` — first 20% of operational rows on each
  trajectory (chronological early-life normal window); artifacts excluded

Do **not** treat last 10%/20%/50% of a run as a failure class.
Do **not** invent RUL for healthy bearings.
Do **not** treat `failure_within_24h = NaN` as class 0.

Failure endpoints exclude near-zero shutdown files.

---

## Temporal gaps

Test 1 has pauses up to ~148 h. `add_temporal_features.py` sets
`large_gap_flag = 1` and zeros delta/rate when
`time_since_previous_hours > 2`.

The first observation of each trajectory is dropped at the temporal step
(no previous timestamp for delta/rate). Targets use actual timestamps and
do not specially adjust RUL across gaps.

---

## Before model training

Use the labeled files from step 6:

- `data/processed/features_train.csv`
- `data/processed/features_val.csv`
- `data/processed/features_test.csv`

or `features_model_ready_with_targets.csv` with the same trajectory split.

When building the model matrix:

1. Build `X` with `pipeline_config.feature_columns(...)` — never include
   metadata or target/mask columns.
2. For classification, use only rows where `failure_within_24h` is not NaN.
3. For RUL regression, use only rows where
   `time_to_operational_endpoint_hours` is not NaN (failed trajectories).
4. Fit Isolation Forest on rows with `isolation_forest_train == True`
   from **train** trajectories.
5. Fit any scaler / imputer on **train only**, then apply to val/test.
6. Expect severe class imbalance (few positives); prefer PR-AUC / F1 over
   accuracy.
7. Evaluate by held-out **trajectory**, not shuffled pooled rows.

---

## Isolation Forest (anomaly detection)

Train:

```bash
python train_isolation_forest.py
```

Smoke-test that the selected detector works:

```bash
python smoke_test_isolation_forest.py
```

Score a feature table with the selected model:

```bash
python isolation_forest_infer.py data/processed/features_val.csv
```

Or in Python:

```python
from isolation_forest_infer import load_detector, score_dataframe

detector = load_detector()
scored = score_dataframe(df, detector)  # adds anomaly_score / anomaly_flag
```

Key artifacts in `models/isolation_forest/`:

| File | Role |
|------|------|
| `isolation_forest_selected.joblib` | Production pipeline (imputer + scaler + IF) |
| `selected_model.json` | Chosen contamination + verification checks |
| `failure_trajectory_analysis.csv` | Early vs late score lift per trajectory |
| `metrics.csv` | Proxy metrics vs `failure_within_24h` (not anomaly GT) |

Selection rule: on **validation only**, keep settings that show positive
late-life score lift on failed bearings, then pick the lower operational
flag rate. Test is reporting-only.

---

## Failure classifier (failure within 24 hours)

Train:

```bash
python train_failure_classifier.py
```

Smoke-test:

```bash
python smoke_test_failure_classifier.py
```

Score a feature table:

```bash
python failure_classifier_infer.py data/processed/features_val.csv
```

Or in Python:

```python
from failure_classifier_infer import load_classifier, score_dataframe

clf = load_classifier()
scored = score_dataframe(df, clf)  # failure_probability + prediction
```

Key artifacts in `models/failure_classifier/`:

| File | Role |
|------|------|
| `xgboost_failure_classifier.joblib` | Calibrated production model |
| `threshold.json` | Validation-selected decision threshold |
| `selected_model.json` | Best trial + verification checks |
| `metrics.csv` | Train/val/test classification metrics |
| `trajectory_metrics.csv` | Per-bearing precision/recall |
| `pr_curves.png` | Precision-recall curves |

Training notes:

- Uses only rows with non-NaN `failure_within_24h` (no censored/artifact labels).
- Hyperparameters chosen by validation PR-AUC; threshold by validation
  precision-floor F1. Test is reporting-only.
- Class imbalance is handled with tuned `scale_pos_weight` (not the raw
  500× ratio) plus sigmoid calibration and F-beta thresholding.

---

## RUL regressor (time to operational endpoint)

Train:

```bash
python train_rul_regressor.py
```

Audit vs train-mean baseline + trajectory plots:

```bash
python audit_rul_regressor.py
```

Smoke-test:

```bash
python smoke_test_rul_regressor.py
```

Score a feature table:

```bash
python rul_regressor_infer.py data/processed/features_val.csv
```

Key artifacts in `models/rul_regressor/`:

| File | Role |
|------|------|
| `xgboost_rul_regressor.joblib` | Production regressor (trained on log1p RUL) |
| `selected_model.json` | Best trial + verification |
| `metrics.csv` | Train/val/test MAE/RMSE/R2 vs mean baseline |
| `trajectory_metrics.csv` | Per-failed-bearing errors |
| `predicted_vs_actual.png` | Scatter diagnostic |

Notes:

- Uses only non-artifact rows with non-NaN
  `time_to_operational_endpoint_hours` (failed trajectories).
- Target is modeled as `log1p(hours)` and inverted to hours for metrics.
- Hyperparameters chosen by validation MAE; test is reporting-only.
- With only one failed train trajectory, cross-bearing generalization
  is limited — always read trajectory metrics, not just pooled MAE.

---

## Score all three models on one CSV

```bash
python score_all_models.py data/processed/features_val.csv
```

Requires the **190 trained feature columns** (schema-checked). Adds
`anomaly_score` / `anomaly_flag`, `failure_probability` /
`predicted_failure_within_24h`, and `predicted_rul_hours`.

RUL was trained on failed trajectories only — treat RUL on healthy
rows as exploratory.

Full ML health check:

```bash
python smoke_test_ml.py
```

---

## SHAP explanations (Feature 4)

Generate TreeSHAP explanations for the three trained models:

```bash
python explain_failure_shap.py
python explain_rul_shap.py
python explain_isolation_forest_shap.py
```

Smoke-test (checks artifacts; optional `--regenerate` rebuilds a small sample):

```bash
python smoke_test_shap.py
python smoke_test_shap.py --regenerate
```

| Script | Explains | Output directory |
|--------|----------|------------------|
| `explain_failure_shap.py` | Unwrapped `XGBClassifier` raw margin (not calibration) | `models/failure_classifier/shap_explanations/` |
| `explain_rul_shap.py` | `XGBRegressor` in **log1p(RUL)** space | `models/rul_regressor/shap_explanations/` |
| `explain_isolation_forest_shap.py` | `IsolationForest` expected path length on imputed+scaled features | `models/isolation_forest/shap_explanations/` |

Each run writes:

- `global_shap_importance.csv`
- `individual_shap_explanation.csv`
- `shap_summary.png`
- `individual_shap_waterfall.png`
- `shap_report.json` (includes additivity check)

Notes:

- Classifier SHAP uses the raw XGBoost estimator; calibrated probabilities
  are reported for context only.
- RUL SHAP is additive in log1p space, not in hours.
- Isolation Forest SHAP is on the final tree step after impute/scale and
  explains expected path length (invertible to `score_samples`), not
  `decision_function` directly.

---

## Next work

1. FastAPI inference service wrapping `score_all_models.score_all`
   (schema check + three joblibs + classifier threshold). See
   `ML_DESIGN.md` section 11 for the recommended v1 contract.
