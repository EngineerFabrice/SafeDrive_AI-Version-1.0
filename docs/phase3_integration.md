# Phase 3 integration foundation: model and dataset interfaces

This is the replaceable architecture for the impairment model. Today it runs
on a **synthetic dataset and a mock model**. Later the approved Keshtkaran
(WACV 2024) dataset and a really trained model plug in behind the **same
interfaces**, without changes to camera capture, driver selection, face
detection, Phase 2 features, the API or the dashboard.

> **Development only.** `MockImpairmentModel` and the mock dataset are
> synthetic. Their output says nothing about alcohol or impairment and must
> never be used for, or reported as, scientific or performance results.

## Flow

```
CURRENT  Camera -> Phase 1 -> Phase 2 (29 features, schema v1) -> ImpairmentModel[mock] -> API (flagged MOCK)
FUTURE   Keshtkaran videos -> audit -> same Phase 2 extractor -> trained model artifact
                                                               -> ImpairmentModel[real] -> same API
```

## Model interface: `engine/impairment/`

- **`ImpairmentInput`:** the 29-value vector (FEATURE_NAMES order, NaN = missing),
  `schema_version`, `timestamp`, `frame_id`, and optionally `feature_names`.
  Use `ImpairmentInput.from_features(FaceFeatures)` to build one from Phase 2 output.
- **`ImpairmentModel.predict(input) -> ImpairmentResult`:** the base class
  validates the input first. A schema version other than 1, a different feature
  order or a wrong length gives **`SCHEMA_MISMATCH`**. Infinite values or an
  all-NaN vector give **`INVALID_INPUT`**. A model exception gives
  **`MODEL_ERROR`**. A model that is not loaded gives **`MODEL_UNAVAILABLE`**.
  Only valid input reaches the subclass's `_predict(vector)`.
- **`ImpairmentResult`:** `status`, `prediction`, `probabilities`,
  `model_name`, `model_version`, `model_provider`, `schema_version`,
  `is_mock`, `development_only`, `error`, `note`, `timestamp` and `frame_id`.
  The target is named neutrally `impairment_class`; class meanings come from
  the model's and dataset's metadata.
- A result is a per-frame model output only. **There are no risk levels or
  alerts here** (Phase 5). The monitoring `status` is never changed by it.

## Dataset interface: `engine/datasets/`

- **`DatasetProvider.load() -> FeatureDataset`:** `X (n, 29)`, `y`,
  `subject_ids`, `session_ids`, `timestamps`, `frame_ids`, `conditions` and
  `bac` (NaN when not measured), plus `DatasetInfo`: type, the
  `not_for_research_claims` flag, schema, classes and class definitions,
  source and licence.
- `load()` rejects a dataset whose schema version or feature order differs
  from Phase 2.
- `subject_ids` is the grouping unit for any split; the same person must
  never be in both training and test.

## Mock implementations (development only)

| Part | What it is |
|---|---|
| `MockImpairmentModel` | A fixed, deterministic function of the vector (arbitrary weights, softmax). It returns `MOCK_LEVEL_0/1/2`, with `is_mock=true` and `development_only=true` and a note on every result. |
| `MockDatasetProvider` | Reads `data/mock/phase3/` (`features.csv` and `metadata.json`, with `DATASET_TYPE=MOCK` and `NOT_FOR_RESEARCH_CLAIMS=true`). It refuses to load if those flags are missing. 8 synthetic subjects × 3 sessions × 30 frames; first-frame motion and about 10% of gaze values are NaN, as in Phase 2. |
| `scripts/generate_mock_dataset.py` | Regenerates the mock data deterministically from a fixed seed. Random values only; no real people or faces. It adds an arbitrary per-level shift on 3 features so that training code can be smoke-tested. |

## Configuration

| Variable | Values | Default |
|---|---|---|
| `MODEL_PROVIDER` | `none`, `mock` | `none` (the live pipeline runs without predictions) |
| `DATASET_PROVIDER` | `mock` | `mock` |
| `MOCK_DATASET_DIR` | path to the mock data | `data/mock/phase3` |

- Only `create_impairment_model()` and `create_dataset_provider()` read
  these variables. The pipeline receives the model by injection, and nothing
  else imports a mock class.
- An unknown or failing model provider becomes an
  `UnavailableImpairmentModel` (results are `MODEL_UNAVAILABLE`). The camera
  and feature pipeline keep running.

## API exposure

- `GET /monitoring/status` has two fields:
  - **`impairment`:** the last result, or `null` when no model is configured or no features are available.
  - **`impairment_model`:** the configured model's info, or `null` when disabled.
- `GET /monitoring/model` returns `enabled`, the model info,
  `development_only`, and a MOCK notice when the mock model is active.

## Replacing the mock with the Keshtkaran dataset and a real model

Do this only after the dataset has been obtained under its transfer agreement
and audited.

1. **`KeshtkaranDatasetProvider(DatasetProvider)`** in `engine/datasets/`:
   - For each subject and trip, run the existing Phase 2 extractor
     frame by frame on the RGB face video, in order, so that motion features are valid.
   - Map the RFID driver ID to `subject_id`, the trip to `session_id`,
     the video time to `timestamps`/`frame_ids`, the breathalyser reading to
     `bac`, and the protocol state to `conditions` and `y`.
   - Set `dataset_type="RESEARCH"`, the real class definitions, source and
     licence terms. Register it as `keshtkaran` in `engine/datasets/registry.py`.
2. **Train offline** on `FeatureDataset`, with subject-grouped splits and
   preprocessing (imputation, scaling) fitted on the training folds only.
   Save the model, its preprocessing and metadata (schema version, feature
   names, dataset, target definitions, validation method, metrics).
3. **An artifact-backed `ImpairmentModel`:** loads that artifact, checks
   that its metadata schema equals `FEATURE_SCHEMA_VERSION`, and implements
   `_predict`. Register it (for example as `artifact`) in
   `engine/impairment/registry.py`, then set `MODEL_PROVIDER=artifact`.

Nothing outside `engine/datasets/` and `engine/impairment/` changes.

## Not in this foundation

- No real model or training code yet.
- No drowsiness features.
- No temporal aggregation (Phase 4).
- No risk scoring or alerts (Phase 5).
- The old `AlcoholDetectionDataset`, `DrunkingDetectionDataset` and `.h5`
  models are not used (see `docs/phase3_dataset_audit.md`).
