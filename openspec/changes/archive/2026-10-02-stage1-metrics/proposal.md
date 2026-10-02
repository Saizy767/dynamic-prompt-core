# Proposal

## Why

Stage 1 produces a results artifact (`results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl`)
per run, but there is no component that turns those rows into a numerical point of
reference for comparing prompt versions. Without accuracy, per-class F1, a confusion
matrix, prediction-distribution bias, and a group-by-length breakdown, the
optimization loop has no principled way to accept, reject, or roll back a prompt
version — and no way to detect class imbalance or systematic prediction drift
between versions. This change adds the metrics stage that closes that gap.

## What Changes

- **Metrics computation**: a new component loads a results artifact from
  `stage1-baseline-runner`, validates that every record carries the required fields
  (`id`, `text`, `true_label`, `predicted_decision`, `confidence`,
  `classify_status`, `theses_norm`), and computes classification metrics only over
  records with `classify_status=ok`. Parse failures (`failed`, `repaired`,
  `not_attempted`) are counted and reported separately and MUST NOT be treated as
  classification errors.
- **Core metrics**: accuracy, per-class precision/recall/F1 (class 0 and class 1),
  macro-F1, weighted-F1, and a 2×2 confusion matrix (TP, TN, FP, FN).
- **Minority-class F1**: the minority class is determined from the dev split label
  distribution and its F1 is reported separately. Accuracy MUST NOT be the sole
  acceptance metric under class imbalance.
- **Prediction distribution & bias**: count and share of predicted 0/1, compared
  against the true-label distribution; a bias flag is set when the gap exceeds a
  configurable threshold.
- **Group breakdown by text length**: accuracy and F1 per group (short <10 words,
  medium 10–30, long >30) with configurable boundaries.
- **Metrics artifact**: persisted as
  `metrics_{run_id}_{prompt_version}_{split}_{timestamp}.json` containing every
  metric, the confusion matrix, distribution, group breakdown, and parse-failure
  counts. Loadable without recomputation.
- **Version comparison**: compare two metrics artifacts (same split, different
  `prompt_version`) and return per-metric deltas plus the list of `id`s whose
  `predicted_decision` changed, with direction (fixed / broke).
- **Dev vs holdout**: metrics are computed for dev and holdout separately. Holdout
  metrics are logged but MUST NOT drive accept/reject decisions on Stage 1.

## Capabilities

### New Capabilities

- `stage1-metrics`: Loads a stage1-baseline-runner results artifact, computes
  classification quality metrics (accuracy, per-class/macro/weighted F1, confusion
  matrix, minority-class F1, prediction distribution and bias, group-by-length
  breakdown, parse-failure counts), persists a json metrics artifact, and compares
  two prompt-version metrics on the same split.

### Modified Capabilities

<!-- None. The metrics component consumes the existing stage1-baseline-runner
     artifact and dataset-and-prompt split without changing their specs. -->

## Impact

- **New module(s)**: a metrics module (exact layout decided in design.md). Flat
  layout to match the project (`runner.py`, `dataset.py`, `normalize.py`).
- **`config.toml`**: optional new `[metrics]` section (bias threshold, group
  boundaries, output dir). No change to existing keys.
- **Dependencies**: `numpy` or `scikit-learn` for metric computation (SHALL be
  documented). Loaded lazily where practical so the module imports without them
  until computation runs.
- **Consumes**: the `results_*.jsonl` artifact from `stage1-baseline-runner` and
  the prepared dataset artifact from `dataset-and-prompt` (for minority-class
  determination and length-group boundaries).
- **API compatibility**: purely additive. No existing module signatures or
  behaviors change.
