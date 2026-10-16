# Proposal

## Why

The `stage1-metrics` component computes classification quality metrics from
baseline-runner results, but its input validation and edge-case handling are
loose enough to produce silently misleading numbers. Failed examples are
excluded from the metric denominator without making that exclusion visible;
empty or all-failed input sets return `0.0` (indistinguishable from a model
that gets every example wrong); label values and record IDs are never
validated; and there is no consistency check between the number of input
examples, predictions, and labels. These gaps can cause the optimization cycle
to accept or reject a prompt version on incorrect metrics.

## What Changes

- **Validate label values**: `true_label` and `predicted_decision` for
  `classify_status=ok` records SHALL be in `{0, 1}`. Records with out-of-range
  or non-integer labels raise an error naming the offending record `id` and
  field.
- **Detect duplicate IDs**: the component SHALL raise an error when two records
  share the same `id`, naming the duplicated value.
- **Check input consistency**: the component SHALL verify that every record
  carries a `predicted_decision` when `classify_status=ok` and SHALL raise an
  error when an `ok` record has a missing or `None` prediction.
- **Handle empty and all-failed sets explicitly**: when no `ok` records exist,
  the component SHALL return `None` for accuracy, precision, recall, F1, and
  confusion-matrix counts (not `0.0`), and SHALL set a `metrics_available`
  flag to `false`. The artifact is still written with counts.
- **Make denominator exclusion visible**: the component SHALL report
  `accuracy_over_total` (correct / total records) alongside the existing
  `accuracy` (correct / ok records) and SHALL include a `denominator` field
  recording `ok_count` and `total_count`. When `failed > 0`, the artifact SHALL
  set `denominator_excludes_failures: true` so consumers cannot mistake
  `accuracy` for a rate over all inputs.
- **Verify precision / recall / F1 / averaging**: the component SHALL add
  unit tests that cross-check per-class precision, recall, F1, macro-F1, and
  weighted-F1 against hand-computed values and against sklearn reference
  outputs for balanced, imbalanced, and single-class-predicted scenarios.

## Capabilities

### New Capabilities
<!-- None — this change hardens an existing capability. -->

### Modified Capabilities
- `stage1-metrics`: Input validation (label range, duplicate IDs, prediction
  presence), empty/all-failed set semantics, and explicit denominator reporting
  are added as new requirements.

## Impact

- **Modified code**:
  `src/dynamic_prompt_core/application/services/metrics/metrics.py` —
  `_validate_row` gains label-range and prediction-presence checks;
  `load_results` gains duplicate-ID detection; `compute_metrics` returns `None`
  metrics for empty `ok` sets and adds `accuracy_over_total`, `denominator`,
  `denominator_excludes_failures`, and `metrics_available` fields.
- **Public API**: `compute_metrics` return dict gains new fields; no existing
  fields are removed. `load_results` may raise new `MetricsError` subtypes for
  invalid labels, duplicate IDs, and missing predictions — these are
  `ValueError` subclasses, so existing `except ValueError` handlers are
  unaffected.
- **Metrics artifact format**: New fields added to the JSON artifact. Existing
  fields unchanged. Old artifacts load without modification.
- **Dependencies**: no new third-party dependencies.
- **Tests**: new test module `tests/unit/metrics/test_metrics_validation.py`
  covering label validation, duplicate IDs, empty/all-failed sets, denominator
  reporting, and precision/recall/F1 cross-checks. Existing tests must continue
  to pass.
- **Upstream**: `run_cycle` decision logic and `version-comparator` consume
  metrics — they gain stronger input guarantees. Callers that read
  `accuracy` should be audited for denominator assumptions.
- **Downstream**: stop-criteria evaluation and the CLI summary print gain the
  `metrics_available` flag and `accuracy_over_total` for display.
