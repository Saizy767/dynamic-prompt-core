# Tasks

## 1. Test scaffolding and characterization tests

- [x] 1.1 Create `tests/unit/metrics/__init__.py` and
  `tests/unit/metrics/test_metrics_validation.py` with a helper that builds
  in-memory result rows (dicts with all `REQUIRED_FIELDS`) and a
  `tmp_path`-based fixture that writes a JSONL artifact and calls
  `load_results` + `compute_metrics`. Verify the module is collected by
  `pytest tests/unit/metrics/test_metrics_validation.py --collect-only`.
- [x] 1.2 Add characterization tests for the current happy path: build 10 `ok`
  records with known labels and predictions, compute metrics, and verify
  `accuracy`, `precision`, `recall`, `f1` are floats and `counts.ok == 10`.
  Verify these pass against the current implementation
  (`pytest tests/unit/metrics/test_metrics_validation.py -k happy`).
- [x] 1.3 Add failing tests for the desired new behavior: invalid `true_label=2`
  raises (UT-01), duplicate `id` raises (UT-02), `ok` record with
  `predicted_decision=None` raises (UT-03), all-failed set returns `None`
  metrics (UT-04), `accuracy_over_total` is present (UT-05). Verify these
  tests fail against the current implementation, confirming the defects exist.

## 2. Label value validation

- [x] 2.1 Extend `_validate_row` in `metrics.py` to check that `true_label` is
  an `int` in `{0, 1}`. Raise `MetricsError` naming the record `id`, the field
  `true_label`, and the invalid value when the check fails. Verify UT-01 passes
  (`pytest tests/unit/metrics/test_metrics_validation.py -k invalid_true_label`).
- [x] 2.2 Extend `_validate_row` to check that `predicted_decision` is an `int`
  in `{0, 1}` when `classify_status == "ok"`. Raise `MetricsError` naming the
  record `id`, the field `predicted_decision`, and the invalid value. Do not
  validate `predicted_decision` for non-`ok` records. Verify UT-01b passes
  (`pytest tests/unit/metrics/test_metrics_validation.py -k invalid_prediction`).
- [x] 2.3 Add a test that a `failed` record with `predicted_decision=null` is
  accepted by `load_results`. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k failed_null_prediction`).

## 3. Duplicate ID detection

- [x] 3.1 Add a `seen_ids: set` accumulator to `load_results` that raises
  `MetricsError` naming the duplicated `id` value when a duplicate is
  encountered. Verify UT-02 passes
  (`pytest tests/unit/metrics/test_metrics_validation.py -k duplicate_id`).
- [x] 3.2 Add a test that records with all-distinct `id` values load
  successfully. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k unique_ids`).

## 4. Prediction presence for ok records

- [x] 4.1 Extend `_validate_row` to check that `predicted_decision` is not
  `None` when `classify_status == "ok"`. Raise `MetricsError` naming the record
  `id`. This check runs after the required-fields check and before the
  label-range check. Verify UT-03 passes
  (`pytest tests/unit/metrics/test_metrics_validation.py -k missing_prediction`).

## 5. Empty and all-failed set handling

- [x] 5.1 Modify `compute_metrics` to detect when `ok_rows` is empty. When
  empty, set `metrics_available: false` and return `None` for `accuracy`,
  `accuracy_over_total`, `precision`, `recall`, `f1`, `confusion_matrix`,
  `prediction_distribution`, `true_distribution`, `bias`, and `groups`.
  Populate `counts` as before. Verify UT-04 passes
  (`pytest tests/unit/metrics/test_metrics_validation.py -k empty_set`).
- [x] 5.2 Add a test for the all-failed scenario: 5 records all with
  `classify_status=failed`. Verify `metrics_available` is `false`, all metric
  fields are `None`, and `counts.total == 5`, `counts.failed == 5`. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k all_failed`).
- [x] 5.3 Add a test for the empty-artifact scenario: zero records. Verify
  `metrics_available` is `false`, all metric fields are `None`, and
  `counts.total == 0`. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k empty_artifact`).
- [x] 5.4 Add a test for the mixed scenario: at least one `ok` record. Verify
  `metrics_available` is `true` and all metric fields are populated with
  numeric values. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k mixed_ok_failed`).

## 6. Explicit metric denominator

- [x] 6.1 Add `accuracy_over_total` to the `compute_metrics` return dict:
  correct predictions / total records (including failed). When `ok_rows` is
  empty, set to `None`. Verify UT-05 passes
  (`pytest tests/unit/metrics/test_metrics_validation.py -k accuracy_over_total`).
- [x] 6.2 Add `denominator` field `{ok_count, total_count}` to the return dict.
  Add `denominator_excludes_failures` boolean: `true` when any record has
  `classify_status != "ok"`, `false` otherwise. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k denominator`).
- [x] 6.3 Add a test with 180 `ok` + 20 `failed` records, 150 correct.
  Verify `accuracy == 150/180`, `accuracy_over_total == 150/200`,
  `denominator == {ok_count: 180, total_count: 200}`,
  `denominator_excludes_failures == true`. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k denominator_with_failures`).
- [x] 6.4 Add a test with 200 `ok` records, 150 correct, 0 failed. Verify
  `accuracy == accuracy_over_total == 150/200` and
  `denominator_excludes_failures == false`. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k denominator_no_failures`).

## 7. Precision recall F1 cross-checks

- [x] 7.1 Add a test with a balanced fixture (50 TP, 50 TN, 0 FP, 0 FN).
  Assert `precision["0"] == 1.0`, `precision["1"] == 1.0`, `recall["0"] == 1.0`,
  `recall["1"] == 1.0`, `f1["0"] == 1.0`, `f1["1"] == 1.0`,
  `f1["macro"] == 1.0`, `f1["weighted"] == 1.0`. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k balanced`).
- [x] 7.2 Add a test with a single-class-predicted fixture (model predicts 1
  for all, half are class 0). Hand-compute: class 0 precision=0, recall=0,
  F1=0; class 1 precision=0.5, recall=1.0, F1=2/3. Assert these values.
  Also call `sklearn.metrics.precision_recall_fscore_support` directly on the
  same `y_true`/`y_pred` and assert the production values match. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k single_class`).
- [x] 7.3 Add a test with an imbalanced fixture (class 0: 180 examples,
  class 1: 20 examples, with known TP/TN/FP/FN). Hand-compute macro-F1
  (unweighted mean) and weighted-F1 (support-weighted mean). Assert both
  values. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k imbalanced`).
- [x] 7.4 Add a test verifying `zero_division=0` convention: a class with no
  predicted examples yields precision=0.0, recall=0.0, F1=0.0 for that class,
  not `NaN` or an exception. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k zero_division`).

## 8. compare_versions and CLI integration

- [x] 8.1 Add a guard in `compare_versions` for `None` metrics: when either
  version has `metrics_available == false`, return `None` for all deltas and
  include a `metrics_available` field per version in the result. Do not raise.
  Add a test with an all-failed v0 and a normal v1. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k compare_none_metrics`).
- [x] 8.2 Update `_print_summary` in `metrics.py` to display
  `accuracy_over_total` alongside `accuracy` when `metrics_available` is `true`,
  and to print "metrics unavailable" when `metrics_available` is `false`. Add
  a test that captures stdout and verifies the output. Verify
  (`pytest tests/unit/metrics/test_metrics_validation.py -k print_summary`).

## 9. Lint, typecheck, and full test suite

- [x] 9.1 Run `ruff check src/dynamic_prompt_core/application/services/metrics/metrics.py
  tests/unit/metrics/test_metrics_validation.py` and
  `ruff format --check src/dynamic_prompt_core/application/services/metrics/metrics.py
  tests/unit/metrics/test_metrics_validation.py`. Fix any issues. Verify both
  commands exit zero.
- [x] 9.2 Run `mypy src/dynamic_prompt_core/application/services/metrics/metrics.py`.
  Fix any type errors introduced by the `None` return values (the return type
  of `compute_metrics` is `dict[str, Any]` so `None` values are type-safe, but
  verify no new errors). Verify the command exits zero.
- [x] 9.3 Run the full metrics test suite:
  `pytest tests/unit/metrics/test_metrics_validation.py -v`. Verify all tests
  pass.
- [x] 9.4 Run `lint-imports` to verify architectural boundaries are not
  violated by the new test module. Verify no new contract breaches.
- [x] 9.5 Run the full project test suite:
  `pytest tests/ -x -m "not gguf"`. Verify all existing tests still pass
  alongside the new metrics validation tests.
