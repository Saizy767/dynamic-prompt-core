# Design

## Context

The metrics module (`application/services/metrics/metrics.py`) loads a
stage1-baseline-runner JSONL results artifact, computes classification metrics
via `sklearn.metrics`, and persists a JSON metrics artifact. The current
`_validate_row` checks only that required fields are present and non-null.
`load_results` has no duplicate-ID check. `_compute_accuracy` and `_compute_prf`
return `0.0` for empty `ok_rows`. The `compute_metrics` return dict reports
`accuracy` over `ok` records but does not make the denominator explicit —
consumers cannot distinguish "95% accuracy with 0 failures" from "95% accuracy
with 10% of examples silently excluded."

The existing `MetricsError(ValueError)` is the sole error type. The `REQUIRED_FIELDS`
tuple includes `confidence` which is no longer produced by the runner (see
`ResultRow` in `run_baseline/runner.py`), but that is a separate concern — this
change does not alter `REQUIRED_FIELDS`.

Callers of `compute_metrics` include the CLI (`_cmd_compute`) and
`compare_versions`. The `run_cycle` orchestrator consumes metrics indirectly
via `version-comparator`, which calls `compare_versions`.

## Goals / Non-Goals

**Goals:**
- Validate label values (`true_label`, `predicted_decision` for `ok` records)
  are integers in `{0, 1}`.
- Detect duplicate record IDs during loading.
- Verify `predicted_decision` is present for `ok` records.
- Return `null` (not `0.0`) for all metrics when the `ok` set is empty.
- Make denominator exclusion visible via `accuracy_over_total`, `denominator`,
  and `denominator_excludes_failures`.
- Add unit tests cross-checking precision/recall/F1/averaging against
  hand-computed and sklearn reference values.

**Non-Goals:**
- Changing `REQUIRED_FIELDS` or the artifact-loading schema (separate concern).
- Removing the sklearn dependency or reimplementing metric formulas in
  production code (sklearn is the reference; tests cross-check it).
- Changing the on-disk metrics artifact filename or directory layout.
- Modifying the `compare_versions` signature or the version-comparator
  decision logic (it reads `accuracy` and `f1.macro` which remain present
  when `ok` records exist).
- Adding multi-class support (the system is binary: labels `{0, 1}`).

## Decisions

### D1: Validate labels during `_validate_row`, duplicate IDs in `load_results`

**Choice:** Extend `_validate_row` to check that `true_label` is an int in
`{0, 1}` and that `predicted_decision` is an int in `{0, 1}` when
`classify_status == "ok"`. Add a `seen_ids: set` accumulator in `load_results`
that raises `MetricsError` on the first duplicate.

**Rationale:** `_validate_row` already runs per-record during loading, so label
validation adds no extra pass. Duplicate-ID detection requires cross-record
state, so it belongs in `load_results` where the accumulator is natural. Both
raise `MetricsError` (a `ValueError` subclass), preserving backward
compatibility with existing `except ValueError` handlers.

**Alternatives:** A separate `validate_results(rows)` pass after loading
(cleaner separation but adds a second iteration and a new public function that
callers must remember to call); validate in `compute_metrics` (too late —
invalid data would already be loaded).

### D2: `None` for empty-set metrics, `metrics_available` flag

**Choice:** When `ok_rows` is empty, `compute_metrics` sets `metrics_available:
false` and returns `None` for `accuracy`, `accuracy_over_total`, `precision`,
`recall`, `f1`, `confusion_matrix`, `prediction_distribution`,
`true_distribution`, `bias`, and `groups`. The `counts` dict is still populated.

**Rationale:** `0.0` is a valid metric value (model gets everything wrong) and
is indistinguishable from "no data." `None` / `null` in JSON is unambiguous.
The `metrics_available` flag lets consumers short-circuit without checking
every field for null.

**Alternatives:** Raise an error on empty `ok` sets (too strict — the artifact
should still be written with counts for diagnostics); return a special
`EmptyMetrics` sentinel object (over-engineered for a JSON output format).

### D3: Denominator transparency via three new fields

**Choice:** Add `accuracy_over_total` (correct / total), `denominator`
(`{ok_count, total_count}`), and `denominator_excludes_failures` (boolean) to
the metrics dict. `accuracy` remains correct / ok_count (unchanged).

**Rationale:** Three fields cover the full picture: `accuracy` is the
classification quality rate (unchanged for backward compatibility),
`accuracy_over_total` is the end-to-end success rate, and
`denominator_excludes_failures` is a single boolean that consumers can assert
on to detect silent exclusion. The `denominator` dict gives the raw counts for
auditability.

**Alternatives:** Replace `accuracy` with `accuracy_over_total` (breaking
change — `version-comparator` and stop-criteria read `accuracy`); add only a
boolean flag without `accuracy_over_total` (consumers still can't see the
end-to-end rate without recomputing).

### D4: Reuse `MetricsError` for all new validation errors

**Choice:** All new validation failures (invalid label, duplicate ID, missing
prediction) raise `MetricsError` with a descriptive message. No new error
subclasses.

**Rationale:** The existing `MetricsError(ValueError)` is already the single
error type for loading failures. Adding subclasses would be cleaner but is not
required by the spec — the error messages are descriptive enough to distinguish
failure modes, and consumers catch `MetricsError` or `ValueError`. If finer
error handling is needed later, subclasses can be added without breaking
callers.

**Alternatives:** Add `InvalidLabelError`, `DuplicateIdError`,
`MissingPredictionError` subclasses (cleaner for programmatic handling but
adds API surface for no current consumer).

### D5: Precision/recall/F1 cross-checks as tests, not production code

**Choice:** Add unit tests in
`tests/unit/metrics/test_metrics_validation.py` that compute metrics for
balanced, imbalanced, and single-class-predicted fixtures, then assert
per-class precision, recall, F1, macro-F1, and weighted-F1 against
hand-computed values. The tests also call
`sklearn.metrics.precision_recall_fscore_support` directly on the same
fixtures to confirm the production code matches sklearn.

**Rationale:** The production code already delegates to sklearn, so the
formulas are correct by delegation. The tests guard against future changes to
the sklearn call parameters (e.g., `labels`, `zero_division`) and document the
expected behavior for edge cases. Hand-computed values catch sklearn version
regressions.

**Alternatives:** Reimplement formulas in production code and cross-check
against sklearn (adds maintenance burden and a second source of truth);
trust sklearn entirely without tests (no guard against parameter changes).

## Risks / Trade-offs

- **[`None` metrics break consumers expecting `float`]** → The `metrics_available`
  flag is the primary guard. Audit `version-comparator` and stop-criteria
  consumers: they read `accuracy` and `f1.macro`, which are `None` only when
  no `ok` records exist — a degenerate case where the decision should be
  "reject" anyway. Add a guard in `compare_versions` to handle `None` metrics.
- **[New validation rejects previously-loadable artifacts]** → Artifacts with
  invalid labels or duplicate IDs were always incorrect; rejecting them early
  is the intended fix. The error message names the offending record for easy
  debugging.
- **[`accuracy_over_total` confusion]** → Document in the artifact and CLI
  summary that `accuracy` is over `ok` records and `accuracy_over_total` is
  over all records. The `denominator` field makes the distinction explicit.
- **[Empty artifact still writes a file]** → This is intentional for
  diagnostics (counts are populated). Consumers check `metrics_available`
  before using metric values.

## Migration Plan

1. **No data migration** — existing metrics JSON artifacts load without
   modification. New fields are additive.
2. **No API migration** — `compute_metrics` return dict gains fields; no
   existing fields are removed or renamed. `load_results` may raise
   `MetricsError` for previously-accepted invalid inputs — this is a bug fix,
   not a breaking change.
3. **Deploy** — drop-in replacement for `metrics.py`, plus a new test module.
4. **Rollback** — revert `metrics.py`. New fields in previously-written
   artifacts are ignored by the old code.

## Open Questions

- Should `compare_versions` skip the delta computation and return a
  `metrics_unavailable` result when either version has no `ok` records, or
  should it raise? For now, return `None` deltas with a
  `metrics_available` flag per version, consistent with `compute_metrics`.
- Should the CLI `_print_summary` display `accuracy_over_total` alongside
  `accuracy`, or only when `denominator_excludes_failures` is true? For now,
  always display both when metrics are available.
