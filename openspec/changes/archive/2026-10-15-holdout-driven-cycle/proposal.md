# Proposal

## Why

The optimization cycle currently decides accept/rollback using **dev-split
metrics**. Since candidates are also selected from dev-side thesis analysis,
this creates an overfitting risk: the cycle optimizes against the same data
it explores. Holdout metrics are computed but explicitly logged-only and
never influence the decision. Switching the decision to holdout gives a
valid generalization estimate and prevents the cycle from accepting rules
that only memorize dev-side patterns.

Additionally, the current linear versioning (`classify-v0 → classify-v1 → …`)
does not expose the cycle/cluster structure, making it hard to trace which
cycle produced which version and which cluster was added at each step.

## What Changes

- **BREAKING**: The accept/rollback decision SHALL be driven by **holdout**
  metrics, not dev metrics. Dev is used only for thesis collection (step 1
  of each cycle) and dev metrics are logged for visibility but do not
  influence the decision.
- **BREAKING**: The decision rule changes from "accept when `new >= best`"
  (with tie-breaker) to **strict improvement**: accept only when
  `new > best` on the holdout decision metric. Equal metrics → rollback.
- **BREAKING**: Versioning changes from linear `classify-vN` to a
  cycle/cluster scheme `vN_k` where `N` is the cycle number and `k` is the
  cluster index within that cycle. The finalized best of cycle `N` becomes
  `vN`. Baseline is `v0`.
- **BREAKING**: `max_rounds` is renamed to `max_cycles`. The concept of
  "round" is replaced by "cycle": one cycle produces one finalized version
  with the maximum-quality set of clusters.
- **BREAKING**: `max_consecutive_rollbacks` is removed. The stop condition
  is now `max_cycles_reached` or `no_improvement` (the first candidate of a
  cycle does not strictly improve over the base).
- The candidate accumulation loop runs each candidate **only on holdout**
  (no dev run per candidate). Dev runs happen once per cycle for the active
  version (thesis collection + dev metrics for logging).
- Per-cycle reports include both dev and holdout metrics, the list of
  `vN_k` attempts with their decisions, and the finalized version.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `stage3-cycle-orchestrator`: Cycle flow changes from dev-driven rounds to
  holdout-driven cycles. Decision metric source switches from dev to holdout.
  Stop conditions change from `max_rounds`/`max_rollbacks`/`candidate_queue_exhausted`
  to `max_cycles`/`no_improvement`. Versioning changes to `vN_k` scheme.
  Candidate accumulation loop runs only holdout per candidate.
- `stage2-version-comparator`: Decision rule changes from "accept on `>=`"
  with tie-breaker to strict improvement (`>` only). The comparison metric
  source shifts from dev to holdout. Tie-breaker is no longer used for the
  decision (kept for logging only).

## Impact

- `application/use_cases/run_cycle/steps.py` — major rewrite of
  `run_round` → `run_cycle_step`: holdout-driven accumulation, new
  versioning, no per-candidate dev runs.
- `application/use_cases/run_cycle/run_cycle.py` — stop logic:
  `max_cycles_reached`, `no_improvement`.
- `application/use_cases/run_cycle/state.py` — add `cycle_number`,
  `cluster_index`; rename `round_counter` → `cycle_counter`; remove
  `rollback_counter`.
- `application/use_cases/run_cycle/config.py` — `max_rounds` →
  `max_cycles`; remove `max_consecutive_rollbacks`.
- `application/use_cases/run_cycle/report.py` — cycle/cluster reporting,
  new stop reasons.
- `application/use_cases/compare_versions/comparator.py` — `decide()`:
  strict improvement only.
- `application/use_cases/compose_prompt/composer.py` — version naming
  `vN_k` / `vN`.
- `application/ports/inbound/run_cycle_input.py` — `max_rounds` →
  `max_cycles`.
- `interfaces/cli/main.py` — `--rounds` → `--cycles`.
- `config.toml` — `[cycle_orchestrator]` section updates.
- Tests — ~8 test files affected (counters, steps, state, report,
  stop_criteria, queue, error_handling, conftest).
