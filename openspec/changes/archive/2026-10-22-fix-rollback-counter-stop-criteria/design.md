# Design

## Context

`run_cycle.py:270-273` builds `StopEvaluationContext` with
`rollback_counter=0` and `max_consecutive_rollbacks=0`. The `rollback_streak`
criterion (`criteria.py:114`) checks `counter >= max`, so `0 >= 0` always
triggers, stopping the cycle after one round. `StopCriteriaConfig` has no
`max_consecutive_rollbacks` field. `steps.py:647` also hardcodes
`rollback_counter=0` in per-cycle reports. `decision_history` (list of
`"accept"`/`"rollback"` strings) is already available at the call site and can
derive the consecutive streak.

## Goals / Non-Goals

**Goals:**
- Pass the real consecutive rollback count and a configurable max to
  `StopEvaluationContext`.
- Add `max_consecutive_rollbacks` to `StopCriteriaConfig` (default 2).
- Fix the per-cycle report's `rollback_counter`.
- Add an integration test proving the cycle runs multiple rounds with stop
  criteria enabled.

**Non-Goals:**
- Changing the `rollback_streak` criterion logic itself.
- Adding `total_teacher_tokens` tracking (also hardcoded to 0, but out of
  scope).
- Refactoring `StopEvaluationContext` fields.

## Decisions

### Compute consecutive rollbacks from `decision_history`

Count trailing `"rollback"` entries in `decision_history` at the call site in
`run_cycle.py`. This is O(streak length) and uses data already collected.

**Alternative**: track a dedicated `consecutive_rollback_count` field in
`CycleState`. Rejected — `decision_history` already encodes the information;
adding a redundant counter risks desynchronization.

### Add `max_consecutive_rollbacks` to `StopCriteriaConfig`

New field with default 2 (matching the conftest default and the deprecated
`VersionComparatorConfig.max_consecutive_rollbacks`). Loaded from
`[stop_criteria] max_consecutive_rollbacks` in TOML.

### Fix `steps.py:647` to `len(state.rollback_history)`

The report's `rollback_counter` should reflect total rollbacks, matching the
field name in `report.py:49`.

## Risks / Trade-offs

- [Behavior change] Cycles that previously stopped after one round will now
  continue. → This is a bug fix; the previous behavior was never intended and
  no production caller relied on it.
- [Config addition] New TOML key `max_consecutive_rollbacks`. → Optional with
  default 2; existing configs without the key are unaffected.
