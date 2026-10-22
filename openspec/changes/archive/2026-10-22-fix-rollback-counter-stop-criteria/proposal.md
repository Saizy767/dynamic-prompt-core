# Proposal

## Why

`run_cycle.py` hardcodes `rollback_counter=0` and `max_consecutive_rollbacks=0`
when building `StopEvaluationContext`. The `rollback_streak` stop criterion
checks `rollback_counter >= max_consecutive_rollbacks`, so `0 >= 0` is always
true. Since `enabled_criteria` defaults to all-enabled, the cycle **always stops
after the first round** whenever stop-criteria deps are wired. Additionally,
`StopCriteriaConfig` has no `max_consecutive_rollbacks` field, so there is no
way to configure the threshold. `steps.py:647` also hardcodes
`rollback_counter=0` in per-cycle reports, so reports always show zero rollbacks.

## What Changes

- Add `max_consecutive_rollbacks` (default 2) to `StopCriteriaConfig`, loaded
  from the `[stop_criteria]` TOML section.
- Compute the **consecutive** rollback count from `decision_history` (trailing
  `"rollback"` entries) and pass it as `rollback_counter` in
  `StopEvaluationContext`.
- Pass `max_consecutive_rollbacks` from `stop_config` into
  `StopEvaluationContext`.
- Fix `steps.py:647` to pass the actual rollback count
  (`len(state.rollback_history)`) instead of `0`.
- Add integration tests that verify the cycle runs more than one round when
  stop criteria are enabled and rollbacks are below the limit.

## Capabilities

### New Capabilities
<!-- None — this change fixes existing orchestrator behavior. -->

### Modified Capabilities
- `stage3-cycle-orchestrator`: The orchestrator SHALL pass the real consecutive
  rollback count and a configurable `max_consecutive_rollbacks` to the stop
  criteria evaluator, instead of hardcoded zeros.

## Impact

- **Code**: `run_cycle.py` (lines 270-273), `steps.py` (line 647),
  `evaluate_stop_criteria/config.py` (new field).
- **Config**: new optional `max_consecutive_rollbacks` key in `[stop_criteria]`
  TOML section (default 2).
- **Tests**: new integration test in `tests/unit/run_cycle/` verifying
  multi-round execution with stop criteria enabled.
- **Behavior change**: cycles will no longer stop after one round when stop
  criteria are enabled. This is a bug fix — the previous behavior was never
  intended.
