# Tasks

## 1. Add `max_consecutive_rollbacks` to `StopCriteriaConfig`

- [x] 1.1 In `evaluate_stop_criteria/config.py`, add field
      `max_consecutive_rollbacks: int = 2` to `StopCriteriaConfig` and a
      `DEFAULT_MAX_CONSECUTIVE_ROLLBACKS = 2` constant. In `from_toml`, read
      `s.get("max_consecutive_rollbacks", DEFAULT_MAX_CONSECUTIVE_ROLLBACKS)`.
      Verify by running `python3 -m pytest tests/unit/evaluate_stop_criteria/ -x`.
- [x] 1.2 Update `tests/unit/evaluate_stop_criteria/conftest.py` to default
      `max_consecutive_rollbacks` from `StopCriteriaConfig` instead of
      hardcoding 2, if the current default already matches. Verify existing
      tests still pass with `python3 -m pytest tests/unit/evaluate_stop_criteria/ -x`.

## 2. Fix `run_cycle.py` stop-criteria context

- [x] 2.1 In `run_cycle.py`, add a helper to count trailing `"rollback"` entries
      in `decision_history` (e.g. iterate reversed, count until a non-rollback
      entry is found). Use it to set `rollback_counter` in
      `StopEvaluationContext` instead of `0`. Verify by reading the updated
      code.
- [x] 2.2 In `run_cycle.py`, change `max_consecutive_rollbacks=0` to
      `max_consecutive_rollbacks=getattr(stop_config, "max_consecutive_rollbacks", 2)`
      in the `StopEvaluationContext` constructor. Verify by reading the updated
      code.

## 3. Fix `steps.py` report rollback counter

- [x] 3.1 In `steps.py` line 647, change `rollback_counter=0` to
      `rollback_counter=len(state.rollback_history)`. Verify by reading the
      updated code.

## 4. Add integration test

- [x] 4.1 In `tests/unit/run_cycle/`, add a test that builds a `CycleState`
      with `decision_history=["rollback"]` and verifies the
      `StopEvaluationContext` gets `rollback_counter=1` and
      `max_consecutive_rollbacks=2`, confirming the `rollback_streak` criterion
      does not trigger. Verify with
      `python3 -m pytest tests/unit/run_cycle/test_stop_criteria.py -x -v`.
- [x] 4.2 Add a test verifying that when `decision_history` has 2 trailing
      `"rollback"` entries and `max_consecutive_rollbacks=2`, the
      `rollback_streak` criterion does trigger. Verify with
      `python3 -m pytest tests/unit/run_cycle/test_stop_criteria.py -x -v`.

## 5. Full test suite and lint

- [x] 5.1 Run `python3 -m pytest tests/unit/evaluate_stop_criteria/
      tests/unit/run_cycle/ -v` and confirm all tests pass.
- [x] 5.2 Run `ruff check` and `mypy` on the changed files and confirm no new
      errors.
