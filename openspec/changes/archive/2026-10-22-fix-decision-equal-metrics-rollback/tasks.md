# Tasks

## 1. Fix the domain service decision rule

- [x] 1.1 In `src/dynamic_prompt_core/domain/services/decision.py`, change the
      else-branch inside the `new_primary == active_primary` block (lines 64-68)
      to return `("rollback", ...)` with a reason string stating the primary
      metric tied and the tie-breaker did not improve. Verify by reading the
      updated function and confirming the branch returns `rollback`.
- [x] 1.2 Verify the docstring (lines 40-42) already matches the new behavior
      and requires no edit. Confirm by reading the docstring.

## 2. Add unit tests for `decision.py`

- [x] 2.1 Create `tests/unit/domain/services/test_decision.py` with test
      fixtures that build metrics dicts using the `f1` sub-dict layout
      (`{"f1": {"macro": ..., "minority": ...}}`). Add `__init__.py` files for
      any missing packages in the test path. Verify the test file is importable
      by running `python -m pytest tests/unit/domain/services/test_decision.py
      --collect-only`.
- [x] 2.2 Add a test for primary improvement: new primary > active primary →
      `accept`. Verify with `python -m pytest tests/unit/domain/services/
      test_decision.py::test_accept_on_primary_improvement -x`.
- [x] 2.3 Add a test for primary tie with tie-breaker improvement: equal
      primary, new tie > active tie → `accept`. Verify with `python -m pytest
      tests/unit/domain/services/test_decision.py::test_accept_on_tie_with_tie_breaker_improvement
      -x`.
- [x] 2.4 Add a test for primary tie without tie-breaker improvement: equal
      primary, new tie <= active tie → `rollback`. Verify with `python -m pytest
      tests/unit/domain/services/test_decision.py::test_rollback_on_tie_without_tie_breaker_improvement
      -x`.
- [x] 2.5 Add a test for primary degradation: new primary < active primary →
      `rollback`. Verify with `python -m pytest tests/unit/domain/services/
      test_decision.py::test_rollback_on_primary_degradation -x`.

## 3. Full test suite

- [x] 3.1 Run `python -m pytest tests/unit/domain/services/test_decision.py -v`
      and confirm all tests pass.
- [x] 3.2 Run the project linter and type checker (e.g. `ruff check` and
      `mypy` or project-equivalent) on the changed files and confirm no new
      errors.
