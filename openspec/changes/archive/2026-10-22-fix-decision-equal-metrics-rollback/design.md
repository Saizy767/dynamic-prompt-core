# Design

## Context

`domain/services/decision.py` exposes a `decide()` function with a docstring
that promises tie-breaker-on-ties behavior, but the equal-primary-metric branch
(lines 64-68) returns `accept` unconditionally instead of `rollback` when the
tie-breaker does not improve. The function is not imported in production code;
`comparator.py:decide()` (strict-improvement-only) is the live decision rule.
No tests exist for `decision.py`.

## Goals / Non-Goals

**Goals:**
- Make `decision.py:decide()` match its docstring: rollback on primary tie
  when the tie-breaker does not strictly improve.
- Add unit tests covering all four decision branches.

**Non-Goals:**
- Switching production callers from `comparator.py:decide()` to
  `decision.py:decide()`.
- Changing `comparator.py:decide()` or its strict-improvement-only spec.
- Refactoring `decision.py` signature or metric extraction logic.

## Decisions

### Fix the else-branch to return rollback

Change lines 64-68 in `decision.py` from returning `("accept", ...)` to
returning `("rollback", ...)` with a reason string that reports the tie-breaker
did not improve.

**Alternative considered**: align `decision.py` with the strict-improvement-only
rule from `comparator.py` and update the docstring. Rejected per user decision:
the docstring's tie-breaker-on-ties rule is the intended contract for the domain
service.

### Test placement: `tests/unit/domain/services/test_decision.py`

Follows the existing mirror convention (`tests/unit/` mirrors `src/` package
path). No test file exists yet for this module.

## Risks / Trade-offs

- [Latent divergence] `decision.py` and `comparator.py` now have intentionally
  different decision rules. → Both are documented in the spec as separate
  requirements; the docstrings clearly state each rule.
- [Unused code stays] `decision.py:decide()` remains uncalled in production.
  → The fix makes it correct and safe to adopt; removing it is out of scope.
