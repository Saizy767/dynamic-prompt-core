# Proposal

## Why

The domain service `decide()` in `domain/services/decision.py` has a bug in the
equal-primary-metric branch: it returns `accept` even when the tie-breaker metric
does not improve, contradicting its own docstring. This can silently accept a
classifier version that neither improves the primary metric nor the secondary
metric. The error does not raise an exception, so it goes unnoticed and corrupts
experiment results.

## What Changes

- Fix `decide()` in `domain/services/decision.py` so that when the primary metric
  is equal, it returns `rollback` unless the tie-breaker metric is strictly
  higher — matching the existing docstring contract.
- Add unit tests for `decision.py:decide()` covering all three branches: primary
  improvement, primary tie with tie-breaker improvement, primary tie without
  tie-breaker improvement, and primary degradation.
- Note: `decision.py:decide()` is currently not imported anywhere in production
  code (`steps.py` uses `comparator.py:decide()` instead). This fix corrects the
  domain service so it is safe to use as the authoritative decision rule if
  callers switch to it.

## Capabilities

### New Capabilities
<!-- None — this change fixes an existing domain service, no new capability is introduced. -->

### Modified Capabilities
- `stage2-version-comparator`: Adds a requirement for the domain service
  `decision.py:decide()` that specifies tie-breaker-on-ties behavior, distinct
  from the application use case `comparator.py:decide()` which requires strict
  improvement only.

## Impact

- **Code**: `src/dynamic_prompt_core/domain/services/decision.py` — fix the
  equal-primary-metric branch (lines 64-68) to return `rollback` instead of
  `accept` when the tie-breaker does not improve.
- **Tests**: new test module for `decision.py` (none currently exist).
- **No production behavior change**: `decision.py:decide()` is not called in
  production; `comparator.py:decide()` governs the live cycle. The fix makes the
  domain service correct and usable.
- **Spec divergence**: the domain service's tie-breaker-on-ties rule intentionally
  differs from the application use case's strict-improvement-only rule already
  specified for `comparator.py`. Both are documented as separate requirements.
