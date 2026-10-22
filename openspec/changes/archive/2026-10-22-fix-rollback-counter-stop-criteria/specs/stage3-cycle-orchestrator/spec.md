# Spec Delta

## ADDED Requirements

### Requirement: Consecutive rollback count in stop-criteria context

When the orchestrator builds `StopEvaluationContext` for the stop-criteria
evaluator, it SHALL compute `rollback_counter` as the number of trailing
`"rollback"` entries in `decision_history` (the consecutive rollback streak),
not a hardcoded zero. It SHALL pass `max_consecutive_rollbacks` from
`StopCriteriaConfig` (default 2), not a hardcoded zero.

#### Scenario: Zero consecutive rollbacks after accept
- **WHEN** the last decision in `decision_history` is `"accept"`
- **THEN** `rollback_counter` in `StopEvaluationContext` is 0

#### Scenario: Consecutive rollback streak counted
- **WHEN** `decision_history` is `["accept", "rollback", "rollback"]`
- **THEN** `rollback_counter` in `StopEvaluationContext` is 2

#### Scenario: Max consecutive rollbacks from config
- **WHEN** `StopCriteriaConfig` sets `max_consecutive_rollbacks=3`
- **THEN** `max_consecutive_rollbacks` in `StopEvaluationContext` is 3

#### Scenario: Default max consecutive rollbacks
- **WHEN** `StopCriteriaConfig` does not set `max_consecutive_rollbacks`
- **THEN** `max_consecutive_rollbacks` in `StopEvaluationContext` is 2

#### Scenario: Cycle continues below rollback limit
- **WHEN** the consecutive rollback count is 1 and
  `max_consecutive_rollbacks` is 2
- **THEN** the `rollback_streak` criterion does not trigger and the cycle
  continues

### Requirement: Rollback count in per-cycle report

The per-cycle report SHALL include the actual total rollback count
(`len(state.rollback_history)`), not a hardcoded zero.

#### Scenario: Report reflects rollback count
- **WHEN** a cycle completes with 3 entries in `state.rollback_history`
- **THEN** the report's `rollback_counter` field is 3
