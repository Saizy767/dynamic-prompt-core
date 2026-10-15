# Spec Delta

## MODIFIED Requirements

### Requirement: Compare metrics
The component SHALL compare the new version's metrics against the active
version's metrics on holdout. The comparison SHALL include accuracy,
macro-F1, and minority-class F1. The component SHALL report the difference
for each metric. Holdout metrics SHALL drive the accept/rollback decision.
Dev metrics SHALL be logged for visibility but SHALL NOT participate in the
decision.

#### Scenario: Comparison reported
- **WHEN** the new version is evaluated
- **THEN** the difference in accuracy, macro-F1, and minority-class F1
  relative to the active version is reported

#### Scenario: Metrics on holdout
- **WHEN** metrics are computed on holdout
- **THEN** they drive the accept/rollback decision

### Requirement: Decision rule
The component SHALL decide to accept the new version only when its
decision metric on holdout is **strictly greater** than the active
version's. When the decision metric is equal or lower, the component SHALL
decide to roll back. The tie-breaker metric SHALL NOT be used in the
decision; it SHALL be logged for visibility only.

#### Scenario: Accept on improvement
- **WHEN** the new version's holdout decision metric is strictly higher than
  the active version's
- **THEN** the new version is accepted

#### Scenario: Accept on tie
- **WHEN** the new version's holdout decision metric equals the active
  version's
- **THEN** the new version is rolled back (strict improvement required; a
  tie is not sufficient)

#### Scenario: Rollback on degradation
- **WHEN** the new version's holdout decision metric is lower than the
  active version's
- **THEN** the new version is rolled back

### Requirement: Configurable parameters
The component SHALL support configuration via config for `decision_metric`
(default `macro_f1`) and `tie_breaker_metric` (default `minority_f1`). The
`tie_breaker_metric` SHALL be logged but SHALL NOT influence the
accept/rollback decision. The `max_consecutive_rollbacks` parameter SHALL
no longer be used.

#### Scenario: Custom rollback limit
- **WHEN** config sets `max_consecutive_rollbacks`
- **THEN** the value is ignored; the cycle orchestrator's `no_improvement`
  stop condition governs stopping instead

#### Scenario: Custom decision metric
- **WHEN** config sets `decision_metric=accuracy`
- **THEN** the decision is based on accuracy on holdout instead of macro-F1

## REMOVED Requirements

### Requirement: Rollback counter
**Reason**: The consecutive-rollback counter is no longer used. The cycle
orchestrator's `no_improvement` stop condition replaces it. A rollback is
the normal termination of a cycle's accumulation loop, not a failure state
to count.
**Migration**: Remove `max_consecutive_rollbacks` from
`VersionComparatorConfig` and `[version_comparator]` config. The
`update_rollback_count` and `check_stop` functions are removed.
