# Spec

## Purpose

Decide whether a newly composed prompt version should be accepted or rolled
back, from the holdout metrics of the new version and the active version. The
comparator is a pure decision component: the cycle orchestrator loads versions,
runs evaluations, and computes metrics; the comparator only evaluates the
decision rule.

## Requirements

### Requirement: Decision evaluation scope
The component SHALL expose a single decision function over the new version's
holdout metrics and the active version's holdout metrics. The component SHALL
NOT load prompt artifacts, run evaluations, or compute metrics — those are the
cycle orchestrator's responsibilities. The decision SHALL be driven by holdout
metrics; dev metrics are logged by the orchestrator for visibility and SHALL
NOT participate in the decision.

#### Scenario: Pure decision function
- **WHEN** the orchestrator has holdout metrics for both versions
- **THEN** it passes the two metric dicts to the comparator's decision function

#### Scenario: Metrics on holdout drive the decision
- **WHEN** the decision is evaluated
- **THEN** only holdout metrics participate; dev metrics are logged for visibility

### Requirement: Decision rule
The component SHALL decide to accept the new version when its decision metric
on holdout is **strictly greater** than the active version's. When the decision
metric is **equal**, the component SHALL accept only when the tie-breaker
metric is strictly higher than the active version's; otherwise it SHALL decide
to roll back. When the decision metric is **lower**, the component SHALL decide
to roll back.

#### Scenario: Accept on improvement
- **WHEN** the new version's holdout decision metric is strictly higher than
  the active version's
- **THEN** the new version is accepted

#### Scenario: Accept on tie with tie-breaker improvement
- **WHEN** the new version's holdout decision metric equals the active
  version's and the tie-breaker metric is strictly higher
- **THEN** the new version is accepted

#### Scenario: Rollback on tie without tie-breaker improvement
- **WHEN** the new version's holdout decision metric equals the active
  version's and the tie-breaker metric is equal or lower
- **THEN** the new version is rolled back

#### Scenario: Rollback on degradation
- **WHEN** the new version's holdout decision metric is lower than the active
  version's
- **THEN** the new version is rolled back

### Requirement: Rollback semantics in the cycle
On rollback the cycle orchestrator SHALL stop the current cycle's greedy
accumulation. The rejected version's rules SHALL NOT be reused. The next
candidate is selected fresh at the start of the next cycle.

#### Scenario: Accumulation stops on rollback
- **WHEN** a version is rolled back
- **THEN** the current cycle's accumulation stops and the best accepted version
  is finalized

#### Scenario: Next candidate next cycle
- **WHEN** a rollback occurred in one cycle
- **THEN** the next candidate is tried in the following cycle after fresh
  selection

### Requirement: Configurable parameters
The component SHALL support configuration via config for `decision_metric`
(default `macro_f1`) and `tie_breaker_metric` (default `minority_f1`). The
`tie_breaker_metric` SHALL participate only in tie decisions on the primary
metric. The `max_consecutive_rollbacks` parameter SHALL no longer be used.

#### Scenario: Custom decision metric
- **WHEN** config sets `decision_metric=accuracy`
- **THEN** the decision is based on accuracy on holdout instead of macro-F1

#### Scenario: Custom tie-breaker metric
- **WHEN** config sets `tie_breaker_metric=accuracy`
- **THEN** ties on the decision metric are resolved by accuracy on holdout

#### Scenario: Unused rollback limit
- **WHEN** config sets `max_consecutive_rollbacks`
- **THEN** the value is ignored; the cycle orchestrator's `no_improvement`
  stop condition governs stopping instead

### Requirement: Logging of decision
The cycle orchestrator SHALL log each decision with the new version, the
active version, both versions' holdout metrics, and the final decision.

#### Scenario: Decision logged
- **WHEN** a decision is made
- **THEN** the cycle log contains `new_version`, `active_version`, the holdout
  metrics of both, and `decision`
