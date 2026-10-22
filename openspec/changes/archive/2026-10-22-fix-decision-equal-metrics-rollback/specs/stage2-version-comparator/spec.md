# Spec Delta

## ADDED Requirements

### Requirement: Domain service decision rule with tie-breaker

The domain service `decide()` in `domain/services/decision.py` SHALL decide
whether to accept or roll back a new version using the following rule:

- When the new version's decision metric is **strictly greater** than the
  active version's, the domain service SHALL accept.
- When the decision metric is **equal**, the domain service SHALL accept only
  if the tie-breaker metric is strictly higher; otherwise it SHALL roll back.
- When the decision metric is **lower**, the domain service SHALL roll back.

This rule is distinct from the application use case's strict-improvement-only
rule. Both rules coexist: the application use case (`comparator.py`) requires
strict improvement and ignores the tie-breaker; the domain service
(`decision.py`) uses the tie-breaker to break ties on the primary metric.

#### Scenario: Accept on primary improvement
- **WHEN** the new version's decision metric is strictly greater than the
  active version's
- **THEN** the domain service returns `accept`

#### Scenario: Accept on primary tie with tie-breaker improvement
- **WHEN** the new version's decision metric equals the active version's
  and the tie-breaker metric is strictly higher
- **THEN** the domain service returns `accept`

#### Scenario: Rollback on primary tie without tie-breaker improvement
- **WHEN** the new version's decision metric equals the active version's
  and the tie-breaker metric is equal or lower
- **THEN** the domain service returns `rollback`

#### Scenario: Rollback on primary degradation
- **WHEN** the new version's decision metric is lower than the active
  version's
- **THEN** the domain service returns `rollback`
