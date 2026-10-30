# Spec Delta

## ADDED Requirements

### Requirement: Content-based stop evaluation
`run_cycle` SHALL call `evaluate_stop_criteria` at the end of each round, after
the per-round report is written and before deciding whether to start the next
round. The call SHALL pass the current cycle state: per-round metric history,
decision history, round counter, rollback counter, candidate-queue size, total
teacher-model token spend, and the stop-criteria configuration. When the
evaluation returns `stop`, the cycle SHALL terminate with the returned reason.
When the evaluation returns `continue`, the cycle SHALL proceed to the next
round. This evaluation is additional to the hard-limit checks (`max_rounds`,
`max_consecutive_rollbacks`, `candidate_queue_exhausted`) which SHALL still be
performed before starting a round.

#### Scenario: Orchestrator stops on criteria
- **WHEN** `evaluate_stop_criteria` returns `stop`
- **THEN** the cycle terminates with the returned reason

#### Scenario: Orchestrator continues
- **WHEN** `evaluate_stop_criteria` returns `continue`
- **THEN** the cycle proceeds to the next round

#### Scenario: Hard limits checked first
- **WHEN** a hard limit (`max_rounds`, `max_consecutive_rollbacks`,
  `candidate_queue_exhausted`) is reached before a round starts
- **THEN** the cycle stops without calling `evaluate_stop_criteria`

## MODIFIED Requirements

### Requirement: Final summary
`run_cycle` SHALL produce a final summary when the cycle stops. The summary
SHALL contain: total rounds executed, stop reason, final active version, final
dev and holdout metrics, history of accepted versions, history of rollbacks,
and the list of all changed decisions across rounds. The summary SHALL be saved
via `run_repository`. The stop reason SHALL be one of: `max_rounds_reached`,
`max_rollbacks_reached`, `candidate_queue_exhausted`, `unrecoverable_error`,
`plateau_detected`, `metric_degradation`, `no_candidates_available`,
`budget_exhausted`, `rule_stagnation`, `rollback_streak`.

#### Scenario: Summary written on stop
- **WHEN** the cycle stops for any reason
- **THEN** a final summary is written with all required fields

#### Scenario: Stop reasons enumerated
- **WHEN** the cycle stops
- **THEN** the stop reason is one of: `max_rounds_reached`,
  `max_rollbacks_reached`, `candidate_queue_exhausted`,
  `unrecoverable_error`, `plateau_detected`, `metric_degradation`,
  `no_candidates_available`, `budget_exhausted`, `rule_stagnation`,
  `rollback_streak`
