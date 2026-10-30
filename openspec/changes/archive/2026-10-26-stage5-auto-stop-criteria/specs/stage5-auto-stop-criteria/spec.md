# Spec Delta

## Purpose

Determine when the optimization cycle should stop based on content-based
criteria — metric plateaus, degradation, candidate exhaustion, budget
exhaustion, rule stagnation, and rollback streaks — rather than only hard round
and rollback limits. The component evaluates cycle history at the end of each
round and signals the orchestrator to continue or stop.

## ADDED Requirements

### Requirement: Stop criteria port
The component SHALL define a `StopCriteria` port in
`application/ports/outbound/stop_criteria`. The port SHALL define the contract
for evaluating cycle state and returning a continue/stop decision. The port
SHALL use `typing.Protocol` with `@runtime_checkable`. The port SHALL NOT depend
on `infrastructure`.

#### Scenario: Port defined
- **WHEN** the orchestrator requests a stop evaluation
- **THEN** it depends on `application.ports.outbound.stop_criteria`, not on
  `infrastructure`

#### Scenario: Protocol used
- **WHEN** the port module is inspected
- **THEN** `typing.Protocol` is used and `@runtime_checkable` is applied

#### Scenario: Port independent of infrastructure
- **WHEN** the port module is imported
- **THEN** it does not import `infrastructure`

### Requirement: Use case location
The stop-evaluation use case SHALL reside in
`application/use_cases/evaluate_stop_criteria/`. The package SHALL expose a
single public function through `__init__.py` and `__all__`. The use case SHALL
receive dependencies through a typed dependency object and SHALL NOT create
concrete dependencies itself.

#### Scenario: Use case invoked
- **WHEN** the orchestrator requests an evaluation
- **THEN** `application.use_cases.evaluate_stop_criteria` is called

#### Scenario: Public API exposed
- **WHEN** the package `__all__` is read
- **THEN** it contains only the public function and result types

#### Scenario: No concrete dependencies
- **WHEN** the use case executes
- **THEN** it uses ports, not concrete infrastructure implementations

### Requirement: Input from cycle state
The use case SHALL accept the cycle state as input: per-round metric history
(`accuracy`, `macro_f1`, `minority_f1`), per-round decision history
(`accept` / `rollback`), the round counter, the rollback counter, the number of
candidates available in the queue, the total teacher-model token spend (when
teacher refinement is used), and the stop-criteria configuration. When the
metric history is empty, the use case SHALL return a `continue` decision without
stopping.

#### Scenario: Input provided
- **WHEN** the use case is called
- **THEN** it receives the metric history, decision history, counters, candidate
  count, token total, and configuration

#### Scenario: Missing input
- **WHEN** the metric history is empty
- **THEN** the use case returns a `continue` decision

### Requirement: Plateau detection
The component SHALL detect a plateau — a situation where the decision metric
does not improve by more than `min_improvement` for `plateau_window` consecutive
rounds. A plateau SHALL produce a `stop` decision with reason
`plateau_detected`.

#### Scenario: Plateau detected
- **WHEN** `macro_f1` does not improve by more than `min_improvement` for
  `plateau_window` consecutive rounds
- **THEN** a `stop` decision with reason `plateau_detected` is returned

#### Scenario: Improvement resets window
- **WHEN** after two rounds without improvement `macro_f1` improves by more than
  `min_improvement`
- **THEN** the plateau window is reset to zero

#### Scenario: Window not reached
- **WHEN** only two rounds without improvement occur with `plateau_window=3`
- **THEN** a `continue` decision is returned

### Requirement: Metric degradation detection
The component SHALL detect sustained metric degradation — the decision metric
decreases for `degradation_window` consecutive rounds. Degradation SHALL produce
a `stop` decision with reason `metric_degradation`. This criterion is distinct
from plateau: plateau is the absence of improvement; degradation is active
worsening.

#### Scenario: Degradation detected
- **WHEN** `macro_f1` decreases for `degradation_window` consecutive rounds
- **THEN** a `stop` decision with reason `metric_degradation` is returned

#### Scenario: Single drop not degradation
- **WHEN** `macro_f1` drops once then improves
- **THEN** a `continue` decision is returned

### Requirement: Candidate exhaustion detection
The component SHALL detect when the candidate queue is empty and no new
candidates were found in the current round. This SHALL produce a `stop` decision
with reason `no_candidates_available`.

#### Scenario: No candidates
- **WHEN** the queue is empty and no new candidates were found in the current
  round
- **THEN** a `stop` decision with reason `no_candidates_available` is returned

#### Scenario: Candidates still available
- **WHEN** the queue is empty but new candidates were found in the current round
- **THEN** a `continue` decision is returned

### Requirement: Budget exhaustion detection
The component SHALL track the total teacher-model token spend and the total
number of rounds. When `max_total_tokens` is exceeded or `max_total_rounds` is
exceeded, a `stop` decision with reason `budget_exhausted` SHALL be returned.

#### Scenario: Token budget exceeded
- **WHEN** the total teacher-model tokens exceed `max_total_tokens`
- **THEN** a `stop` decision with reason `budget_exhausted` is returned

#### Scenario: Round budget exceeded
- **WHEN** the number of rounds exceeds `max_total_rounds`
- **THEN** a `stop` decision with reason `budget_exhausted` is returned

#### Scenario: Budget not exceeded
- **WHEN** neither limit is reached
- **THEN** a `continue` decision is returned

### Requirement: Stagnation detection
The component SHALL detect rule stagnation — the set of rules in the active
prompt version does not change for `stagnation_window` consecutive rounds. This
SHALL produce a `stop` decision with reason `rule_stagnation`.

#### Scenario: Stagnation detected
- **WHEN** the rule set does not change for `stagnation_window` consecutive
  rounds
- **THEN** a `stop` decision with reason `rule_stagnation` is returned

#### Scenario: Rules changed
- **WHEN** a rule is added to or removed from the active version
- **THEN** the stagnation window is reset

### Requirement: Rollback streak detection
The component SHALL detect a streak of consecutive rollbacks. When the rollback
counter reaches `max_consecutive_rollbacks`, a `stop` decision with reason
`rollback_streak` SHALL be returned. This criterion duplicates the
`stage3-cycle-orchestrator` rollback check in a unified component.

#### Scenario: Rollback streak
- **WHEN** the rollback counter reaches `max_consecutive_rollbacks`
- **THEN** a `stop` decision with reason `rollback_streak` is returned

#### Scenario: Streak broken
- **WHEN** an accept follows a rollback
- **THEN** the rollback counter is reset

### Requirement: Combined decision
The component SHALL evaluate all enabled criteria independently and return a
`stop` decision if at least one criterion triggers. The check order SHALL be
deterministic: critical criteria first (`budget_exhausted`,
`no_candidates_available`), then qualitative (`plateau_detected`,
`metric_degradation`, `rule_stagnation`), then `rollback_streak`. The stop
reason SHALL be the first triggered criterion in this order. All triggered
criteria SHALL be recorded in the decision.

#### Scenario: Multiple criteria triggered
- **WHEN** `plateau_detected` and `rollback_streak` both trigger
- **THEN** a `stop` decision with reason `plateau_detected` is returned (first
  in order)

#### Scenario: No criteria triggered
- **WHEN** no criterion triggers
- **THEN** a `continue` decision is returned

#### Scenario: All triggered criteria recorded
- **WHEN** multiple criteria trigger
- **THEN** the decision records every triggered criterion, with the first in
  order as the reason

### Requirement: Stop decision artifact
The component SHALL persist the stop decision as an artifact via
`run_repository` when the decision is `stop`. The artifact SHALL contain: the
round number, the stop reason, the values of all tracked metrics for the last
`plateau_window` rounds, the counter values (rounds, rollbacks, tokens), and the
list of triggered criteria. The artifact SHALL be analyzable without
recomputation. When the decision is `continue`, no artifact SHALL be written.

#### Scenario: Artifact written on stop
- **WHEN** the decision is `stop`
- **THEN** an artifact `stop_decision_{run_id}_{timestamp}.json` is written with
  all required fields

#### Scenario: No artifact on continue
- **WHEN** the decision is `continue`
- **THEN** no artifact is written

### Requirement: Configurable parameters
The component SHALL support configuration via `StopCriteriaConfig`:
`plateau_window` (default 3), `min_improvement` (default 0.01),
`degradation_window` (default 3), `stagnation_window` (default 3),
`max_total_rounds` (default 10), `max_total_tokens` (default `None`), and
`enabled_criteria` (default: all criteria). The configuration SHALL be loaded by
the composition root and passed in. The use case SHALL NOT read `config.toml`
directly.

#### Scenario: Custom plateau window
- **WHEN** the configuration sets `plateau_window=5`
- **THEN** a plateau is detected after 5 rounds without improvement

#### Scenario: Criteria disabled
- **WHEN** the configuration sets `enabled_criteria=["plateau_detected"]`
- **THEN** only the plateau criterion is evaluated; all others are ignored

#### Scenario: No file reads
- **WHEN** the use case executes
- **THEN** it does not open `config.toml`

### Requirement: Logging of evaluation
The component SHALL log each evaluation: which criteria were checked, which
metric values were used, which criteria triggered, and the final decision. The
log SHALL be append-only. The component SHALL NOT call
`logging.basicConfig`.

#### Scenario: Evaluation logged
- **WHEN** an evaluation is performed
- **THEN** the log contains all checked criteria, the metric values, the
  triggered criteria, and the decision

#### Scenario: No logging configuration
- **WHEN** the module is imported
- **THEN** `logging.basicConfig` is not called

### Requirement: Testability
The component SHALL be testable with synthetic metric histories. Tests SHALL
NOT require a running cycle or inference server. Tests SHALL cover each
criterion in isolation, combined scenarios, the deterministic check order, and
window resets.

#### Scenario: Unit tests with synthetic histories
- **WHEN** the plateau criterion is tested
- **THEN** the metric history is provided directly without running a cycle

#### Scenario: No server required
- **WHEN** tests are run
- **THEN** no inference server is required
