# Spec Delta

## MODIFIED Requirements

### Requirement: Run a single round
`run_cycle` SHALL execute a single round as a fixed sequence of steps:

1. Run the active prompt version on dev via `stage1-baseline-runner`.
2. Compute metrics via `stage1-metrics`.
3. Update the thesis collection and clusters via `stage1-thesis-analyzer`.
4. Select rule candidates via `stage2-rule-candidate-selector`.
5. Greedily accumulate rules from the candidate queue:
   a. Compose a new prompt version from the next candidate via
      `stage3-prompt-composer`, appending the new rule to the rules already
      accepted in this round.
   b. Run the new version on dev via `stage1-baseline-runner`.
   c. Compare the new version against the current best version via
      `stage2-version-comparator`.
   d. When the new version does not degrade the decision metric, accept it:
      update the current best version, mark the candidate's cluster as
      `in_prompt`, and continue to the next candidate (step 5a).
   e. When the new version degrades the decision metric, stop accumulation:
      discard the candidate's rule, keep the current best version, and
      proceed to step 6.
   f. When the number of accumulated rules reaches `max_rules`, stop
      accumulation and proceed to step 6.
   g. When the candidate queue is exhausted, proceed to step 6.
6. Update the active version to the current best version.
7. Write the per-round report and state dump.

The round SHALL NOT end on the first acceptance. The cycle SHALL continue
trying candidates until degradation, the `max_rules` limit, or queue
exhaustion.

#### Scenario: Round completes
- **WHEN** a round is executed
- **THEN** steps 1–7 are performed in order

#### Scenario: Multiple rules accumulated
- **WHEN** candidates 1 and 2 both improve the metric and candidate 3 degrades it
- **THEN** the round ends with the active version containing rules from candidates 1 and 2, and candidate 3's rule is discarded

#### Scenario: First candidate degrades
- **WHEN** the first candidate degrades the metric
- **THEN** the round ends with the active version unchanged (no rules added) and the rollback counter is incremented

#### Scenario: Accumulation stops at max_rules
- **WHEN** the number of accepted rules reaches `max_rules`
- **THEN** accumulation stops and the remaining candidates are left in the queue

#### Scenario: Step failure
- **WHEN** a step fails with an unrecoverable error
- **THEN** the round is aborted and the cycle stops with an error naming the failing step

### Requirement: Candidate queue
`run_cycle` SHALL maintain a candidate queue populated by the
`stage2-rule-candidate-selector` at the start of each round. The queue SHALL be
consumed in order during greedy accumulation. When accumulation stops due to
degradation, remaining candidates SHALL stay in the queue for the next round.
When the queue is exhausted without any acceptance and the rollback counter
exceeds `max_consecutive_rollbacks`, the cycle SHALL stop with reason
`candidate_queue_exhausted`.

#### Scenario: Queue consumed in order
- **WHEN** the composer requests a candidate
- **THEN** the next candidate from the queue is returned

#### Scenario: Queue refilled each round
- **WHEN** a new round starts
- **THEN** the queue is refilled from the current round's selection

#### Scenario: Remaining candidates preserved on degradation
- **WHEN** accumulation stops due to degradation with candidates remaining
- **THEN** the remaining candidates stay in the queue for the next round

#### Scenario: Queue exhausted without acceptance
- **WHEN** the queue is exhausted and no candidate was accepted
- **THEN** the rollback counter is incremented

### Requirement: Rollback state
`run_cycle` SHALL track consecutive rollbacks. A round in which no candidate
was accepted SHALL increment the rollback counter. A round with at least one
acceptance SHALL reset the rollback counter to zero. When the rollback counter
reaches `max_consecutive_rollbacks`, the cycle SHALL stop with reason
`max_rollbacks_reached`.

#### Scenario: No acceptance increments rollback
- **WHEN** a round ends with no candidate accepted
- **THEN** the rollback counter is incremented

#### Scenario: Acceptance resets rollback
- **WHEN** a round ends with at least one candidate accepted
- **THEN** the rollback counter is reset to zero

#### Scenario: Cycle stops after max rollbacks
- **WHEN** the rollback counter reaches `max_consecutive_rollbacks`
- **THEN** the cycle stops with reason `max_rollbacks_reached`

## ADDED Requirements

### Requirement: Holdout without thesis extraction
`run_cycle` SHALL run holdout evaluation with thesis extraction disabled. The
holdout run SHALL only perform candidate-scoring classification for metrics
computation and SHALL NOT call `extract_theses_detailed`. Holdout metrics SHALL
be logged only and SHALL NOT influence the accept/rollback decision.

#### Scenario: Holdout skips thesis extraction
- **WHEN** the holdout evaluation runs
- **THEN** no thesis-extraction LLM calls are made and only classification scoring is performed

#### Scenario: Holdout metrics logged only
- **WHEN** holdout metrics are computed
- **THEN** they are included in the per-round report but are not used in the accept/rollback decision
