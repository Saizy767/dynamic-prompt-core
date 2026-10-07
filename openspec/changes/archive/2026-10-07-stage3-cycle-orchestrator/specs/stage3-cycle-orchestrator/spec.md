# Spec Delta

## ADDED Requirements

### Requirement: Use case location
`run_cycle` SHALL reside in `application/use_cases/run_cycle/`. The package
SHALL export a single public function and result types through `__init__.py`
and `__all__`. Internal implementation modules (`steps.py`, `state.py`,
`report.py`, `config.py`) SHALL NOT appear in `__all__`.

#### Scenario: Public function exposed
- **WHEN** `run_cycle` is imported
- **THEN** it is accessible via `application.use_cases.run_cycle`

#### Scenario: Internal modules hidden
- **WHEN** the package `__all__` is read
- **THEN** it contains only the public function and result types

### Requirement: Dependency object
`run_cycle` SHALL receive all dependencies through a typed dependency object
(`RunCycleDeps`). The dependency object SHALL contain outbound ports:
`llm_client`, `prompt_repository`, `run_repository`, `dataset_repository`,
`embedding_client`, `normalizer`, and optionally `teacher_llm_client`. The
dependency object SHALL NOT contain concrete infrastructure implementations.
The use case SHALL NOT create dependencies itself or use a service locator.

#### Scenario: Typed dependency object
- **WHEN** the use case is called
- **THEN** it receives a dependency object containing typed ports

#### Scenario: No concrete dependencies
- **WHEN** the use case executes
- **THEN** it does not import from `infrastructure`

#### Scenario: No service locator
- **WHEN** the use case receives dependencies
- **THEN** they are passed explicitly, not via a global registry

### Requirement: Integration with existing components
`run_cycle` SHALL call existing use cases and services through their public
APIs: `stage1-baseline-runner`, `stage1-metrics`, `stage1-thesis-analyzer`,
`stage2-rule-candidate-selector`, `stage2-version-comparator`,
`stage3-prompt-composer`, and optionally `stage4-teacher-refinement`. The
orchestrator SHALL NOT duplicate their logic.

#### Scenario: Components called via public API
- **WHEN** `run_cycle` invokes a component
- **THEN** it uses the component's public API

#### Scenario: No duplication
- **WHEN** `run_cycle` is read
- **THEN** it contains no metrics, clustering, or composition logic

### Requirement: Optional teacher refinement
`run_cycle` SHALL support an optional call to `stage4-teacher-refinement`
between the baseline run and thesis analysis when `use_teacher_refinement` is
enabled in the configuration. By default refinement SHALL be disabled. When
enabled, refined theses SHALL be used for analysis; when disabled, raw theses
SHALL be used.

#### Scenario: Refinement disabled
- **WHEN** the configuration does not enable teacher refinement
- **THEN** raw theses (`theses_raw_tiny`) are used for analysis

#### Scenario: Refinement enabled
- **WHEN** the configuration enables teacher refinement
- **THEN** refined theses are used for analysis

### Requirement: Composition root wiring
`interfaces/cli/main.py` SHALL assemble all dependencies for `run_cycle`:
concrete port implementations from `infrastructure`, including the LLM client,
prompt repository, run repository, dataset repository, embedding client,
normalizer, and teacher client (when enabled). The composition root SHALL NOT
contain business logic. No other module SHALL create these dependencies.

#### Scenario: Dependencies wired in CLI
- **WHEN** the CLI starts the cycle command
- **THEN** all dependencies are assembled in `interfaces/cli/main.py`

#### Scenario: Use case receives dependencies
- **WHEN** `run_cycle` is called
- **THEN** it receives a dependency object assembled in the composition root

### Requirement: Error handling
`run_cycle` SHALL continue when an individual example fails (parse error,
timeout, HTTP error), recording the failure in the result record. The cycle
SHALL NOT abort due to a single example error. An unrecoverable exception in
the use case itself SHALL stop the cycle with reason `unrecoverable_error`.

#### Scenario: Single example failure
- **WHEN** the model returns invalid JSON for an example
- **THEN** the record is saved with a failed status and the cycle continues

#### Scenario: Unrecoverable error
- **WHEN** the use case raises an unrecoverable exception
- **THEN** the cycle stops with reason `unrecoverable_error`

### Requirement: Testability
`run_cycle` SHALL be testable with mock implementations of the ports. Tests
SHALL NOT require a running inference server. Tests SHALL verify the step
sequence, counter behavior, rollback handling, report and summary formation,
and resumability from a dumped state.

#### Scenario: Unit test with mocks
- **WHEN** `run_cycle` is tested
- **THEN** ports are replaced by mock objects

#### Scenario: No server required
- **WHEN** tests are run
- **THEN** no inference server is required

## MODIFIED Requirements

### Requirement: Load configuration
`run_cycle` SHALL accept cycle configuration as a typed object (`CycleConfig`)
containing `max_rounds` (default 5), `max_consecutive_rollbacks` (default 2),
`decision_metric` (default `macro_f1`), `tie_breaker_metric` (default
`minority_f1`), `dev_split`, `holdout_split`, artifact paths, `run_id`,
`stop_on_first_error` (default false), `dump_state_after_each_round` (default
true), and `use_teacher_refinement` (default false). The configuration SHALL be
loaded by the composition root and passed in. The use case SHALL NOT read
`config.toml` directly.

#### Scenario: Valid configuration loads
- **WHEN** the use case is called with a `CycleConfig` containing all required
  keys
- **THEN** the cycle is initialized with those values

#### Scenario: Missing configuration key
- **WHEN** a required configuration value is absent from the `CycleConfig`
- **THEN** initialization fails with an error naming the missing key

#### Scenario: No file reads
- **WHEN** the use case executes
- **THEN** it does not open `config.toml`

### Requirement: Load initial state
`run_cycle` SHALL load the initial state: the active prompt version via
`prompt_repository`, the prepared dataset via `dataset_repository`, and the
fixed extraction prompt. The cycle SHALL start with an empty thesis collection
and the active prompt. The use case SHALL raise an exception naming the missing
artifact when the active version or dataset is not found.

#### Scenario: Initial state loaded
- **WHEN** the active version v0 and dataset exist
- **THEN** the cycle starts with an empty thesis collection and v0 active

#### Scenario: Missing active prompt
- **WHEN** the active version is not found
- **THEN** the use case raises an exception naming the missing active prompt

#### Scenario: Missing dataset
- **WHEN** the dataset is not found
- **THEN** the use case raises an exception naming the missing dataset

### Requirement: Run a single round
`run_cycle` SHALL execute a single round as a fixed sequence of steps:

1. Run the active prompt version on dev via `stage1-baseline-runner`.
2. Compute metrics via `stage1-metrics`.
3. Update the thesis collection and clusters via `stage1-thesis-analyzer`.
4. Select rule candidates via `stage2-rule-candidate-selector`.
5. Compose a new prompt version via `stage3-prompt-composer`.
6. Run the new version on dev via `stage1-baseline-runner`.
7. Compare versions and decide via `stage2-version-comparator`.
8. Update the active version on accept or return the next candidate on
   rollback.
9. Write the per-round report and state dump.

#### Scenario: Round completes
- **WHEN** a round is executed
- **THEN** all nine steps are performed in order

#### Scenario: Step failure
- **WHEN** a step fails with an unrecoverable error
- **THEN** the round is aborted and the cycle stops with an error naming the
  failing step

### Requirement: Round counter
`run_cycle` SHALL maintain a round counter starting at 0. Each successfully
completed round SHALL increment the counter by one. The cycle SHALL stop with
reason `max_rounds_reached` when the counter reaches `max_rounds`.

#### Scenario: Counter increments
- **WHEN** a round completes successfully
- **THEN** the round counter is incremented by one

#### Scenario: Cycle stops at max rounds
- **WHEN** the counter reaches `max_rounds`
- **THEN** the cycle stops with reason `max_rounds_reached`

### Requirement: Candidate queue
`run_cycle` SHALL maintain a candidate queue populated by the
`stage2-rule-candidate-selector` at the start of each round. The queue SHALL be
consumed in order. On rollback, the next candidate from the queue SHALL be
returned to the composer. When the queue is exhausted and a rollback occurs,
the cycle SHALL stop with reason `candidate_queue_exhausted`.

#### Scenario: Queue consumed in order
- **WHEN** the composer requests a candidate
- **THEN** the next candidate from the queue is returned

#### Scenario: Queue refilled each round
- **WHEN** a new round starts
- **THEN** the queue is refilled from the current round's selection

#### Scenario: Queue exhausted on rollback
- **WHEN** a rollback occurs and the queue is empty
- **THEN** the cycle stops with reason `candidate_queue_exhausted`

### Requirement: Rollback state
`run_cycle` SHALL track consecutive rollbacks. When the rollback counter reaches
`max_consecutive_rollbacks`, the cycle SHALL stop with reason
`max_rollbacks_reached`. The counter SHALL be reset to zero on acceptance.

#### Scenario: Rollbacks tracked
- **WHEN** a version is rolled back
- **THEN** the rollback counter is incremented

#### Scenario: Cycle stops after max rollbacks
- **WHEN** the rollback counter reaches `max_consecutive_rollbacks`
- **THEN** the cycle stops with reason `max_rollbacks_reached`

#### Scenario: Cycle stops after two rollbacks
- **WHEN** the rollback counter reaches `max_consecutive_rollbacks` (default 2)
- **THEN** the cycle stops with reason `max_rollbacks_reached`

#### Scenario: Counter reset on accept
- **WHEN** a version is accepted
- **THEN** the rollback counter is reset to zero

### Requirement: Accept updates active version
When a new version is accepted, `run_cycle` SHALL activate the new version via
`prompt_repository`. The previous active version SHALL be archived. The
`in_prompt` flags in the thesis collection SHALL be updated: `true` for theses
whose clusters produced accepted rules, `false` for all others.

#### Scenario: Active version updated
- **WHEN** a new version is accepted
- **THEN** the prompt repository marks it as active

#### Scenario: Previous version archived
- **WHEN** a new version is activated
- **THEN** the previous active version is archived

#### Scenario: Thesis flags updated
- **WHEN** a new version is accepted
- **THEN** `in_prompt` is set to true for theses in accepted clusters and false
  for others

#### Scenario: Thesis bank updated
- **WHEN** a new version is accepted
- **THEN** the thesis collection's `in_prompt` flags are updated to reflect the
  new rule set

### Requirement: Per-round report
`run_cycle` SHALL produce a per-round report after each completed round. The
report SHALL contain: round number, active version at round start, new version
composed, decision (accept or rollback), dev metrics for both versions, holdout
metrics for both versions (logged only, not used in decisions), rollback
counter value, list of changed decision ids, and next action. The report SHALL
be saved via `run_repository`.

#### Scenario: Report written
- **WHEN** a round completes
- **THEN** a report is written with all required fields

#### Scenario: Report readable
- **WHEN** the report is loaded
- **THEN** all fields are available without recomputation

### Requirement: Final summary
`run_cycle` SHALL produce a final summary when the cycle stops. The summary
SHALL contain: total rounds executed, stop reason, final active version, final
dev and holdout metrics, history of accepted versions, history of rollbacks,
and the list of all changed decisions across rounds. The summary SHALL be saved
via `run_repository`. The stop reason SHALL be one of: `max_rounds_reached`,
`max_rollbacks_reached`, `candidate_queue_exhausted`, `unrecoverable_error`.

#### Scenario: Summary written on stop
- **WHEN** the cycle stops for any reason
- **THEN** a final summary is written with all required fields

#### Scenario: Stop reasons enumerated
- **WHEN** the cycle stops
- **THEN** the stop reason is one of: `max_rounds_reached`,
  `max_rollbacks_reached`, `candidate_queue_exhausted`,
  `unrecoverable_error`

### Requirement: State dump
`run_cycle` SHALL save the cycle state after each round via `run_repository`.
The state SHALL contain: round counter, active version, thesis collection,
clusters, candidate queue, rollback counter, and the path to the latest report.
The state SHALL be sufficient to resume the cycle without re-running completed
rounds.

#### Scenario: State dumped
- **WHEN** a round completes
- **THEN** the full state is saved

#### Scenario: State reloaded
- **WHEN** the state is loaded
- **THEN** the cycle can resume from the same point without re-running
  completed rounds

### Requirement: Resumability
`run_cycle` SHALL support resuming the cycle from a saved state. On resume,
completed rounds SHALL be skipped. The cycle SHALL continue from the next
round. The thesis collection, clusters, active version, and rollback counter
SHALL match the saved state.

#### Scenario: Resume after interruption
- **WHEN** the cycle is interrupted after round 3
- **THEN** it can be resumed starting from round 4

#### Scenario: Resume preserves state
- **WHEN** resuming
- **THEN** the thesis collection, clusters, active version, and rollback
  counter match the saved state

### Requirement: Logging of cycle events
`run_cycle` SHALL log cycle events: round start, round end, decision,
accept/rollback, stop reason, and errors. The log SHALL be append-only. Each
event SHALL carry a timestamp and round number. The module SHALL NOT call
`logging.basicConfig`.

#### Scenario: Events logged
- **WHEN** any cycle event occurs
- **THEN** it is appended to the cycle log with timestamp and round number

#### Scenario: No logging configuration
- **WHEN** the module is imported
- **THEN** no logging configuration is performed

#### Scenario: Log readable per round
- **WHEN** the log is filtered by round number
- **THEN** only events from that round are returned

### Requirement: Configurable parameters
`run_cycle` SHALL support configuration via `CycleConfig` for `max_rounds`
(default 5), `max_consecutive_rollbacks` (default 2), `decision_metric`
(default `macro_f1`), `tie_breaker_metric` (default `minority_f1`),
`stop_on_first_error` (default false), `dump_state_after_each_round` (default
true), and `use_teacher_refinement` (default false).

#### Scenario: Custom max rounds
- **WHEN** the configuration sets `max_rounds=3`
- **THEN** the cycle stops after three rounds

#### Scenario: Stop on first error
- **WHEN** the configuration sets `stop_on_first_error=true` and a step fails
- **THEN** the cycle stops immediately
