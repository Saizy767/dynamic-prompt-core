# Spec

## Purpose

Drive the full optimization loop across a configured number of rounds,
coordinating the baseline runner, metrics, thesis analyzer, rule-candidate
selector, prompt composer, and version comparator. Maintain the round counter,
candidate queue, and rollback state, produce per-round reports and a final
summary, dump reloadable state after each round, and support resuming an
interrupted cycle. This is the top-level entry point of the system.

## Requirements

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
(`RunCycleDeps`). The dependency object SHALL contain outbound ports and
collaborators: `llm_client`, `prompt_repository`, `run_repository`,
`normalizer`, `classifier`, `extractor`, `candidates` (a tuple), and
optionally `teacher_llm_client` and `stop_criteria_deps`. The dataset is
supplied as a file path; embeddings are loaded inside the analyzer/composer
components. The dependency object SHALL NOT contain concrete infrastructure
implementations. The use case SHALL NOT create dependencies itself or use a
service locator.

#### Scenario: Typed dependency object
- **WHEN** the use case is called
- **THEN** it receives a dependency object containing typed ports and collaborators

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
prompt repository, run repository, normalizer, classifier, extractor, the
candidate tuple, teacher client (when enabled), and stop-criteria dependencies
(when a `[stop_criteria]` config section exists). The dataset is passed as a
file path. The composition root SHALL NOT contain business logic. No other
module SHALL create these dependencies.

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
the use case itself SHALL stop the cycle: the use case SHALL write a final
summary with reason `unrecoverable_error` and then re-raise the exception to
the caller.

#### Scenario: Single example failure
- **WHEN** the model returns invalid JSON for an example
- **THEN** the record is saved with a failed status and the cycle continues

#### Scenario: Unrecoverable error
- **WHEN** the use case raises an unrecoverable exception
- **THEN** a final summary with reason `unrecoverable_error` is written and the exception propagates to the caller

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

### Requirement: Load configuration
`run_cycle` SHALL accept cycle configuration as a typed object (`CycleConfig`)
containing `max_cycles` (default 5), `decision_metric` (default `macro_f1`),
`tie_breaker_metric` (default `minority_f1`), `dev_split`, `holdout_split`,
artifact paths, `run_id`, `dump_state_after_each_cycle` (default true), and
`use_teacher_refinement` (default false). The configuration SHALL be loaded
by the composition root and passed in. When no config object is supplied, the
use case SHALL fall back to loading `CycleConfig` from the run input's config
path. Per-step component configurations (runner, analyzer, selector, composer)
SHALL be loaded by each step from the same config path — documented delegation
to the components' own loaders.

#### Scenario: Valid configuration loads
- **WHEN** the use case is called with a `CycleConfig` containing all required
  keys
- **THEN** the cycle is initialized with those values

#### Scenario: Missing configuration key
- **WHEN** a required configuration value is absent from the `CycleConfig`
- **THEN** initialization fails with an error naming the missing key

#### Scenario: Fallback config load
- **WHEN** the use case is invoked without a config object
- **THEN** it loads `CycleConfig` from the run input's config path

### Requirement: Load initial state
`run_cycle` SHALL load the initial state: the active prompt version via
`prompt_repository`, the dataset from its file path, and the fixed extraction
prompt. The cycle SHALL start with an empty thesis collection and the active
prompt. The use case SHALL raise an exception naming the missing artifact when
the active version or dataset is not found; a missing active prompt SHALL NOT
silently fall back to the baseline prompt.

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
`run_cycle` SHALL execute a single cycle as a fixed sequence of steps:

1. Run the active prompt version on dev via `stage1-baseline-runner` with
   thesis extraction enabled. This is the only dev run in the cycle.
2. Compute dev metrics via `stage1-metrics` for logging.
3. Update the thesis collection and clusters via `stage1-thesis-analyzer`.
4. Select rule candidates via `stage2-rule-candidate-selector`.
5. Run the active version on holdout via `stage1-baseline-runner` with
   thesis extraction disabled. Cache the holdout metrics as the baseline
   for the accumulation loop.
6. Greedily accumulate rules from the candidate queue:
   a. Compose a new prompt version `vN_k` from the next candidate via
      `stage3-prompt-composer`, appending the new rule to the rules already
      accepted in this cycle.
   b. Run `vN_k` on holdout only (no dev run per candidate) via
      `stage1-baseline-runner` with thesis extraction disabled.
   c. Compare `vN_k` against the current best version via
      `stage2-version-comparator` using holdout metrics.
   d. Accept `vN_k` when its holdout decision metric is strictly greater
      than the current best's, or when the decision metric ties and the
      tie-breaker metric is strictly greater: update the current best
      version, mark the candidate's cluster as `in_prompt`, and continue
      to the next candidate (step 6a).
   e. Otherwise (decision metric lower, or tie without tie-breaker
      improvement), stop accumulation: discard the candidate's rule, keep
      the current best version, and proceed to step 7.
   f. When the number of accumulated rules reaches `max_rules`, stop
      accumulation and proceed to step 7.
   g. When the candidate queue is exhausted, proceed to step 7.
7. Finalize the current best version as `vN`.
8. Write the per-cycle report and state dump.

The cycle SHALL NOT end on the first acceptance. The cycle SHALL continue
trying candidates until degradation, the `max_rules` limit, or queue
exhaustion. Candidate versions SHALL NOT be run on dev — only holdout.

#### Scenario: Round completes
- **WHEN** a cycle is executed
- **THEN** steps 1–8 are performed in order

#### Scenario: Multiple rules accumulated
- **WHEN** candidates 1 and 2 both improve the holdout decision metric
  (strictly, or on a tie via the tie-breaker) and candidate 3 does not
- **THEN** the cycle ends with the finalized version containing rules from
  candidates 1 and 2, and candidate 3's rule is discarded

#### Scenario: First candidate degrades
- **WHEN** the first candidate neither strictly improves the holdout decision
  metric nor wins the tie-breaker on a tie
- **THEN** the cycle ends with the active version unchanged (no rules added)
  and the process stops with reason `no_improvement`

#### Scenario: Accumulation stops at max_rules
- **WHEN** the number of accepted rules reaches `max_rules`
- **THEN** accumulation stops and the remaining candidates are left in the queue

#### Scenario: Step failure
- **WHEN** a step fails with an unrecoverable error
- **THEN** the cycle is aborted and the process stops with an error naming the failing step

### Requirement: Round counter
`run_cycle` SHALL maintain a cycle counter starting at 0. Each successfully
completed cycle SHALL increment the counter by one. The process SHALL stop
with reason `max_cycles_reached` when the counter reaches `max_cycles`.

#### Scenario: Counter increments
- **WHEN** a cycle completes successfully
- **THEN** the cycle counter is incremented by one

#### Scenario: Cycle stops at max rounds
- **WHEN** the cycle counter reaches `max_cycles`
- **THEN** the process stops with reason `max_cycles_reached`

### Requirement: Candidate queue
`run_cycle` SHALL maintain a candidate queue populated by the
`stage2-rule-candidate-selector` at the start of each cycle. The queue SHALL
be consumed in order during greedy accumulation. The queue SHALL be REPLACED
by the current cycle's fresh selection at the start of each cycle: leftover
candidates from the previous cycle are discarded, because each cycle's
selection reflects the current active version and thesis state.

#### Scenario: Queue consumed in order
- **WHEN** the composer requests a candidate
- **THEN** the next candidate from the queue is returned

#### Scenario: Queue replaced each cycle
- **WHEN** a new cycle starts
- **THEN** the queue is replaced by the current cycle's fresh selection, not appended to

#### Scenario: Queue exhausted without acceptance
- **WHEN** the queue is exhausted and no candidate was accepted
- **THEN** the process stops with reason `no_improvement`

### Requirement: Holdout without thesis extraction
`run_cycle` SHALL run holdout evaluation with thesis extraction disabled. The
holdout run SHALL only perform candidate-scoring classification for metrics
computation and SHALL NOT call `extract_theses_detailed`. Holdout metrics
SHALL drive the accept/rollback decision in the accumulation loop. The active
version's holdout metrics SHALL be computed once per cycle and cached as the
baseline for comparison.

#### Scenario: Holdout skips thesis extraction
- **WHEN** the holdout evaluation runs
- **THEN** no thesis-extraction LLM calls are made and only classification scoring is performed

#### Scenario: Holdout metrics logged only
- **WHEN** holdout metrics are computed
- **THEN** they drive the accept/rollback decision and are included in the
  per-cycle report

### Requirement: Accept updates active version
When a new version is accepted, `run_cycle` SHALL activate the new version via
`prompt_repository`. The previous active version SHALL be archived. The
`in_prompt` flags in the thesis collection SHALL be updated monotonically:
`true` for theses whose clusters produced accepted rules; flags that are
already `true` SHALL stay `true` (clearing them would let the selector
re-propose duplicate rules in later cycles).

#### Scenario: Active version updated
- **WHEN** a new version is accepted
- **THEN** the prompt repository marks it as active

#### Scenario: Previous version archived
- **WHEN** a new version is activated
- **THEN** the previous active version is archived

#### Scenario: Thesis flags updated monotonically
- **WHEN** a new version is accepted
- **THEN** `in_prompt` is set to true for theses in accepted clusters and
  previously-true flags remain true

#### Scenario: Thesis bank updated
- **WHEN** a new version is accepted
- **THEN** the thesis collection's `in_prompt` flags are updated to reflect the
  new rule set

### Requirement: Per-round report
`run_cycle` SHALL produce a per-cycle report after each completed cycle. The
report SHALL contain: cycle number, active version (base) at cycle start,
finalized version, decision (accept or no improvement), dev metrics for the
active version (from step 1), holdout metrics for the active version and the
finalized version, and next action. The report SHALL be saved via
`run_repository`.

#### Scenario: Report written
- **WHEN** a cycle completes
- **THEN** a report is written with all required fields

#### Scenario: Report readable
- **WHEN** the report is loaded
- **THEN** all fields are available without recomputation

### Requirement: Final summary
`run_cycle` SHALL produce a final summary when the process stops. The summary
SHALL contain: total cycles executed, stop reason, final active version, final
dev and holdout metrics, history of accepted versions, and history of
rollbacks. The summary SHALL be saved via `run_repository`. The stop reason
SHALL be one of: `max_cycles_reached`, `no_improvement`,
`unrecoverable_error`, `plateau_detected`, `budget_exhausted`,
`no_candidates_available`, `metric_degradation`, `rule_stagnation`,
`rollback_streak` (the last six arise from the optional stop-criteria
evaluation).

#### Scenario: Summary written on stop
- **WHEN** the process stops for any reason
- **THEN** a final summary is written with all required fields

#### Scenario: Stop reasons enumerated
- **WHEN** the process stops
- **THEN** the stop reason is one of the enumerated values above

### Requirement: State dump
`run_cycle` SHALL save the cycle state after each cycle via
`run_repository`. The state SHALL contain: cycle counter, cycle number,
cluster index, active version, thesis collection, clusters, candidate queue,
and the path to the latest report. The state SHALL be sufficient to resume
the process without re-running completed cycles.

#### Scenario: State dumped
- **WHEN** a cycle completes
- **THEN** the full state is saved

#### Scenario: State reloaded
- **WHEN** the state is loaded
- **THEN** the process can resume from the same point without re-running
  completed cycles

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
`run_cycle` SHALL log cycle events: cycle start, cycle end, decision,
accept/rollback, stop reason, and errors. The log SHALL be append-only. Each
event SHALL carry a timestamp and cycle number. The module SHALL NOT call
`logging.basicConfig`.

#### Scenario: Events logged
- **WHEN** any cycle event occurs
- **THEN** it is appended to the cycle log with timestamp and cycle number

#### Scenario: No logging configuration
- **WHEN** the module is imported
- **THEN** no logging configuration is performed

#### Scenario: Log readable per round
- **WHEN** the log is filtered by cycle number
- **THEN** only events from that cycle are returned

### Requirement: Configurable parameters
`run_cycle` SHALL support configuration via `CycleConfig` for `max_cycles`
(default 5), `decision_metric` (default `macro_f1`), `tie_breaker_metric`
(default `minority_f1`), `dump_state_after_each_cycle` (default true), and
`use_teacher_refinement` (default false).

#### Scenario: Custom max cycles
- **WHEN** the configuration sets `max_cycles=3`
- **THEN** the process stops after three cycles

### Requirement: Cycle/cluster versioning
`run_cycle` SHALL name intermediate prompt versions using the scheme
`vN_k` where `N` is the cycle number and `k` is the cluster index within
that cycle (1-indexed). The finalized best version of cycle `N` SHALL be
renamed to `vN`. The baseline `CLASSIFICATION_PROMPT_V0` carries the version
id `classify-v0`. Cycle `N` starts with base `v(N-1)`, produces intermediates
`vN_1`, `vN_2`, …, and finalizes the best as `vN`.

#### Scenario: Baseline version id
- **WHEN** the cycle starts from `CLASSIFICATION_PROMPT_V0`
- **THEN** the baseline version is named `classify-v0`

#### Scenario: Intermediate version naming
- **WHEN** cycle 2 composes its third cluster candidate
- **THEN** the intermediate version is named `v2_3`

#### Scenario: Finalized version naming
- **WHEN** cycle 2 finalizes its best version
- **THEN** the finalized version is named `v2`

#### Scenario: Next cycle base
- **WHEN** cycle 2 finalizes `v2`
- **THEN** cycle 3 starts with base `v2`

### Requirement: No improvement stop condition
`run_cycle` SHALL stop with reason `no_improvement` when a cycle ends with
no candidate accepted. This occurs when the first candidate of a cycle does
not strictly improve the holdout metric over the base version, or when the
candidate queue is empty at the start of accumulation.

#### Scenario: First candidate does not improve
- **WHEN** the first candidate of a cycle produces a holdout metric that is
  not strictly greater than the base
- **THEN** the cycle ends with no candidate accepted and the process stops
  with reason `no_improvement`

#### Scenario: Empty candidate queue
- **WHEN** the candidate queue is empty at the start of accumulation
- **THEN** the process stops with reason `no_improvement`

#### Scenario: Later candidate rollback is not no_improvement
- **WHEN** the first candidate of a cycle is accepted but a later candidate
  degrades
- **THEN** the cycle finalizes the best and the process continues to the
  next cycle (not `no_improvement`)

### Requirement: Content-based stop evaluation
`run_cycle` SHALL call `evaluate_stop_criteria` at the end of each cycle, after
the per-cycle report is written and before deciding whether to start the next
cycle, WHEN stop-criteria dependencies are wired. The composition root wires
`stop_criteria_deps` only when a `[stop_criteria]` section exists in
`config.toml`; with the section absent the evaluation is skipped and the
cycle relies on its hard limits alone. The call SHALL pass the current cycle
state: per-cycle metric history, decision history, cycle counter, rollback
counter, candidate-queue size, total teacher-model token spend, and the
stop-criteria configuration. When the evaluation returns `stop`, the cycle
SHALL terminate with the returned reason. When the evaluation returns
`continue`, the cycle SHALL proceed to the next cycle. This evaluation is
additional to the hard-limit checks (`max_cycles`, `no_improvement`) which
SHALL still be performed before starting a cycle.

#### Scenario: Orchestrator stops on criteria
- **WHEN** `evaluate_stop_criteria` returns `stop`
- **THEN** the cycle terminates with the returned reason

#### Scenario: Orchestrator continues
- **WHEN** `evaluate_stop_criteria` returns `continue`
- **THEN** the cycle proceeds to the next cycle

#### Scenario: Hard limits checked first
- **WHEN** a hard limit (`max_cycles`, `no_improvement`) is reached before a
  cycle starts
- **THEN** the cycle stops without calling `evaluate_stop_criteria`

#### Scenario: Optional wiring
- **WHEN** no `[stop_criteria]` section exists in `config.toml`
- **THEN** the composition root leaves `stop_criteria_deps` unset and the
  cycle never calls `evaluate_stop_criteria`

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

### Requirement: Atomic thesis bank persistence

When `update_in_prompt_flags` writes the updated thesis bank, it SHALL do so
atomically: the existing file SHALL remain untouched until the new content is
fully serialized, fsynced, and read-back validated as parseable JSON. The
replacement SHALL use an atomic rename (`os.replace`). If any step fails, the
original file SHALL be left intact and the temporary file SHALL be removed.

#### Scenario: Successful update replaces file
- **WHEN** `update_in_prompt_flags` writes a valid updated thesis bank
- **THEN** the file at `thesis_bank_path` contains the new content and the
  old content is fully replaced

#### Scenario: Original file preserved on write failure
- **WHEN** the write fails after the original file exists
- **THEN** the original file at `thesis_bank_path` is unchanged and no
  temporary file remains in the directory

#### Scenario: Temp file cleaned up on failure
- **WHEN** an exception occurs during serialization or validation
- **THEN** the temporary file is removed and does not persist in the
  thesis bank's directory
