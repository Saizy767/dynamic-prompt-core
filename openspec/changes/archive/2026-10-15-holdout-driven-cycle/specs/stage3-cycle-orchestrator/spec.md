# Spec Delta

## ADDED Requirements

### Requirement: Cycle/cluster versioning
`run_cycle` SHALL name intermediate prompt versions using the scheme
`vN_k` where `N` is the cycle number and `k` is the cluster index within
that cycle (1-indexed). The finalized best version of cycle `N` SHALL be
renamed to `vN`. The baseline `CLASSIFICATION_PROMPT_V0` SHALL be `v0`.
Cycle `N` starts with base `v(N-1)`, produces intermediates `vN_1`,
`vN_2`, …, and finalizes the best as `vN`.

#### Scenario: Baseline is v0
- **WHEN** the cycle starts from `CLASSIFICATION_PROMPT_V0`
- **THEN** the baseline version is named `v0`

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

## MODIFIED Requirements

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
   d. When `vN_k`'s holdout metric is strictly greater than the current
      best, accept it: update the current best version, mark the
      candidate's cluster as `in_prompt`, and continue to the next
      candidate (step 6a).
   e. When `vN_k`'s holdout metric is not strictly greater (equal or
      lower), stop accumulation: discard the candidate's rule, keep the
      current best version, and proceed to step 7.
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
- **WHEN** candidates 1 and 2 both strictly improve the holdout metric and
  candidate 3 does not
- **THEN** the cycle ends with the finalized version containing rules from
  candidates 1 and 2, and candidate 3's rule is discarded

#### Scenario: First candidate degrades
- **WHEN** the first candidate does not strictly improve the holdout metric
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
be consumed in order during greedy accumulation. When accumulation stops due
to degradation, remaining candidates SHALL stay in the queue for the next
cycle. The queue SHALL be refilled from the current cycle's selection at the
start of each cycle.

#### Scenario: Queue consumed in order
- **WHEN** the composer requests a candidate
- **THEN** the next candidate from the queue is returned

#### Scenario: Queue refilled each round
- **WHEN** a new cycle starts
- **THEN** the queue is refilled from the current cycle's selection

#### Scenario: Remaining candidates preserved on degradation
- **WHEN** accumulation stops due to degradation with candidates remaining
- **THEN** the remaining candidates stay in the queue for the next cycle

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

### Requirement: Per-round report
`run_cycle` SHALL produce a per-cycle report after each completed cycle. The
report SHALL contain: cycle number, active version (base) at cycle start,
finalized version, decision (accept or no improvement), dev metrics for the
active version (from step 1), holdout metrics for the active version and the
finalized version, the list of `vN_k` intermediate attempts with their
holdout metrics and accept/rollback decisions, accepted cluster ids, and next
action. The report SHALL be saved via `run_repository`.

#### Scenario: Report written
- **WHEN** a cycle completes
- **THEN** a report is written with all required fields

#### Scenario: Report readable
- **WHEN** the report is loaded
- **THEN** all fields are available without recomputation

### Requirement: Final summary
`run_cycle` SHALL produce a final summary when the process stops. The summary
SHALL contain: total cycles executed, stop reason, final active version, final
dev and holdout metrics, history of accepted versions, history of rollbacks,
and the list of all changed decisions across cycles. The summary SHALL be
saved via `run_repository`. The stop reason SHALL be one of:
`max_cycles_reached`, `no_improvement`, `unrecoverable_error`,
`plateau_detected`.

#### Scenario: Summary written on stop
- **WHEN** the process stops for any reason
- **THEN** a final summary is written with all required fields

#### Scenario: Stop reasons enumerated
- **WHEN** the process stops
- **THEN** the stop reason is one of: `max_cycles_reached`,
  `no_improvement`, `unrecoverable_error`, `plateau_detected`

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

### Requirement: Load configuration
`run_cycle` SHALL accept cycle configuration as a typed object (`CycleConfig`)
containing `max_cycles` (default 5), `decision_metric` (default `macro_f1`),
`tie_breaker_metric` (default `minority_f1`), `dev_split`, `holdout_split`,
artifact paths, `run_id`, `stop_on_first_error` (default false),
`dump_state_after_each_cycle` (default true), and
`use_teacher_refinement` (default false). The configuration SHALL be loaded
by the composition root and passed in. The use case SHALL NOT read
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

### Requirement: Configurable parameters
`run_cycle` SHALL support configuration via `CycleConfig` for `max_cycles`
(default 5), `decision_metric` (default `macro_f1`), `tie_breaker_metric`
(default `minority_f1`), `stop_on_first_error` (default false),
`dump_state_after_each_cycle` (default true), and
`use_teacher_refinement` (default false).

#### Scenario: Custom max rounds
- **WHEN** the configuration sets `max_cycles=3`
- **THEN** the process stops after three cycles

#### Scenario: Stop on first error
- **WHEN** the configuration sets `stop_on_first_error=true` and a step fails
- **THEN** the process stops immediately

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

## REMOVED Requirements

### Requirement: Rollback state
**Reason**: Replaced by the `no_improvement` stop condition. A cycle ends
normally on degradation (the next cluster does not strictly improve). The
process stops only when the first candidate of a cycle fails — i.e., no
improvement at all. The consecutive-rollback counter is no longer needed.
**Migration**: Use the `no_improvement` stop reason instead of
`max_rollbacks_reached`. Remove `max_consecutive_rollbacks` from
`CycleConfig` and `[cycle_orchestrator]` config.
