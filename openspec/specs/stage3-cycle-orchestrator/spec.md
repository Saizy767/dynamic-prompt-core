# Spec

## Purpose

Drive the full optimization loop across a configured number of rounds,
coordinating the baseline runner, metrics, thesis analyzer, rule-candidate
selector, prompt composer, and version comparator. Maintain the round counter,
candidate queue, and rollback state, produce per-round reports and a final
summary, dump reloadable state after each round, and support resuming an
interrupted cycle. This is the top-level entry point of the system.

## Requirements

### Requirement: Load configuration
The component SHALL load all configuration required to run the cycle:
`max_rounds` (default 5), `max_consecutive_rollbacks` (default 2),
`dev_split`, `holdout_split`, `prompt_store_path`, `thesis_bank_path`,
`candidate_queue_path`, and paths for all intermediate artifacts. The component
SHALL fail with an error naming the missing key when required configuration is
absent.

#### Scenario: Valid configuration loads
- **WHEN** the config contains all required keys
- **THEN** the cycle is initialized with those values

#### Scenario: Missing configuration key
- **WHEN** the config is missing `max_rounds`
- **THEN** initialization fails with an error naming `max_rounds`

### Requirement: Load initial state
The component SHALL load the initial state: the active prompt version (v0
baseline), the prepared dataset artifact (dev and holdout splits), the fixed
extraction prompt, and the initial (empty) thesis bank. The component SHALL
fail with an error when the active prompt or the dataset artifact is absent.

#### Scenario: Initial state loaded
- **WHEN** the active prompt v0 and dataset artifact exist
- **THEN** the cycle starts with an empty thesis bank and the v0 prompt active

#### Scenario: Missing active prompt
- **WHEN** no active prompt version exists
- **THEN** the cycle fails with an error naming the missing active prompt

#### Scenario: Missing dataset
- **WHEN** the dataset artifact is absent
- **THEN** the cycle fails with an error naming the missing dataset

### Requirement: Run a single round
The component SHALL execute a single round as a fixed sequence of steps: run the
active prompt on dev via the baseline runner, compute metrics, update the thesis
bank and clusters, select rule candidates, compose a new prompt version, run the
new version on dev, delegate the accept-or-rollback decision to the version
comparator, update the active version or return the next candidate on rollback,
and write the per-round report.

#### Scenario: Round completes
- **WHEN** a round is executed
- **THEN** all nine steps are performed in order and the round report is written

#### Scenario: Step failure
- **WHEN** any step fails with an unrecoverable error
- **THEN** the round is aborted and the cycle stops with an error naming the
  failing step

### Requirement: Round counter
The component SHALL maintain a round counter starting at 0. Each successfully
completed round SHALL increment the counter by one. The cycle SHALL stop when
the counter reaches `max_rounds`.

#### Scenario: Counter increments
- **WHEN** a round completes successfully
- **THEN** the round counter is incremented by one

#### Scenario: Cycle stops at max rounds
- **WHEN** the counter reaches `max_rounds`
- **THEN** the cycle stops and the stop reason is logged as
  `max_rounds_reached`

### Requirement: Candidate queue
The component SHALL maintain a candidate queue populated by the rule-candidate
selector at the start of each round. The queue SHALL be consumed in order. On
rollback, the next candidate from the queue SHALL be returned to the prompt
composer for a new attempt. When the queue is exhausted and a rollback occurs,
the cycle SHALL stop.

#### Scenario: Queue consumed in order
- **WHEN** the composer requests a candidate
- **THEN** the next candidate from the queue is returned

#### Scenario: Queue refilled each round
- **WHEN** a new round starts
- **THEN** the queue is refilled from the current round's selection

#### Scenario: Queue exhausted on rollback
- **WHEN** a rollback occurs and the queue is empty
- **THEN** the cycle stops with stop reason `candidate_queue_exhausted`

### Requirement: Rollback state
The component SHALL track consecutive rollbacks via the version comparator. When
the rollback counter reaches `max_consecutive_rollbacks`, the cycle SHALL stop
with stop reason `max_rollbacks_reached`. The counter SHALL be reset on
acceptance.

#### Scenario: Rollbacks tracked
- **WHEN** a version is rolled back
- **THEN** the rollback counter is incremented

#### Scenario: Cycle stops after two rollbacks
- **WHEN** the rollback counter reaches `max_consecutive_rollbacks`
- **THEN** the cycle stops with stop reason `max_rollbacks_reached`

### Requirement: Accept updates active version
When the version comparator accepts a new version, the orchestrator SHALL mark
the new version as active in the prompt store and update the thesis bank's
`in_prompt` flags to reflect the new rule set.

#### Scenario: Active version updated
- **WHEN** a new version is accepted
- **THEN** the prompt store marks it as active and the previous active version
  is archived

#### Scenario: Thesis bank updated
- **WHEN** a new version is accepted
- **THEN** `in_prompt` is set to true for theses belonging to clusters that
  produced accepted rules, and false for others

### Requirement: Per-round report
The component SHALL produce a per-round report after each completed round
containing: round number, active version at round start, new version composed,
decision (accept or rollback), metrics on dev for both versions, metrics on
holdout for both versions (logged only, not used in decisions), rollback
counter value, list of changed decisions, and next action.

#### Scenario: Report written
- **WHEN** a round completes
- **THEN** a report is written with all required fields

#### Scenario: Report readable
- **WHEN** the report is loaded
- **THEN** all fields are available without recomputation

### Requirement: Final summary
The component SHALL produce a final summary when the cycle stops, containing:
total rounds executed, stop reason, final active version, metrics of the final
version on dev and holdout, the history of accepted versions, the history of
rollbacks, and the list of all changed decisions across rounds.

#### Scenario: Summary written on stop
- **WHEN** the cycle stops for any reason
- **THEN** a final summary is written with all required fields

#### Scenario: Stop reasons enumerated
- **WHEN** the cycle stops
- **THEN** the stop reason is one of: `max_rounds_reached`,
  `max_rollbacks_reached`, `candidate_queue_exhausted`,
  `unrecoverable_error`

### Requirement: State dump
The component SHALL dump the full cycle state after each round to a JSON file
whose filename contains `run_id` and timestamp. The state SHALL contain: round
counter, active version, thesis bank, clusters, candidate queue, rollback
counter, and the path to the latest report.

#### Scenario: State dumped
- **WHEN** a round completes
- **THEN** the full state is dumped to JSON

#### Scenario: State reloaded
- **WHEN** the state file is loaded
- **THEN** the cycle can resume from the same point without re-running completed
  rounds

### Requirement: Resumability
The component SHALL support resuming the cycle from a dumped state. On resume,
the component SHALL skip already-completed rounds and continue from the next
round.

#### Scenario: Resume after interruption
- **WHEN** the cycle is interrupted after round 3
- **THEN** it can be resumed starting from round 4

#### Scenario: Resume preserves state
- **WHEN** resuming
- **THEN** the thesis bank, clusters, active version, and rollback counter match
  the dumped state

### Requirement: Logging of cycle events
The component SHALL log all cycle events: round start, round end, decision,
accept/rollback, stop reason, and any errors. The log SHALL be append-only and
queryable per round.

#### Scenario: Events logged
- **WHEN** any cycle event occurs
- **THEN** it is appended to the cycle log with timestamp and round number

#### Scenario: Log readable per round
- **WHEN** the log is filtered by round number
- **THEN** only events from that round are returned

### Requirement: Configurable parameters
The component SHALL support configuration via config for `max_rounds` (default
5), `max_consecutive_rollbacks` (default 2), `stop_on_first_error` (default
false), `dump_state_after_each_round` (default true), and `report_format`
(default json).

#### Scenario: Custom max rounds
- **WHEN** config sets `max_rounds=3`
- **THEN** the cycle stops after three rounds

#### Scenario: Stop on first error
- **WHEN** config sets `stop_on_first_error=true` and any step fails
- **THEN** the cycle stops immediately
