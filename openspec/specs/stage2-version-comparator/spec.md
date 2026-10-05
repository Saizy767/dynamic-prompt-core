# Spec

## Purpose

Load a newly composed prompt version and the currently active prompt version, run
both on the same dev split, compute comparable metrics, compare by macro-F1 with
minority-class F1 as tie-breaker, decide to accept or roll back the new version,
maintain a consecutive-rollback counter, return the next candidate on rollback,
and persist a reloadable decision artifact with full lineage and changed
predictions.

## Requirements

### Requirement: Load new prompt version
The component SHALL load the prompt artifact produced by the composer component,
containing `version`, `text`, `hash`, `rules`, `source_candidates`, and
`base_version`. The component SHALL fail with an error naming the missing field
when the artifact does not conform to the schema.

#### Scenario: Valid artifact loads
- **WHEN** the artifact contains all required fields
- **THEN** the new version is loaded for evaluation

#### Scenario: Missing field
- **WHEN** the artifact is missing `source_candidates`
- **THEN** loading fails with an error naming the missing field

### Requirement: Load active version
The component SHALL load the currently active prompt version from the prompt
store. The active version SHALL be used as the baseline for comparison. The
component SHALL fail with an error when no active version is found.

#### Scenario: Active version loaded
- **WHEN** the prompt store contains an active version
- **THEN** the active version is loaded as the baseline

#### Scenario: No active version
- **WHEN** the prompt store has no active version
- **THEN** the component fails with an error naming the missing active version

### Requirement: Run new version on dev
The component SHALL run the new prompt version on the same dev split used for the
active version. The run SHALL use the same runner and the same fixed extraction
prompt as the baseline. The component SHALL produce a results artifact for the
new version.

#### Scenario: New version run
- **WHEN** the new version is evaluated
- **THEN** a results artifact is produced for the new version on dev

#### Scenario: Same extraction prompt
- **WHEN** the new version is run
- **THEN** the fixed extraction prompt is used without modification

### Requirement: Compute metrics for new version
The component SHALL compute the same metrics for the new version as for the
active version: accuracy, macro-F1, minority-class F1, confusion matrix,
prediction distribution, and parse-failure count. The metrics SHALL be computed
by the same stage1-metrics component used for the baseline.

#### Scenario: Metrics computed
- **WHEN** the new version is evaluated
- **THEN** accuracy, macro-F1, minority-class F1, confusion matrix, prediction
  distribution, and parse-failure count are computed

#### Scenario: Metrics comparable
- **WHEN** metrics are computed for the new version
- **THEN** they are directly comparable to the active version's metrics on dev

### Requirement: Compare metrics
The component SHALL compare the new version's metrics against the active
version's metrics on dev. The comparison SHALL include accuracy, macro-F1, and
minority-class F1. The component SHALL report the difference for each metric.

#### Scenario: Comparison reported
- **WHEN** the new version is evaluated
- **THEN** the difference in accuracy, macro-F1, and minority-class F1 relative
  to the active version is reported

#### Scenario: Metrics on holdout
- **WHEN** metrics are computed on holdout
- **THEN** they are logged but SHALL NOT participate in the accept/rollback
  decision

### Requirement: Decision rule
The component SHALL decide to accept the new version when its macro-F1 on dev is
greater than or equal to the active version's macro-F1. When macro-F1 is equal,
the component SHALL use minority-class F1 as the tie-breaker. When both macro-F1
and minority-class F1 are lower, the component SHALL decide to roll back.

#### Scenario: Accept on improvement
- **WHEN** the new version's macro-F1 is higher than the active version's
- **THEN** the new version is accepted

#### Scenario: Accept on tie
- **WHEN** the new version's macro-F1 equals the active version's and
  minority-class F1 is higher
- **THEN** the new version is accepted

#### Scenario: Rollback on degradation
- **WHEN** the new version's macro-F1 is lower than the active version's
- **THEN** the new version is rolled back

### Requirement: Rollback counter
The component SHALL maintain a counter of consecutive rollbacks. The counter
SHALL be reset to zero on acceptance. When the counter reaches
`max_consecutive_rollbacks` (default 2), the component SHALL stop the cycle and
log the stop reason.

#### Scenario: Counter reset on accept
- **WHEN** a version is accepted
- **THEN** the rollback counter is reset to zero

#### Scenario: Counter incremented on rollback
- **WHEN** a version is rolled back
- **THEN** the rollback counter is incremented by one

#### Scenario: Cycle stops after two rollbacks
- **WHEN** the rollback counter reaches two
- **THEN** the cycle stops and the stop reason is logged

### Requirement: Next candidate on rollback
The component SHALL return the next candidate from the candidate queue when a
version is rolled back. The component SHALL NOT reuse the rejected version's
rules for the next attempt.

#### Scenario: Next candidate returned
- **WHEN** a version is rolled back
- **THEN** the next candidate from the queue is returned for composition

#### Scenario: Queue exhausted
- **WHEN** the candidate queue is exhausted and a rollback occurs
- **THEN** the cycle stops and the stop reason is logged

### Requirement: Decision artifact
The component SHALL persist the decision as a JSON artifact whose filename
contains `run_id`, `new_version`, `active_version`, and `timestamp`. The
artifact SHALL contain: `decision` (accept or rollback), `new_version`,
`active_version`, `metrics_new`, `metrics_active`, `diff_accuracy`,
`diff_macro_f1`, `diff_minority_f1`, `rollback_count`, and `reason`.

#### Scenario: Artifact named consistently
- **WHEN** a decision is made
- **THEN** the file is named
  `decision_{run_id}_v{new_version}_vs_v{active_version}_{timestamp}.json`

#### Scenario: Artifact reloaded
- **WHEN** the artifact is loaded
- **THEN** the decision, versions, metrics, and reason are available

### Requirement: Changed decisions
The component SHALL report the list of example ids whose `predicted_decision`
changed between the active version and the new version. The list SHALL include
the direction of change (correct→incorrect, incorrect→correct).

#### Scenario: Changed decisions listed
- **WHEN** the new version fixes 5 examples and breaks 3
- **THEN** the list contains 8 ids with the direction of change

#### Scenario: No changed decisions
- **WHEN** no `predicted_decision` changed
- **THEN** the list is empty

### Requirement: Logging of decision
The component SHALL log the decision process: new version, active version,
metrics for both, differences, rollback count, and final decision.

#### Scenario: Decision logged
- **WHEN** a decision is made
- **THEN** the log contains `new_version`, `active_version`, `metrics_new`,
  `metrics_active`, `diff_accuracy`, `diff_macro_f1`, `diff_minority_f1`,
  `rollback_count`, and `decision`

### Requirement: Configurable parameters
The component SHALL support configuration via config for
`max_consecutive_rollbacks` (default 2), `decision_metric` (default `macro_f1`),
and `tie_breaker_metric` (default `minority_f1`).

#### Scenario: Custom rollback limit
- **WHEN** config sets `max_consecutive_rollbacks=3`
- **THEN** the cycle stops after three consecutive rollbacks

#### Scenario: Custom decision metric
- **WHEN** config sets `decision_metric=accuracy`
- **THEN** the decision is based on accuracy instead of macro-F1
