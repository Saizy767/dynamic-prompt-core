# Spec

## Purpose

Transfer an optimized system prompt, thesis collection, and cluster centroids
from a source task to a structurally compatible target task, reformulate the
task layer, clear the rules, and re-run the optimization cycle on the target
dataset — enabling reuse of optimization work across tasks and measuring
whether the transfer accelerates convergence.

## Requirements

### Requirement: Source and target task definition
The component SHALL accept descriptions of two tasks: a source task (for which
the prompt is already optimized) and a target task (to which the prompt is
transferred). Each task SHALL be described with `task_id`, `dataset_path`,
`num_classes`, `class_labels`, and `metric` (one of `accuracy`, `macro_f1`, or
`minority_f1`). The component SHALL fail with an error naming the missing field
when a task description is incomplete.

#### Scenario: Both tasks defined
- **WHEN** the config contains complete descriptions of the source and target
  tasks
- **THEN** the component loads both datasets and determines the metric for each
  task

#### Scenario: Missing task field
- **WHEN** the target task description does not contain `metric`
- **THEN** initialization fails with an error naming the missing field

### Requirement: Same task structure required
The component SHALL require the source and target tasks to have the same
structure: the same number of classes, the same output format
(`decision`/`confidence`), and the same thesis-extraction schema. The component
SHALL fail with an error naming the mismatch when the structures differ.

#### Scenario: Same structure
- **WHEN** both tasks have two classes and the same output contract
- **THEN** the transfer is allowed

#### Scenario: Different number of classes
- **WHEN** the source task has two classes and the target has three
- **THEN** the transfer is rejected with an error naming the class-count mismatch

#### Scenario: Different output contract
- **WHEN** the source uses `decision` as a string and the target uses it as an
  integer
- **THEN** the transfer is rejected with an error naming the contract mismatch

### Requirement: Load source artifacts
The component SHALL load the source task artifacts: the final active prompt
version, the version history, the thesis collection with clusters, and the cycle
metrics. The component SHALL NOT modify the source artifacts. The component
SHALL fail with an error naming the missing artifact when a required artifact is
absent.

#### Scenario: Source artifacts loaded
- **WHEN** the source task has a completed cycle with saved artifacts
- **THEN** the component loads the active version, thesis collection, and
  metrics

#### Scenario: Missing source artifacts
- **WHEN** the source task has no saved active version
- **THEN** initialization fails with an error naming the missing artifact

### Requirement: Transfer base layers
The component SHALL transfer the base prompt layers (role, output contract,
fallback) from the source task to the target unchanged when the output structure
matches. The "task" layer SHALL be reformulated for the target. The "rules"
layer SHALL be fully cleared.

#### Scenario: Base layers transferred
- **WHEN** the source and target have the same output structure
- **THEN** the role, output contract, and fallback are transferred unchanged

#### Scenario: Task layer reformulated
- **WHEN** the target task has a different subject domain
- **THEN** the "task" layer is reformulated via the LLM for the target domain

#### Scenario: Rules cleared
- **WHEN** the transfer is initiated
- **THEN** the rules list in the new prompt version is empty, pending
  re-selection

### Requirement: Seed target collection with source theses
The component SHALL transfer theses from the source thesis collection into the
target collection as an initial approximation. Transferred theses SHALL be
tagged `source=transfer` and SHALL NOT participate in precision computation
until the first target run records occurrences. When `max_transfer_theses` is
set, only that many theses with the highest source frequency SHALL be
transferred.

#### Scenario: Theses transferred
- **WHEN** the source collection contains 120 theses
- **THEN** those 120 theses are added to the target collection tagged
  `source=transfer`

#### Scenario: Precision not affected
- **WHEN** a transferred thesis has not yet appeared in a target run
- **THEN** its precision is not counted in candidate selection

#### Scenario: Transfer budget
- **WHEN** the config sets `max_transfer_theses=50`
- **THEN** only the 50 theses with the highest source frequency are transferred

### Requirement: Seed target clusters with source centroids
The component SHALL transfer cluster centroids from the source task into the
target as initial clustering points. Transferred centroids SHALL be tagged
`source=transfer` and SHALL be updated when new theses from target runs are
added to their clusters.

#### Scenario: Centroids transferred
- **WHEN** the source has 20 clusters
- **THEN** 20 centroids are transferred into the target

#### Scenario: Centroids updated
- **WHEN** a new target thesis is assigned to a transferred cluster
- **THEN** the centroid is recomputed to include the new member

### Requirement: Run optimization cycle on target
The component SHALL run the optimization cycle on the target task using
`stage3-cycle-orchestrator`. The cycle SHALL start from the transferred prompt
version (base layers + empty rules) and the transferred thesis collection. All
cycle components (`stage1-baseline-runner`, `stage1-metrics`,
`stage1-thesis-analyzer`, `stage2-rule-candidate-selector`,
`stage3-prompt-composer-v2`, `stage2-version-comparator`) SHALL be used in their
standard configuration without modification.

#### Scenario: Cycle started
- **WHEN** the transfer is initiated
- **THEN** the cycle runs on the target dataset with the transferred artifacts

#### Scenario: Cycle uses standard components
- **WHEN** the cycle executes
- **THEN** all components are used in their standard configuration

#### Scenario: Target cycle independent
- **WHEN** the target cycle completes
- **THEN** the target artifacts are stored separately from the source and do not
  overwrite them

### Requirement: Transfer artifact
The component SHALL save the transfer result as an artifact whose filename
contains `source_task_id`, `target_task_id`, and timestamp, in JSON format. The
artifact SHALL contain `source_version`, `target_final_version`,
`transferred_theses_count`, `transferred_clusters_count`,
`target_metrics_start`, `target_metrics_final`, `improvement` (absolute and
relative gain on the metric), and `stop_reason`.

#### Scenario: Artifact named consistently
- **WHEN** the transfer completes
- **THEN** the file is named
  `transfer_{source_task_id}_to_{target_task_id}_{timestamp}.json`

#### Scenario: Artifact reloaded
- **WHEN** the artifact is loaded
- **THEN** all fields are available without re-running the cycle

### Requirement: Transfer effectiveness report
The component SHALL produce an effectiveness report containing: target metrics
before transfer (baseline on an empty prompt), metrics after the first round
(with transferred artifacts), and metrics after the final round. The report
SHALL answer whether the transfer accelerates cycle convergence.

#### Scenario: Report produced
- **WHEN** the transfer completes
- **THEN** the report contains three metric snapshots: baseline, after the first
  round, and after the final round

#### Scenario: Acceleration measured
- **WHEN** the target cycle with transfer converges in 3 rounds and without
  transfer in 5
- **THEN** the report records an acceleration of 2 rounds

### Requirement: Comparison with cold start
The component SHALL support an optional comparison mode that runs the target
cycle twice — with transfer and without — and compares the final metrics and
the number of rounds to stop. The mode SHALL be configurable and SHALL be
disabled by default.

#### Scenario: Cold start comparison enabled
- **WHEN** the config sets `compare_with_cold_start=true`
- **THEN** two cycles run on the target task: one with transfer and one without

#### Scenario: Comparison report
- **WHEN** both cycles complete
- **THEN** the report contains a comparison of metrics and round counts

#### Scenario: Cold start disabled by default
- **WHEN** the config does not set `compare_with_cold_start`
- **THEN** only the cycle with transfer runs

### Requirement: Isolation between tasks
The component SHALL isolate source and target artifacts: different `run_id`,
different storage paths, different thesis collections, and different prompt
version stores. The component SHALL NOT allow source artifacts to be overwritten
by target work.

#### Scenario: Different run ids
- **WHEN** the transfer is started
- **THEN** the source and target use different `run_id` values

#### Scenario: No overwrite
- **WHEN** the target cycle writes artifacts
- **THEN** the source artifacts remain unchanged

### Requirement: Configurable parameters
The component SHALL support configuration via config for `source_task_id`
(required), `target_task_id` (required), `source_artifacts_path` (required),
`target_artifacts_path` (required), `max_transfer_theses` (default unlimited),
`transfer_clusters` (default true), `compare_with_cold_start` (default false),
and `max_rounds_target` (default 5).

#### Scenario: Custom transfer budget
- **WHEN** the config sets `max_transfer_theses=30`
- **THEN** only 30 theses are transferred from the source

#### Scenario: Clusters not transferred
- **WHEN** the config sets `transfer_clusters=false`
- **THEN** cluster centroids are not transferred and the target collection
  starts with empty clustering

### Requirement: Logging of transfer
The component SHALL log all transfer operations: loading source artifacts,
transferring base layers, transferring theses, transferring centroids, starting
the target cycle, and final metrics. The log SHALL be append-only.

#### Scenario: Operations logged
- **WHEN** any transfer operation occurs
- **THEN** it is appended to the log with a timestamp and task identifier
