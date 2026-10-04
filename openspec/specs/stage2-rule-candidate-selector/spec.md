# stage2-rule-candidate-selector Specification

## Purpose

Load a thesis artifact with theses and clusters, filter and rank clusters by
frequency and precision, exclude clusters already in the prompt, select top-N
candidates with representative theses, and persist a reloadable candidate-list
artifact for downstream rule formulation.

## Requirements

### Requirement: Load thesis artifact
The component SHALL load a JSON artifact containing theses and clusters with
`metadata` (`run_id`, `prompt_version`), a `theses` dict, and a `clusters` dict.
The component SHALL fail with an error naming the first problematic cluster and
field when a cluster does not conform to the schema.

#### Scenario: Valid artifact loads
- **WHEN** the artifact contains clusters with fields `cluster_id`, `frequency`,
  `precision`, `positive_hits`, `negative_hits`, `in_prompt`
- **THEN** the clusters are loaded for candidate selection

#### Scenario: Missing cluster field
- **WHEN** a cluster is missing `precision`
- **THEN** loading fails with an error naming the `cluster_id` and the missing
  field

### Requirement: Filter by frequency threshold
The component SHALL filter out clusters whose `frequency` is below a configurable
`frequency_threshold`. The threshold SHALL be set in config with a default of 3.

#### Scenario: Cluster passes threshold
- **WHEN** a cluster has `frequency=5` and the threshold is 3
- **THEN** the cluster is included in the candidate list

#### Scenario: Cluster below threshold
- **WHEN** a cluster has `frequency=2` and the threshold is 3
- **THEN** the cluster is excluded from the candidate list

### Requirement: Exclude clusters already in prompt
The component SHALL exclude clusters with `in_prompt=true` from the candidate
list. The component MAY include them in a separate `already_in_prompt` list for
logging.

#### Scenario: Cluster already in prompt
- **WHEN** a cluster has `in_prompt=true`
- **THEN** the cluster is not included in the candidate list

### Requirement: Rank by precision
The component SHALL sort filtered clusters by `precision` descending. When
precision is equal the component SHALL use `frequency` descending as a
tie-breaker.

#### Scenario: Sorted by precision
- **WHEN** cluster A has `precision=0.9` and cluster B has `precision=0.8`
- **THEN** cluster A is ranked before cluster B

#### Scenario: Tie broken by frequency
- **WHEN** two clusters have `precision=0.85` but `frequency=10` and
  `frequency=5`
- **THEN** the cluster with `frequency=10` is ranked first

### Requirement: Select top-N candidates
The component SHALL select the top-N clusters from the sorted list. N SHALL be
configurable via config with a default of 5.

#### Scenario: Top-N selected
- **WHEN** 12 clusters remain after filtering and N=5
- **THEN** the first 5 clusters are selected

#### Scenario: Fewer candidates than N
- **WHEN** 3 clusters remain after filtering and N=5
- **THEN** all 3 clusters are selected

### Requirement: Candidate artifact
The component SHALL persist the candidate list as a JSON artifact whose filename
contains `run_id`, `prompt_version`, and `timestamp`. The artifact SHALL contain
for each candidate: `cluster_id`, `representative_theses` (up to
`max_representative_theses` theses with the highest `frequency`), `precision`,
`frequency`, `positive_hits`, `negative_hits`, `rank`.

#### Scenario: Artifact named consistently
- **WHEN** selection completes
- **THEN** the file is named
  `rule_candidates_{run_id}_{prompt_version}_{timestamp}.json`

#### Scenario: Candidate artifact reloaded
- **WHEN** the artifact is loaded
- **THEN** the candidate list is available without re-running selection

### Requirement: Logging of selection
The component SHALL log the selection process: how many clusters existed before
filtering, how many passed the frequency filter, how many were excluded as
`in_prompt=true`, and how many were selected into the top-N.

#### Scenario: Selection logged
- **WHEN** selection completes
- **THEN** the log contains `total_clusters`, `passed_frequency`,
  `excluded_in_prompt`, `selected_top_n`

### Requirement: Configurable parameters
The component SHALL support configuration via config for `frequency_threshold`
(default 3), `top_n` (default 5), and `max_representative_theses` (default 3).

#### Scenario: Custom threshold
- **WHEN** config sets `frequency_threshold=5`
- **THEN** filtering uses the threshold 5
