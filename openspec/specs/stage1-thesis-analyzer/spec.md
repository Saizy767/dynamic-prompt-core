# Spec

## Purpose

Load a baseline-runner results artifact, build a deduplicated thesis bank with
embeddings and semantic clusters, track precision/frequency/demand/selected/in_prompt
counters per thesis and per cluster, and persist a reloadable thesis-bank dump
that feeds candidate selection for prompt rules.

## Requirements

### Requirement: Load run artifact
The component SHALL load the results artifact created by `stage1-baseline-runner`
and SHALL use only records with `classify_status=ok` and `extract_status=ok`.
The component SHALL fail with an error naming the first problematic field when a
record does not conform to the schema.

#### Scenario: Valid artifact loads
- **WHEN** the artifact contains records with `theses_norm` and `predicted_decision`
- **THEN** theses are extracted from all records with `extract_status=ok`

#### Scenario: Records with failed extraction skipped
- **WHEN** the artifact contains records with `extract_status=failed`
- **THEN** those records do not participate in building the thesis bank

### Requirement: Thesis bank
The component SHALL maintain a thesis bank in memory. Each thesis SHALL be keyed
by its normalized form. For each thesis the component SHALL store: `text_raw`
(last raw form), `text_norm` (normalized form), `embedding` (embedding vector),
`precision` (share of `positive_hits` over total occurrences), `frequency`
(total occurrences across all runs), `positive_hits` (occurrences in texts with
`true_label=1`), `negative_hits` (occurrences in texts with `true_label=0`),
`demand` (times the thesis was a rule candidate), `selected` (times the thesis
was chosen for the prompt), `in_prompt` (boolean: whether the thesis is in the
active prompt version), `cluster_id` (cluster identifier).

#### Scenario: New thesis added
- **WHEN** the thesis "клиент недоволен" is encountered for the first time in a run
- **THEN** a bank entry is created with `frequency=1`, and `positive_hits` or
  `negative_hits` is incremented depending on `true_label`

#### Scenario: Existing thesis updated
- **WHEN** the thesis "клиент недоволен" is already in the bank and is encountered again
- **THEN** `frequency` is incremented, the corresponding counter
  (`positive_hits` or `negative_hits`) is incremented, and `text_raw` is updated
  to the latest raw form

#### Scenario: Precision computed
- **WHEN** a thesis has `positive_hits=8` and `negative_hits=2`
- **THEN** `precision` equals 0.8

### Requirement: Deduplication by exact normalized form
The component SHALL deduplicate theses by exact match of the normalized form.
Two theses with the same `text_norm` SHALL be treated as one thesis. The
component MUST NOT create separate entries for morphological variants of the same
thesis.

#### Scenario: Morphological variants merged
- **WHEN** a run contains "клиент отменить заявка" and "клиент отменить заявку"
- **THEN** both forms reduce to the same normalized form and land in one bank entry

### Requirement: Embeddings for normalized theses
The component SHALL compute embeddings for normalized theses (`text_norm`). The
embedding SHALL be used for clustering. The component SHALL support selecting the
embedding model via config.

#### Scenario: Embedding computed once
- **WHEN** a thesis is already in the bank with an embedding
- **THEN** the embedding is not recomputed

#### Scenario: Embedding stored
- **WHEN** a new thesis is added to the bank
- **THEN** its embedding is computed and stored in the entry

### Requirement: Clustering by cosine similarity
The component SHALL cluster theses by cosine similarity of their embeddings. For
each new thesis the component SHALL search for the nearest existing cluster by
cosine similarity to its centroid. If the maximum cosine exceeds the threshold,
the thesis is added to that cluster. The component SHALL support a configurable
cosine threshold.

#### Scenario: Thesis assigned to existing cluster
- **WHEN** the cosine between the thesis embedding and a cluster centroid exceeds the threshold
- **THEN** the thesis is added to that cluster

#### Scenario: New cluster created
- **WHEN** no existing cluster exceeds the cosine threshold
- **THEN** a new cluster is created and the thesis becomes its first member

### Requirement: Manhattan distance in grey zone
The component SHALL support Manhattan distance as a secondary criterion in the
"grey zone" — when cosine similarity is between a lower and an upper threshold.
If cosine is below the lower threshold, the thesis is not attached to the cluster
by cosine. If cosine is above the upper threshold, the thesis is attached
directly. In the grey zone the decision SHALL be made by Manhattan distance
against a configurable Manhattan threshold.

#### Scenario: Grey zone resolved by Manhattan
- **WHEN** the cosine between a thesis and a cluster is 0.68, the lower threshold is 0.6, and the upper threshold is 0.75
- **THEN** Manhattan distance is checked, and if it is below the Manhattan threshold, the thesis is attached to the cluster

#### Scenario: Below lower threshold
- **WHEN** the cosine is 0.55 and the lower threshold is 0.6
- **THEN** the thesis is not attached to that cluster by cosine

### Requirement: Cluster limit
The component SHALL limit the maximum number of clusters. The value SHALL be
configurable. When the limit is reached, a new thesis that matches no cluster
SHALL be attached to the nearest cluster regardless of threshold, or logged as
unassigned.

#### Scenario: Limit not exceeded
- **WHEN** the number of clusters is below the limit and a thesis matches no cluster
- **THEN** a new cluster is created

#### Scenario: Limit reached
- **WHEN** the number of clusters equals the limit and a thesis matches no cluster
- **THEN** the thesis is attached to the nearest cluster and the fact is logged

### Requirement: Cluster counters
The component SHALL aggregate cluster counters from its member theses:
`precision` (frequency-weighted), `frequency` (sum), `positive_hits` (sum),
`negative_hits` (sum), `demand` (sum), `selected` (sum). The component SHALL
update cluster counters whenever a thesis counter is updated.

#### Scenario: Cluster counters updated
- **WHEN** a thesis in a cluster receives a new occurrence
- **THEN** cluster counters are recomputed from the updated thesis counters

#### Scenario: Cluster precision weighted
- **WHEN** a cluster contains a thesis with `precision=1.0` `frequency=10` and a thesis with `precision=0.5` `frequency=2`
- **THEN** the cluster precision is weighted by frequency

### Requirement: Thesis bank dump
The component SHALL persist the thesis bank and cluster state to a JSON file
after each update. The filename SHALL contain `run_id`, `prompt_version`, and
`timestamp`. The artifact SHALL be loadable to resume work without recomputing
embeddings or clusters.

#### Scenario: Dump written
- **WHEN** thesis analysis completes
- **THEN** a file named `thesis_bank_{run_id}_{prompt_version}_{timestamp}.json` is written containing all bank and cluster entries

#### Scenario: Dump reloaded
- **WHEN** the artifact is loaded
- **THEN** the thesis bank and clusters are restored to their original state

### Requirement: Normalization is already applied
The component SHALL use `theses_norm` from the results artifact as the normalized
form. The component MUST NOT re-normalize theses. The component SHALL use
`theses_raw` to preserve the raw form.

#### Scenario: Norm used directly
- **WHEN** a thesis is loaded from the artifact
- **THEN** `text_norm` is taken from `theses_norm` and `text_raw` from `theses_raw`

### Requirement: Stable cluster centroids
The component SHALL fix a cluster centroid at creation and update it only when
new theses are added. The component MUST NOT recompute all cluster centroids on
every update.

#### Scenario: Centroid fixed
- **WHEN** a cluster is created with a single thesis
- **THEN** the centroid equals that thesis's embedding and is updated only when new members are added
