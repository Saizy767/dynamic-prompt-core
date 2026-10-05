# Spec Delta

## Purpose

Load the artifacts produced by a completed optimization cycle and render a
human- and machine-readable review of what changed between rounds. The observer
is read-only: it aggregates and explains cycle artifacts without taking
decisions or mutating state.

## ADDED Requirements

### Requirement: Load cycle artifacts
The component SHALL load the artifacts produced by the cycle: per-round
reports, decision artifacts, prompt-version records from the prompt store,
metrics artifacts, thesis dumps, and changed-decisions lists. The
component SHALL fail with an error naming the missing or corrupted artifact
when an expected file is not found or does not parse.

#### Scenario: All artifacts present
- **WHEN** the cycle is complete and all artifacts are present
- **THEN** the observer loads them without errors

#### Scenario: Missing artifact
- **WHEN** the decision artifact for a round is absent
- **THEN** loading fails with an error naming the missing file

#### Scenario: Corrupted artifact
- **WHEN** an artifact's JSON is corrupted
- **THEN** loading fails with an error naming the corrupted file

### Requirement: Summarize each round
The component SHALL produce a summary for each round containing: round number,
active version at round start, new composed version, decision (accept or
rollback), key dev metrics before and after, minority-class F1 change, count of
changed predictions, and rollback counter value.

#### Scenario: Round summary produced
- **WHEN** the observer processes round 3
- **THEN** the summary contains all listed fields

#### Scenario: Round skipped
- **WHEN** a round is absent from the artifacts
- **THEN** no summary is produced for it and the skip is logged

### Requirement: Show prompt evolution
The component SHALL produce a prompt-evolution overview: which rules were added,
which were removed, and which were preserved between versions. For each rule
the overview SHALL name the source cluster and the version in which it first
appeared.

#### Scenario: Rule added
- **WHEN** a rule appears in version v2 and is absent in v1
- **THEN** the overview shows the rule as added in v2

#### Scenario: Rule removed
- **WHEN** a rule was in v1 and is absent in v2
- **THEN** the overview shows the rule as removed in v2

#### Scenario: Rule preserved
- **WHEN** a rule is present in both v1 and v2
- **THEN** the overview shows the rule as preserved

### Requirement: Show rejected attempts
The component SHALL produce a rejected-attempts overview: which versions were
rolled back, which rules were formulated, and why they were rejected (meaning
distortion, limit violation, metric degradation). Each rejected attempt SHALL
be linked to its originating cluster.

#### Scenario: Rejected attempt shown
- **WHEN** version v2 was rolled back due to a macro-F1 decrease
- **THEN** the overview contains v2, the formulated rules, the rollback reason,
  and a reference to the source clusters

#### Scenario: Distortion rejection shown
- **WHEN** a rule was rejected on the distortion check
- **THEN** the overview contains the rule text, the cosine value, and the
  threshold

### Requirement: Show metric trends
The component SHALL produce a metric-trends overview by round: accuracy,
macro-F1, minority-class F1, and parse-failure count. The data SHALL be
presented in tabular form with one row per round. Trends SHALL NOT be
interpreted — only displayed.

#### Scenario: Metric trends shown
- **WHEN** the cycle is complete
- **THEN** the table contains one row per round with the metrics

#### Scenario: Missing metric
- **WHEN** a metric is absent for a round
- **THEN** the cell is marked `N/A` and the fact is logged

### Requirement: Show changed decisions
The component SHALL produce a changed-decisions overview: the list of example
ids whose `predicted_decision` changed between the active and new version, with
the direction of change (correct→incorrect, incorrect→correct) and a short
example text truncated to the configured length.

#### Scenario: Changed decisions listed
- **WHEN** 12 predictions changed between v1 and v2
- **THEN** the overview contains all 12 ids with direction and truncated text

#### Scenario: No changes
- **WHEN** no predictions changed
- **THEN** the overview explicitly reports that there are no changes

### Requirement: Show thesis store summary
The component SHALL produce a thesis-store summary at cycle end: total theses,
cluster count, the top-N clusters by frequency with precision and
representative theses, the count of clusters with `in_prompt=true`, and the
count of unassigned theses.

#### Scenario: Store summary produced
- **WHEN** the cycle is complete
- **THEN** the overview contains all listed counters

#### Scenario: Top clusters shown
- **WHEN** the thesis store has 20 clusters
- **THEN** the overview contains the top 10 by frequency

### Requirement: Final observation
The component SHALL produce a final observation containing: start version and
its metrics, final version and its metrics, absolute and relative improvement
per metric, stop reason, total rounds, and counts of accepted and rolled-back
versions.

#### Scenario: Final observation produced
- **WHEN** the cycle is complete
- **THEN** the final observation contains all listed fields

#### Scenario: No improvement
- **WHEN** metrics did not improve relative to the start version
- **THEN** the final observation explicitly reports the absence of improvement

### Requirement: Output formats
The component SHALL support at least two output formats: `markdown` (for
human-readable review) and `json` (for machine processing). The format SHALL be
configurable via config. The markdown output SHALL be suitable for pasting into
a README or issue.

#### Scenario: Markdown output
- **WHEN** config sets `output_format=markdown`
- **THEN** the report is written as a markdown file

#### Scenario: JSON output
- **WHEN** config sets `output_format=json`
- **THEN** the report is written as a JSON file

### Requirement: Observer is read-only
The component SHALL NOT mutate any cycle artifact, the active version, the
thesis store, or the prompt store. The observer MAY only read and write its own
report.

#### Scenario: No state mutation
- **WHEN** the observer produces a report
- **THEN** no cycle artifact is modified

#### Scenario: Only report written
- **WHEN** the observer finishes
- **THEN** only the report file is created

### Requirement: Configurable parameters
The component SHALL support configuration via config for `output_format`
(default `markdown`), `output_path` (default `observer_report.md`),
`max_example_text_length` (default 100), `top_clusters_count` (default 10), and
`include_holdout` (default true).

#### Scenario: Custom output path
- **WHEN** config sets `output_path=/tmp/report.md`
- **THEN** the report is written to that path

#### Scenario: Holdout excluded
- **WHEN** config sets `include_holdout=false`
- **THEN** holdout metrics are not included in the report

### Requirement: Logging of observer operations
The component SHALL log all observer operations: artifact loading, formation of
each section, and report writing. The log SHALL be append-only.

#### Scenario: Operations logged
- **WHEN** the observer forms a section
- **THEN** the operation is appended to the log with a timestamp
