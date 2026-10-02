# Spec

## Purpose

Load a stage1-baseline-runner results artifact, compute classification quality
metrics (accuracy, per-class / macro / weighted F1, confusion matrix,
minority-class F1, prediction distribution and bias, group-by-length breakdown,
parse-failure counts), persist a json metrics artifact tied to the prompt version
and run, and compare two prompt-version metrics on the same split.

## Requirements

### Requirement: Load run artifact
The component SHALL load the results artifact produced by `stage1-baseline-runner`
and SHALL verify that every record carries the required fields: `id`, `text`,
`true_label`, `predicted_decision`, `confidence`, `classify_status`,
`theses_norm`. The component SHALL fail with an error naming the first offending
record's `id` and the missing field when a record does not conform to the schema.

#### Scenario: Valid artifact loads
- **WHEN** the artifact contains records with all required fields
- **THEN** metrics are computed over all records with `classify_status=ok`

#### Scenario: Missing field in record
- **WHEN** a record is missing `predicted_decision`
- **THEN** loading fails with an error naming the record's `id` and the missing field

### Requirement: Accuracy and F1
The component SHALL compute accuracy, precision, recall, and F1 for each class
separately, plus macro-F1 and weighted-F1. The component SHALL compute accuracy
only over records with `classify_status=ok` and SHALL separately report the share
of records with `classify_status=failed`.

#### Scenario: Accuracy computed
- **WHEN** the artifact has 180 records with `classify_status=ok`, of which 150 are predicted correctly
- **THEN** accuracy equals 150/180 and the share of `classify_status=failed` is reported separately

#### Scenario: Per-class metrics
- **WHEN** metrics are computed
- **THEN** precision, recall, and F1 are returned for class 0 and for class 1 separately

### Requirement: Confusion matrix
The component SHALL build a confusion matrix over records with
`classify_status=ok` in a 2×2 form: true positives, true negatives, false
positives, false negatives. The component SHALL store the matrix in the metrics
artifact.

#### Scenario: Confusion matrix built
- **WHEN** there are 100 records with `true_label=1` and 80 with `true_label=0`
- **THEN** the matrix contains four values TP, TN, FP, FN whose sum equals 180

### Requirement: Minority class F1
The component SHALL determine the minority class (the class with fewer examples
in the dev split) and SHALL report the minority class's F1 separately from
macro-F1. The component MUST NOT use accuracy as the sole metric for accepting a
prompt version under class imbalance.

#### Scenario: Minority class identified
- **WHEN** the dev split has 150 examples of class 1 and 50 of class 0
- **THEN** class 0 is the minority class and its F1 is reported separately

### Requirement: Distribution of predictions
The component SHALL compute the distribution of predictions: how many times the
model predicted 0 and how many times 1, plus each class's share of the
predictions. The component SHALL compare this distribution with the
`true_label` distribution and SHALL flag a bias when the gap exceeds a configured
threshold.

#### Scenario: Prediction distribution computed
- **WHEN** the model predicted 0 in 30 cases and 1 in 150 cases
- **THEN** the prediction distribution is 0 — 16.7%, 1 — 83.3%

#### Scenario: Bias detected
- **WHEN** the true distribution is 50/50 and the predicted distribution is 10/90
- **THEN** a bias toward class 1 is flagged and the bias metric is stored in the artifact

### Requirement: Group breakdown
The component SHALL compute metrics by text-length group: short (fewer than 10
words), medium (10–30 words), long (more than 30 words). Groups SHALL be defined
in config and MAY be overridden. The component SHALL report accuracy and F1 for
each group separately.

#### Scenario: Group metrics computed
- **WHEN** the artifact contains records with varying text length
- **THEN** metrics are computed for each group separately and the result contains the breakdown

### Requirement: Parse failure metrics
The component SHALL count records with `classify_status=failed`,
`classify_status=repaired`, and `classify_status=not_attempted` and SHALL report
them separately from classification errors. A parse failure MUST NOT be counted
as a classification error when computing accuracy.

#### Scenario: Parse failures reported
- **WHEN** the artifact has 10 records with `classify_status=failed`
- **THEN** those 10 records are excluded from accuracy and their count is reported separately

### Requirement: Metrics artifact
The component SHALL persist metrics as an artifact whose filename contains
`run_id`, `prompt_version`, `split`, and `timestamp`. The format SHALL be json.
The artifact SHALL contain every metric: accuracy, precision, recall, F1 (macro,
weighted, per-class, minority-class), confusion matrix, prediction distribution,
group breakdown, and parse-failure counts.

#### Scenario: Artifact named consistently
- **WHEN** metrics are computed
- **THEN** the file is named `metrics_{run_id}_{prompt_version}_{split}_{timestamp}.json`

#### Scenario: Artifact reloaded
- **WHEN** the artifact is loaded
- **THEN** all metrics are available without recomputation

### Requirement: Comparison between versions
The component SHALL support comparing the metrics of two prompt versions on the
same split. The component SHALL return the delta for accuracy, F1, and
minority-class F1, plus the list of record `id`s whose `predicted_decision`
changed between the versions.

#### Scenario: Versions compared
- **WHEN** the metrics of v0 and v1 on dev are compared
- **THEN** the delta for each metric is returned alongside the list of `id`s whose prediction changed

#### Scenario: Changed decisions listed
- **WHEN** v1 fixed 5 examples and broke 3
- **THEN** the list of changed records contains 8 `id`s with the direction of each change

### Requirement: Metrics are computed for dev only
The component SHALL compute metrics for dev and holdout separately. Holdout
metrics SHALL be logged but MUST NOT participate in accepting or rolling back a
prompt version on Stage 1.

#### Scenario: Dev metrics used for decisions
- **WHEN** a prompt version is evaluated
- **THEN** the decision is based on dev metrics, while holdout metrics are stored separately for later analysis
