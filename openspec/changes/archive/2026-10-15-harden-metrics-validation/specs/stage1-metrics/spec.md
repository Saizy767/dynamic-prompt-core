# Spec Delta

## ADDED Requirements

### Requirement: Label value validation
The component SHALL validate that `true_label` is an integer in `{0, 1}` for
every record. The component SHALL validate that `predicted_decision` is an
integer in `{0, 1}` for every record with `classify_status=ok`. The component
SHALL raise an error naming the offending record `id`, the field name, and the
invalid value when a label is out of range or non-integer. The component SHALL
perform this validation during artifact loading, before any metric is computed.

#### Scenario: Invalid true_label rejected
- **WHEN** a record has `true_label=2`
- **THEN** loading fails with an error naming the record's `id`, the field `true_label`, and the value `2`

#### Scenario: Non-integer predicted_decision rejected
- **WHEN** a record with `classify_status=ok` has `predicted_decision="yes"`
- **THEN** loading fails with an error naming the record's `id`, the field `predicted_decision`, and the value `"yes"`

#### Scenario: Failed record with non-integer prediction is accepted
- **WHEN** a record with `classify_status=failed` has `predicted_decision=null`
- **THEN** loading succeeds because label validation applies only to `ok` records

### Requirement: Duplicate ID detection
The component SHALL detect when two or more records share the same `id` value
and SHALL raise an error naming the duplicated `id`. The component SHALL perform
this detection during artifact loading, before any metric is computed.

#### Scenario: Duplicate IDs rejected
- **WHEN** two records have `id=42`
- **THEN** loading fails with an error naming the duplicated `id` value `42`

#### Scenario: Unique IDs accepted
- **WHEN** all records have distinct `id` values
- **THEN** loading succeeds

### Requirement: Prediction presence for ok records
The component SHALL verify that every record with `classify_status=ok` carries a
non-null `predicted_decision`. The component SHALL raise an error naming the
record `id` when an `ok` record has a missing or `None` prediction. The
component SHALL perform this check during artifact loading.

#### Scenario: Ok record missing prediction rejected
- **WHEN** a record with `classify_status=ok` has `predicted_decision=null`
- **THEN** loading fails with an error naming the record's `id`

#### Scenario: Failed record missing prediction accepted
- **WHEN** a record with `classify_status=failed` has `predicted_decision=null`
- **THEN** loading succeeds

### Requirement: Empty and all-failed set handling
The component SHALL detect when no records with `classify_status=ok` exist. In
that case the component SHALL set `metrics_available` to `false` and SHALL
return `null` for `accuracy`, `accuracy_over_total`, `precision`, `recall`,
`f1`, `confusion_matrix`, `prediction_distribution`, `true_distribution`,
`bias`, and `groups`. The component SHALL still write the metrics artifact with
`counts` populated. The component SHALL NOT return `0.0` for any metric when
the `ok` set is empty, because `0.0` is indistinguishable from a model that
misclassifies every example.

#### Scenario: All records failed
- **WHEN** all records have `classify_status=failed`
- **THEN** `metrics_available` is `false`, all metric fields are `null`, and `counts` are populated

#### Scenario: Empty artifact
- **WHEN** the artifact contains zero records
- **THEN** `metrics_available` is `false`, all metric fields are `null`, and `counts.total` is `0`

#### Scenario: Mixed ok and failed records produce metrics
- **WHEN** the artifact has at least one record with `classify_status=ok`
- **THEN** `metrics_available` is `true` and all metric fields are populated with numeric values

### Requirement: Explicit metric denominator
The component SHALL report `accuracy_over_total` computed as correct predictions
divided by the total number of records (including failed), alongside the
existing `accuracy` computed over `ok` records only. The component SHALL include
a `denominator` field recording `ok_count` and `total_count`. The component
SHALL set `denominator_excludes_failures` to `true` when any record has
`classify_status` other than `ok`, and to `false` otherwise. The component
SHALL NOT silently exclude failed records from the denominator: the exclusion
SHALL be visible through `denominator_excludes_failures`, `denominator`, and
`accuracy_over_total`.

#### Scenario: Denominator reported with failures
- **WHEN** the artifact has 180 `ok` records and 20 `failed` records, with 150 correct predictions
- **THEN** `accuracy` is 150/180, `accuracy_over_total` is 150/200, `denominator` is `{ok_count: 180, total_count: 200}`, and `denominator_excludes_failures` is `true`

#### Scenario: Denominator reported without failures
- **WHEN** all 200 records have `classify_status=ok` with 150 correct predictions
- **THEN** `accuracy` is 150/200, `accuracy_over_total` is 150/200, `denominator` is `{ok_count: 200, total_count: 200}`, and `denominator_excludes_failures` is `false`

#### Scenario: Accuracy over total with all failed
- **WHEN** all records have `classify_status=failed`
- **THEN** `accuracy_over_total` is `null` (no correct predictions to report) and `denominator_excludes_failures` is `true`

### Requirement: Precision recall F1 correctness
The component SHALL compute per-class precision, recall, and F1 using the
standard definitions: precision = TP / (TP + FP), recall = TP / (TP + FN),
F1 = 2 * precision * recall / (precision + recall). The component SHALL compute
macro-F1 as the unweighted mean of per-class F1 scores. The component SHALL
compute weighted-F1 as the support-weighted mean of per-class F1 scores. The
component SHALL use `zero_division=0` convention: a class with no predicted or
no actual examples yields a precision, recall, and F1 of `0.0` for that class.
The component SHALL be verified by unit tests that cross-check these values
against hand-computed expectations for balanced, imbalanced, and
single-class-predicted scenarios.

#### Scenario: Balanced binary classification
- **WHEN** there are 50 true-positive, 50 true-negative, 0 false-positive, and 0 false-negative
- **THEN** precision, recall, and F1 are `1.0` for both classes, macro-F1 is `1.0`, and weighted-F1 is `1.0`

#### Scenario: Single class predicted
- **WHEN** the model predicts class 1 for all examples but half are class 0
- **THEN** class 0 has precision `0.0`, recall `0.0`, F1 `0.0`; class 1 has precision `0.5`, recall `1.0`, F1 `2/3`

#### Scenario: Imbalanced support
- **WHEN** class 0 has 180 examples and class 1 has 20 examples
- **THEN** weighted-F1 weights class 0 by 180 and class 1 by 20, and macro-F1 weights both equally

## MODIFIED Requirements

### Requirement: Accuracy and F1
The component SHALL compute accuracy, precision, recall, and F1 for each class
separately, plus macro-F1 and weighted-F1. The component SHALL compute accuracy
only over records with `classify_status=ok` and SHALL separately report the
share of records with `classify_status=failed`. When no records with
`classify_status=ok` exist, the component SHALL return `null` for accuracy,
precision, recall, and F1 (not `0.0`) and SHALL set `metrics_available` to
`false`. The component SHALL report `accuracy_over_total` as correct predictions
divided by total records alongside the existing `accuracy`.

#### Scenario: Accuracy computed
- **WHEN** the artifact has 180 records with `classify_status=ok`, of which 150 are predicted correctly
- **THEN** accuracy equals 150/180 and the share of `classify_status=failed` is reported separately

#### Scenario: Per-class metrics
- **WHEN** metrics are computed and at least one `ok` record exists
- **THEN** precision, recall, and F1 are returned for class 0 and for class 1 separately

#### Scenario: No ok records returns null metrics
- **WHEN** no records have `classify_status=ok`
- **THEN** accuracy, precision, recall, and F1 are `null` and `metrics_available` is `false`
