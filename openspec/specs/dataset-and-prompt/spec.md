# Spec

## Purpose

Load and prepare a binary-classification dataset with a fixed dev/holdout split,
and provide the baseline classification prompt v0, a fixed extraction prompt,
and the Pydantic schemas that the Stage 1 runner consumes for structured output.

## Requirements

### Requirement: Load dataset from file
The component SHALL load a dataset from a file (jsonl or parquet) with required
fields `id`, `text`, `label`, where `label` is `0` or `1`. The component SHALL
verify the presence of the required fields and the validity of `label` values,
and SHALL fail with an error naming the first offending record when validation
does not pass.

#### Scenario: Valid dataset loads
- **WHEN** the file contains records with fields `id`, `text`, `label`, and every `label` is `0` or `1`
- **THEN** the dataset loads into memory and the record count equals the number of lines in the file

#### Scenario: Missing required field
- **WHEN** a record is missing `text` or `label`
- **THEN** loading fails with an error naming the record's `id` and the missing field

#### Scenario: Invalid label value
- **WHEN** `label` takes a value other than `0` or `1`
- **THEN** loading fails with an error naming the record's `id` and the invalid value

### Requirement: Fixed train/holdout split
The component SHALL split the dataset into `dev` and `holdout` using a fixed
seed. The split SHALL be reproducible: the same dataset and seed yield the same
`dev` and `holdout`. `holdout` SHALL be labeled and available for logging, but
its metrics MUST NOT participate in accept or reject decisions for prompt
versions on Stage 1.

#### Scenario: Reproducible split
- **WHEN** the dataset is loaded twice with the same seed
- **THEN** the composition of `dev` and `holdout` is identical

#### Scenario: Holdout metrics logged but not used
- **WHEN** a prompt version is run
- **THEN** metrics are computed on both `dev` and `holdout`, but the accept decision is based only on `dev` metrics

### Requirement: Ambiguous examples separated
The component SHALL support separating a group of `ambiguous` examples — records
marked as doubtful. `ambiguous` examples SHALL be excluded from `dev` and
`holdout` and MUST NOT participate in Stage 1 runs.

#### Scenario: Ambiguous examples excluded
- **WHEN** the dataset contains records with `ambiguous=true`
- **THEN** those records appear in neither `dev` nor `holdout`

### Requirement: Prompt v0 for classification
The component SHALL provide a baseline system prompt v0 for binary
classification, composed of five layers: role, task, rules (3–5 items), output
contract, and fallback. The prompt SHALL be stored as an artifact with a version
and a content hash. Rendering the prompt to text SHALL be performed by a simple
templating helper with no external dependencies.

#### Scenario: Prompt v0 rendered
- **WHEN** prompt v0 is requested
- **THEN** the returned prompt text is assembled from the five layers and carries a version and a hash

#### Scenario: Rules count within limit
- **WHEN** prompt v0 is assembled
- **THEN** the number of rules is between 3 and 5, otherwise assembly fails with an error

### Requirement: Fixed extraction prompt
The component SHALL provide a fixed system prompt for thesis extraction. This
prompt MUST NOT be optimized on Stage 1 and SHALL be used unchanged across all
optimization rounds.

#### Scenario: Extraction prompt available
- **WHEN** the runner requests the extraction prompt
- **THEN** a fixed prompt text is returned with a version and a hash

### Requirement: Thesis extraction schema
The component SHALL provide a Pydantic schema for thesis extraction output: a
list of strings where each string is 2–6 words and the total number of theses is
3–5. The schema SHALL be used by the client (`AsyncTask.extract_theses`) for
structured output.

#### Scenario: Schema validates extraction output
- **WHEN** the model returns JSON with a `theses` field that is a list of 4 strings of 4 words each
- **THEN** Pydantic validation passes

#### Scenario: Schema rejects too many theses
- **WHEN** the model returns JSON with 8 theses
- **THEN** Pydantic validation fails

### Requirement: Classification schema
The component SHALL provide a Pydantic schema for classification output:
`decision` (`0` or `1`) and `confidence` (0–100). The schema SHALL be used by
the client (`AsyncTask.classify`) for structured output.

#### Scenario: Schema validates classification output
- **WHEN** the model returns JSON `{"decision": 1, "confidence": 85}`
- **THEN** Pydantic validation passes

#### Scenario: Schema rejects invalid decision
- **WHEN** the model returns `decision` equal to `2`
- **THEN** Pydantic validation fails

### Requirement: Dataset artifact persisted
The component SHALL persist the prepared dataset (with the dev, holdout, and
ambiguous split) as an artifact in jsonl or parquet format with fields `id`,
`text`, `label`, `split`, `notes`. The artifact SHALL be reproducible from the
source file and seed.

#### Scenario: Artifact written
- **WHEN** the dataset is prepared
- **THEN** the artifact is written to a file whose name includes the seed and a timestamp

#### Scenario: Artifact reloaded
- **WHEN** the artifact is loaded
- **THEN** the dev, holdout, and ambiguous split matches the original
