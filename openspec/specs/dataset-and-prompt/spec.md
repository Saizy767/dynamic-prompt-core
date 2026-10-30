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
`dev` and `holdout`. `holdout` SHALL be labeled and available for evaluation;
the optimization cycle drives accept/rollback decisions by holdout metrics
(see the stage3-cycle-orchestrator spec).

#### Scenario: Reproducible split
- **WHEN** the dataset is loaded twice with the same seed
- **THEN** the composition of `dev` and `holdout` is identical

#### Scenario: Holdout metrics drive cycle decisions
- **WHEN** a prompt version is evaluated by the cycle
- **THEN** metrics are computed on both `dev` and `holdout`, and the accept decision is driven by holdout metrics

### Requirement: Ambiguous examples separated
The component SHALL support separating a group of `ambiguous` examples — records
marked as doubtful. `ambiguous` examples SHALL be excluded from `dev` and
`holdout` and MUST NOT participate in Stage 1 runs.

#### Scenario: Ambiguous examples excluded
- **WHEN** the dataset contains records with `ambiguous=true`
- **THEN** those records appear in neither `dev` nor `holdout`

### Requirement: Prompt v0 for classification
The component SHALL provide a baseline system prompt v0 for binary
classification, composed of five layers: role, task, rules (an empty list in
the baseline — the zero-shot baseline), output contract, and fallback. The
prompt SHALL be stored as an artifact with a version and a content hash.
Rendering the prompt to text SHALL be performed by a simple templating helper
with no external dependencies.

#### Scenario: Prompt v0 rendered
- **WHEN** prompt v0 is requested
- **THEN** the returned prompt text is assembled from the five layers and carries a version and a hash

#### Scenario: Baseline v0 is zero-shot
- **WHEN** prompt v0 is assembled
- **THEN** its rules layer is an empty list, and the runner passes empty `ClassificationRules` to the `Classifier` as a zero-shot baseline

### Requirement: Fixed extraction prompt
The infrastructure SHALL provide a fixed system prompt for thesis extraction
as a packaged resource file (see the system-prompt-resources spec). This
prompt MUST NOT be optimized on Stage 1 and SHALL be used unchanged across all
optimization rounds.

#### Scenario: Extraction prompt available
- **WHEN** the runner requests the extraction prompt
- **THEN** the fixed prompt text is loaded from its resource file

### Requirement: Thesis extraction schema
The infrastructure extractor (`infrastructure/llm/extractor.py`) SHALL provide
a Pydantic schema (`ThesisExtraction`) for thesis extraction output: a list of
strings where each string is 2–6 words and the total number of theses is 3–5.
The schema SHALL be used by the client (`AsyncTask.extract_theses_detailed`)
for structured output.

#### Scenario: Schema validates extraction output
- **WHEN** the model returns JSON with a `theses` field that is a list of 4 strings of 4 words each
- **THEN** Pydantic validation passes

#### Scenario: Schema rejects too many theses
- **WHEN** the model returns JSON with 8 theses
- **THEN** Pydantic validation fails

### Requirement: No generative classification schema
The component SHALL NOT provide a Pydantic schema for generative
classification output: classification is performed through candidate scoring
(see the classifier spec), not through generative LLM calls, so no
`decision`/`confidence` response schema and no `AsyncTask.classify` method
exist.

#### Scenario: No classification response schema
- **WHEN** the codebase is inspected for a generative classification schema
- **THEN** no `decision`/`confidence` response model and no `classify` client method exist

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
