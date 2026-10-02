# Spec

## Purpose

Run the prepared dataset (dev or holdout) through the model with a fixed thesis
extraction prompt and the current classification prompt version, collect parsed
output, raw responses, latencies, and parse statuses into a jsonl results
artifact with checkpointing and resumption, feeding the metrics and
thesis-analysis stages.

## Requirements

### Requirement: Run dataset through the model
The component SHALL run each dataset example through two calls:
`extract_theses` with the fixed extraction prompt and `classify` with the current
classification prompt version. The component SHALL use `AsyncTask` from
`llm-transport-client` and MUST NOT contact the HTTP server directly.

#### Scenario: Single example processed
- **WHEN** the runner receives an example with `id`, `text`, `true_label`
- **THEN** two model calls are performed and the result contains `predicted_decision`, `confidence`, `theses_raw`, `theses_norm`

#### Scenario: Both calls succeed
- **WHEN** both calls return valid JSON matching their schemas
- **THEN** the result has all fields filled and `extract_status` and `classify_status` are `ok`

### Requirement: Result schema
The component SHALL save results with the fields: `id`, `text`, `true_label`,
`predicted_decision`, `confidence`, `theses_raw` (list of strings as returned by
the model), `theses_norm` (list of strings after normalization), `extract_status`,
`classify_status`, `extract_latency_ms`, `classify_latency_ms`, `raw_extract`,
`raw_classify`. The status fields SHALL distinguish four states: `ok`,
`repaired`, `failed`, `not_attempted`.

#### Scenario: Result contains both parse statuses
- **WHEN** extraction succeeds and classification returns invalid JSON
- **THEN** the result has `extract_status=ok`, `classify_status=failed`, and `predicted_decision` is empty

#### Scenario: Raw responses preserved
- **WHEN** a result is saved
- **THEN** the raw model responses for both calls are stored in `raw_extract` and `raw_classify`

### Requirement: Normalization of theses
The component SHALL normalize theses before saving them in `theses_norm`:
lemmatization (pymorphy3 for Russian, spaCy for English), lower-casing, and
whitespace collapsing. Negation particles (`не`, `нет`, `без`, `никогда`) MUST
NOT be removed or altered.

#### Scenario: Lemma applied
- **WHEN** a thesis is «Клиент отменил заявку»
- **THEN** `theses_norm` contains «клиент отменить заявка»

#### Scenario: Negation preserved
- **WHEN** a thesis is «Клиент не отменил заявку»
- **THEN** `theses_norm` contains «клиент не отменить заявка»

#### Scenario: Raw preserved
- **WHEN** a thesis is normalized
- **THEN** the original form is preserved in `theses_raw` unchanged

### Requirement: Progress and checkpoints
The component SHALL support saving progress to a file after processing each
example (or every N examples) so a run can resume from where it stopped. The
checkpoint format SHALL allow determining which examples are already processed
and MUST NOT require reprocessing successful examples.

#### Scenario: Resume after interruption
- **WHEN** a run is interrupted after processing 50 of 200 examples
- **THEN** a subsequent run processes only the remaining 150

#### Scenario: Checkpoint file written
- **WHEN** an example is processed
- **THEN** its result is written to the checkpoint file and the file is not corrupted by an abnormal termination

### Requirement: Run artifact
The component SHALL save the full run result as an artifact whose filename
contains `run_id`, `prompt_version`, `split` (`dev` or `holdout`), and
`timestamp`. The format SHALL be jsonl. The artifact SHALL be loadable by the
next stage (metrics, analyzer) without additional post-processing.

#### Scenario: Artifact named consistently
- **WHEN** a run completes
- **THEN** the file is named `results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl`

#### Scenario: Artifact reloaded
- **WHEN** an artifact is loaded
- **THEN** each record contains all fields from the result schema

### Requirement: Both splits are runnable
The component SHALL support running both `dev` and `holdout`. The split SHALL be
taken from the artifact prepared by `dataset-and-prompt` and MUST NOT be
recomputed.

#### Scenario: Dev run
- **WHEN** the runner is started with `split=dev`
- **THEN** only examples with `split=dev` are processed

#### Scenario: Holdout run
- **WHEN** the runner is started with `split=holdout`
- **THEN** only examples with `split=holdout` are processed

### Requirement: Batch execution
The component SHALL support batch execution through `AsyncTask` batch methods
with a configurable concurrency level. The default concurrency SHALL be sane for
local llama.cpp (no more than 8) and MAY be overridden via config.

#### Scenario: Concurrency honored
- **WHEN** the runner is started with `concurrency=4`
- **THEN** no more than 4 requests execute simultaneously

#### Scenario: Order of results preserved
- **WHEN** a batch completes
- **THEN** the order of results matches the order of input examples

### Requirement: Logging requests
The component SHALL pass `log_path` to `AsyncTask` so all requests are logged to
jsonl. The log filename SHALL be tied to `run_id` so a run can be correlated with
its log.

#### Scenario: Request log written
- **WHEN** a run completes
- **THEN** a request log file exists containing one entry per call

### Requirement: Failure handling
The component SHALL continue the run when an individual example fails (parse
failure, timeout, HTTP error) and record the error in the corresponding result
field. The component MUST NOT abort the entire run because of a single error.

#### Scenario: Single failure does not stop run
- **WHEN** the model returns invalid JSON for the example with `id=42`
- **THEN** the record with `id=42` is saved with `classify_status=failed` and the run continues

#### Scenario: Summary at the end
- **WHEN** a run completes
- **THEN** a summary is printed: count of `ok`, count of `parse_status=failed`, count of network errors, count of timeouts
