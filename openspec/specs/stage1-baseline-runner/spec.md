# Spec

## Purpose

Run the prepared dataset (dev or holdout) through the model with a fixed thesis
extraction prompt and the current classification prompt version, collect parsed
output, raw responses, latencies, and parse statuses into a jsonl results
artifact with checkpointing and resumption, feeding the metrics and
thesis-analysis stages.

## Requirements

### Requirement: Active prompt rules extraction by runner
The runner SHALL extract classification rules from the active
`PromptArtifact.layers.rules`. When the active prompt is the default
classification prompt, the runner SHALL use its default rules. When a
non-default active prompt has missing or `None` layers, the runner SHALL raise
a clear, actionable error rather than silently substituting default rules. The
runner SHALL NOT reload the default prompt to recover rules for a non-default
active prompt. The runner SHALL NOT parse rendered prompt text to recover
rules when the structured `layers.rules` data is available.

#### Scenario: Default prompt rules used
- **WHEN** the runner uses the default classification prompt
- **THEN** the default prompt's `layers.rules` are passed to the scorer

#### Scenario: Non-default prompt rules used
- **WHEN** the runner uses a non-default active prompt with valid layers
- **THEN** that prompt's `layers.rules` are passed to the scorer

#### Scenario: Malformed non-default prompt raises
- **WHEN** the runner uses a non-default active prompt with missing or `None` layers
- **THEN** the runner raises a clear error rather than silently substituting default rules

#### Scenario: No default prompt reload for non-default active prompt
- **WHEN** the runner uses a non-default active prompt
- **THEN** it does not reload the default prompt to recover the rule set

### Requirement: Run dataset through the model
The component SHALL run each dataset example through two steps: thesis
extraction using `extract_theses_detailed` with the fixed extraction prompt,
and classification using the candidate-scoring flow
(`CandidateScorer.score` → `ClassificationPolicy.classify`). The component
SHALL NOT use `classify_detailed` or any generative classification method. The
component SHALL use `AsyncTask` from `llm-transport-client` for thesis
extraction and `CandidateScorer` for classification, and MUST NOT contact the
HTTP server directly. The candidate set SHALL be supplied to the runner at
construction time; the runner SHALL NOT infer or generate candidates. The
runner SHALL extract the active classification rules from the classification
prompt artifact's `layers.rules` and pass them to every
`CandidateScorer.score()` invocation. When `skip_extraction` is set to true,
the component SHALL skip thesis extraction entirely and only perform
candidate-scoring classification; `theses_raw` and `theses_norm` SHALL be
empty lists and `extract_status` SHALL be `skipped`.

#### Scenario: Single example processed
- **WHEN** the runner receives an example with `id`, `text`, `true_label`
- **THEN** thesis extraction and candidate-scoring classification are performed and the result contains `predicted_decision`, `selected_candidate`, `judgment_scores`, `theses_raw`, `theses_norm`

#### Scenario: Both steps succeed
- **WHEN** both extraction and candidate scoring return valid results
- **THEN** the result has all fields filled and `extract_status` and `classify_status` are `ok`

#### Scenario: Classification uses candidate scoring
- **WHEN** the runner classifies an example
- **THEN** it invokes `CandidateScorer.score(text, candidates, rules)` with rules from the active prompt artifact's `layers.rules`, followed by `ClassificationPolicy.classify(judgments)`, and does not call `classify_detailed`

#### Scenario: Candidate set supplied at construction
- **WHEN** the runner is constructed
- **THEN** it receives the candidate set and does not infer or generate candidates at runtime

#### Scenario: Skip extraction mode
- **WHEN** the runner is constructed with `skip_extraction=True`
- **THEN** thesis extraction is not called, `theses_raw` and `theses_norm` are empty lists, `extract_status` is `skipped`, and only candidate-scoring classification is performed

### Requirement: Skip extraction flag
The component SHALL accept a `skip_extraction` flag at construction time,
defaulting to false. When true, the component SHALL NOT make any
thesis-extraction LLM calls and SHALL only run candidate-scoring
classification for each example. The results artifact SHALL still contain all
fields from the result schema, with `theses_raw` and `theses_norm` as empty
lists and `extract_status` set to `skipped`.

#### Scenario: Skip extraction defaults to false
- **WHEN** the runner is constructed without `skip_extraction`
- **THEN** thesis extraction is performed as normal

#### Scenario: Skip extraction avoids LLM calls
- **WHEN** the runner is constructed with `skip_extraction=True` and processes N examples
- **THEN** zero thesis-extraction LLM calls are made and N classification-scoring calls are made

#### Scenario: Skip extraction result schema
- **WHEN** `skip_extraction=True` and a result is saved
- **THEN** the result contains `extract_status=skipped`, `theses_raw=[]`, `theses_norm=[]`, `extract_latency_ms=0`, and `raw_extract=None`

### Requirement: Result schema
The component SHALL save results with the fields: `id`, `text`, `true_label`,
`predicted_decision` (integer, populated from the selected candidate's value),
`selected_candidate` (string, the selected `Candidate.value`),
`judgment_scores` (mapping of candidate values to scores), `theses_raw` (list
of strings as returned by the model), `theses_norm` (list of strings after
normalization), `extract_status`, `classify_status`, `extract_latency_ms`,
`classify_latency_ms`, `raw_extract`. The status fields SHALL distinguish four
states: `ok`, `repaired`, `failed`, `not_attempted`. The fields `confidence`
and `raw_classify` SHALL NOT be present because generative classification no
longer exists. `predicted_decision` is preserved for metrics compatibility and
populated from the candidate-scoring result.

#### Scenario: Result contains classification from candidate scoring
- **WHEN** candidate scoring succeeds and the policy selects a candidate
- **THEN** the result has `classify_status=ok`, `predicted_decision` set to the integer value of the selected candidate, `selected_candidate` set to the selected candidate's value, and `judgment_scores` populated

#### Scenario: Result contains both parse statuses
- **WHEN** extraction succeeds and candidate scoring fails
- **THEN** the result has `extract_status=ok`, `classify_status=failed`, and `predicted_decision` is empty

#### Scenario: Raw extraction response preserved
- **WHEN** a result is saved
- **THEN** the raw model response for extraction is stored in `raw_extract`

#### Scenario: No generative classification fields
- **WHEN** a result is saved
- **THEN** the result does not contain `confidence` or `raw_classify`

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
