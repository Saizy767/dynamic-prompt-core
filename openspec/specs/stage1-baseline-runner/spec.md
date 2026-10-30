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
`PromptArtifact` and convert them to `ClassificationRules`. When the active
prompt has valid `layers.rules`, the runner SHALL pass the converted
`ClassificationRules` to the `Classifier`. When `layers` is `None`, the runner
SHALL set classification to `0` without calling the `Classifier`, and SHALL
still proceed to extraction. When `layers.rules` is an empty list, the runner
SHALL pass an empty `ClassificationRules` to the `Classifier` as an honest
zero-shot baseline. The runner SHALL NOT reload the default prompt to recover
rules for a non-default active prompt. The runner SHALL NOT parse rendered
prompt text to recover rules when the structured `layers.rules` data is
available. When rules are present but invalid, the runner SHALL apply the
existing error-handling policy and SHALL NOT substitute the error with a `0`
classification.

#### Scenario: Default prompt rules used
- **WHEN** the runner uses the default classification prompt
- **THEN** the default prompt's `layers.rules` are converted to `ClassificationRules` and passed to the `Classifier`

#### Scenario: Non-default prompt rules used
- **WHEN** the runner uses a non-default active prompt with valid layers
- **THEN** that prompt's `layers.rules` are converted to `ClassificationRules` and passed to the `Classifier`

#### Scenario: Absent layers sets classification to zero
- **WHEN** the runner uses an active prompt with missing or `None` layers
- **THEN** classification is set to `0` without calling the `Classifier`, and extraction proceeds

#### Scenario: Empty rules list runs zero-shot baseline
- **WHEN** the runner uses an active prompt with `layers.rules = []`
- **THEN** the `Classifier` is called with an empty `ClassificationRules` (zero-shot baseline), and extraction proceeds

#### Scenario: No default prompt reload for non-default active prompt
- **WHEN** the runner uses a non-default active prompt
- **THEN** it does not reload the default prompt to recover the rule set

### Requirement: Run dataset through the model
The component SHALL run each dataset example through two stages: classification
first, then extraction. Classification SHALL use the `Classifier` port, which
returns a `Classification` directly. Extraction SHALL use the `Extractor`
component. The component SHALL NOT use `classify_detailed`, `CandidateScorer`,
or any generative classification method. The candidate set SHALL be supplied to
the runner at construction time; the runner SHALL NOT infer or generate
candidates. The runner SHALL convert the active classification rules from the
classification prompt artifact's `layers.rules` to `ClassificationRules` and
pass them to the `Classifier`. When the prompt artifact has no layers, the
runner SHALL set classification to `0` without calling the `Classifier`; when
the rules list is empty, the runner SHALL call the `Classifier` with an empty
`ClassificationRules` as a zero-shot baseline. Classification SHALL
complete before extraction begins. When `skip_extraction` is set to true, the
component SHALL skip the `Extractor` entirely; `theses_raw` and `theses_norm`
SHALL be empty lists and `extract_status` SHALL be `skipped`.

#### Scenario: Single example processed
- **WHEN** the runner receives an example with `id`, `text`, `true_label`
- **THEN** classification and extraction are performed and the result contains `predicted_decision`, `selected_candidate`, `judgment_scores`, `theses_raw`, `theses_norm`

#### Scenario: Both steps succeed
- **WHEN** both classification and extraction return valid results
- **THEN** the result has all fields filled and `extract_status` and `classify_status` are `ok`

#### Scenario: Classification runs before extraction
- **WHEN** the runner processes an example on dev with rules present
- **THEN** the `Classifier` is called first, and the `Extractor` is called after classification completes

#### Scenario: Classification uses Classifier port
- **WHEN** the runner classifies an example
- **THEN** it invokes the `Classifier` port with `ClassificationRules` from the active prompt artifact, and does not call `CandidateScorer` or `classify_detailed`

#### Scenario: Absent layers skip classifier and still extract
- **WHEN** the runner processes an example with a layerless active prompt on dev
- **THEN** classification is set to `0` without calling the `Classifier`, and the `Extractor` is called

#### Scenario: Empty rules run zero-shot baseline
- **WHEN** the runner processes an example with an empty rules list on dev
- **THEN** the `Classifier` is called with empty `ClassificationRules`, and the `Extractor` is called after classification

#### Scenario: Candidate set supplied at construction
- **WHEN** the runner is constructed
- **THEN** it receives the candidate set and does not infer or generate candidates at runtime

#### Scenario: Skip extraction mode
- **WHEN** the runner is constructed with `skip_extraction=True`
- **THEN** the `Extractor` is not called, `theses_raw` and `theses_norm` are empty lists, `extract_status` is `skipped`, and only classification is performed

### Requirement: Skip extraction flag
The component SHALL accept a `skip_extraction` flag at construction time,
defaulting to false. When true, the component SHALL NOT invoke the `Extractor`
and SHALL only run classification for each example. The results artifact SHALL
still contain all fields from the result schema, with `theses_raw` and
`theses_norm` as empty lists and `extract_status` set to `skipped`.

#### Scenario: Skip extraction defaults to false
- **WHEN** the runner is constructed without `skip_extraction`
- **THEN** extraction is performed as normal

#### Scenario: Skip extraction avoids Extractor calls
- **WHEN** the runner is constructed with `skip_extraction=True` and processes N examples
- **THEN** zero `Extractor` calls are made and N classification calls are made

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
The component SHALL support batch execution by fanning out single-call client
methods with `asyncio.gather` and a semaphore at a configurable concurrency
level. The default concurrency SHALL be sane for local llama.cpp (no more than
8) and MAY be overridden via config.

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
failure, timeout, HTTP error, classification error) and record the error in the
corresponding result field. The component MUST NOT abort the entire run because
of a single error. A classification error SHALL NOT be treated as absent rules;
the existing error-handling policy SHALL be preserved.

#### Scenario: Single failure does not stop run
- **WHEN** the model returns an error for the example with `id=42`
- **THEN** the record with `id=42` is saved with the appropriate `failed` status and the run continues

#### Scenario: Classification error is not absent rules
- **WHEN** the `Classifier` raises an error during classification
- **THEN** `classify_status` is `failed`, `predicted_decision` is `None`, and the error is not silently converted to a `0` classification

#### Scenario: Summary at the end
- **WHEN** a run completes
- **THEN** a summary is printed: count of `ok`, count of `parse_status=failed`, count of network errors, count of timeouts
