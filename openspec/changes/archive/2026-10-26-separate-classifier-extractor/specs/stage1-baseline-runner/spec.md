# Spec Delta

## MODIFIED Requirements

### Requirement: Active prompt rules extraction by runner
The runner SHALL extract classification rules from the active
`PromptArtifact` and convert them to `ClassificationRules`. When the active
prompt has valid `layers.rules`, the runner SHALL pass the converted
`ClassificationRules` to the `Classifier`. When `layers` is `None` or
`layers.rules` is an empty list, the runner SHALL set classification to `0`
without calling the `Classifier`, and SHALL still proceed to extraction. The
runner SHALL NOT reload the default prompt to recover rules for a non-default
active prompt. The runner SHALL NOT parse rendered prompt text to recover rules
when the structured `layers.rules` data is available. When rules are present but
invalid, the runner SHALL apply the existing error-handling policy and SHALL NOT
substitute the error with a `0` classification.

#### Scenario: Default prompt rules used
- **WHEN** the runner uses the default classification prompt
- **THEN** the default prompt's `layers.rules` are converted to `ClassificationRules` and passed to the `Classifier`

#### Scenario: Non-default prompt rules used
- **WHEN** the runner uses a non-default active prompt with valid layers
- **THEN** that prompt's `layers.rules` are converted to `ClassificationRules` and passed to the `Classifier`

#### Scenario: Absent layers sets classification to zero
- **WHEN** the runner uses an active prompt with missing or `None` layers
- **THEN** classification is set to `0` without calling the `Classifier`, and extraction proceeds

#### Scenario: Empty rules list sets classification to zero
- **WHEN** the runner uses an active prompt with `layers.rules = []`
- **THEN** classification is set to `0` without calling the `Classifier`, and extraction proceeds

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
pass them to the `Classifier`. When rules are absent or empty, the runner SHALL
set classification to `0` without calling the `Classifier`. Classification SHALL
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

#### Scenario: Absent rules skip classifier and still extract
- **WHEN** the runner processes an example with absent or empty rules on dev
- **THEN** classification is set to `0` without calling the `Classifier`, and the `Extractor` is called

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
