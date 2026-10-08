# Spec Delta

## MODIFIED Requirements

### Requirement: Run dataset through the model
The component SHALL run each dataset example through two steps: thesis
extraction using `extract_theses_detailed` with the fixed extraction prompt,
and classification using the candidate-scoring flow
(`CandidateScorer.score` → `ClassificationPolicy.classify`). The component
SHALL NOT use `classify_detailed` or any generative classification method. The
component SHALL use `AsyncTask` from `llm-transport-client` for thesis
extraction and `CandidateScorer` for classification, and MUST NOT contact the
HTTP server directly. The candidate set SHALL be supplied to the runner at
construction time; the runner SHALL NOT infer or generate candidates.

#### Scenario: Single example processed
- **WHEN** the runner receives an example with `id`, `text`, `true_label`
- **THEN** thesis extraction and candidate-scoring classification are performed and the result contains `predicted_decision`, `selected_candidate`, `judgment_scores`, `theses_raw`, `theses_norm`

#### Scenario: Both steps succeed
- **WHEN** both extraction and candidate scoring return valid results
- **THEN** the result has all fields filled and `extract_status` and `classify_status` are `ok`

#### Scenario: Classification uses candidate scoring
- **WHEN** the runner classifies an example
- **THEN** it invokes `CandidateScorer.score(text, candidates)` followed by `ClassificationPolicy.classify(judgments)` and does not call `classify_detailed`

#### Scenario: Candidate set supplied at construction
- **WHEN** the runner is constructed
- **THEN** it receives the candidate set and does not infer or generate candidates at runtime

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
