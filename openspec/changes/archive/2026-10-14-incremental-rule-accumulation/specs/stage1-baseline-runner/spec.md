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
construction time; the runner SHALL NOT infer or generate candidates. When
`skip_extraction` is set to true, the component SHALL skip thesis extraction
entirely and only perform candidate-scoring classification; `theses_raw` and
`theses_norm` SHALL be empty lists and `extract_status` SHALL be `skipped`.

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

#### Scenario: Skip extraction mode
- **WHEN** the runner is constructed with `skip_extraction=True`
- **THEN** thesis extraction is not called, `theses_raw` and `theses_norm` are empty lists, `extract_status` is `skipped`, and only candidate-scoring classification is performed

## ADDED Requirements

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
