# Spec Delta

## ADDED Requirements

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
