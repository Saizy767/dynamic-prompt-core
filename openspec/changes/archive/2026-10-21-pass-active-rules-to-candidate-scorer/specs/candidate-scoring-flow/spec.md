# Spec Delta

## ADDED Requirements

### Requirement: Classification rules as explicit use case input
The `classify_input` use case SHALL receive the active classification rules as
an explicit argument. The rules SHALL NOT be loaded from a global source, a
service locator, or an implicit default inside the use case. The caller is
responsible for supplying the rules from the active prompt version.

#### Scenario: Rules passed as explicit argument
- **WHEN** the `classify_input` use case is invoked
- **THEN** the active classification rules are passed as an explicit argument

#### Scenario: No implicit rule loading
- **WHEN** the `classify_input` use case is inspected
- **THEN** it does not load rules from a global source, service locator, or implicit default

## MODIFIED Requirements

### Requirement: ClassifyInput use case orchestration
A `ClassifyInput` use case SHALL coordinate the scoring-classification flow:
validate candidate uniqueness at the application boundary, invoke
`CandidateScorer.score` with the active classification rules to produce
judgments, and invoke `ClassificationPolicy.classify` to produce a
`Classification`. The use case SHALL receive its dependencies
(`CandidateScorer`, `ClassificationPolicy`) from outside via a typed dependency
object. The use case SHALL receive the active classification rules as an
explicit input. The use case SHALL NOT implement scoring mechanics, inspect
model outputs, or know whether the scorer uses logits, embeddings, an API, a
local model, or a mock.

#### Scenario: Use case validates candidates before scoring
- **WHEN** the use case is invoked with candidates
- **THEN** it validates candidate uniqueness before invoking the scorer

#### Scenario: Use case scores then classifies
- **WHEN** the use case is invoked with text, candidates, and rules
- **THEN** it invokes the scorer with the rules to produce judgments, then the policy to produce a classification

#### Scenario: Use case receives dependencies from outside
- **WHEN** the use case is constructed
- **THEN** the scorer and policy are passed in via a typed dependency object

#### Scenario: Use case receives rules as explicit input
- **WHEN** the use case is invoked
- **THEN** it receives the active classification rules as an explicit argument, not from global state

#### Scenario: Use case does not inspect model outputs
- **WHEN** the use case orchestrates the flow
- **THEN** it does not inspect logits, token IDs, or model internals

#### Scenario: Use case is scorer-mechanism agnostic
- **WHEN** the use case is run with different scorer implementations
- **THEN** its orchestration is identical regardless of the scoring mechanism
