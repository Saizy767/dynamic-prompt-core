# Spec

## Purpose

Provide a standalone key-phrase extraction component with its own system prompt
and independent model call, preserving the existing extraction algorithm,
thesis normalization, statuses, latencies, and error handling, without
receiving classification results as input.

## Requirements

### Requirement: Extractor is a standalone component
The `Extractor` SHALL be a component responsible for key-phrase extraction from
input text. It SHALL perform an independent model call, separate from the
`Classifier` call. The `Extractor` SHALL NOT perform classification and SHALL
NOT receive the classification result as an additional input. The `Extractor`
SHALL receive the same input data that extraction receives in the current
implementation.

#### Scenario: Independent model call
- **WHEN** the extractor processes an example
- **THEN** it makes its own model call, separate from the classifier call

#### Scenario: No classification result as input
- **WHEN** the extractor is invoked
- **THEN** it does not receive the classification result, selected candidate, or judgment scores

#### Scenario: Same input data as before refactoring
- **WHEN** the extractor processes an example
- **THEN** it receives the same input text and configuration that extraction received before the refactoring

### Requirement: Extractor uses a dedicated system prompt
The `Extractor` SHALL load and use a dedicated system prompt from a resource
file under `infrastructure/llm/prompts/`, separate from the `Classifier` system
prompt. Changing the `Extractor` system prompt SHALL NOT affect classification.
Changing the `Classifier` system prompt SHALL NOT affect extraction. The
`PromptArtifact` SHALL NOT be used to store the `Extractor`'s fixed system
prompt.

#### Scenario: Dedicated system prompt loaded
- **WHEN** the extractor is initialized
- **THEN** it loads its system prompt from a dedicated resource file separate from the classifier's

#### Scenario: Extractor prompt change does not affect classification
- **WHEN** the extractor system prompt is changed
- **THEN** classification behavior is unaffected

#### Scenario: Classifier prompt change does not affect extraction
- **WHEN** the classifier system prompt is changed
- **THEN** extraction behavior is unaffected

### Requirement: Extractor preserves extraction algorithm and normalization
The `Extractor` SHALL preserve the existing extraction algorithm: calling the
LLM transport with the extraction prompt, parsing the response into theses,
and normalizing theses via the existing normalizer (lemmatization,
lower-casing, whitespace collapsing, negation preservation). The `Extractor`
SHALL NOT redesign or rework the extraction algorithm.

#### Scenario: Theses extracted as before
- **WHEN** the extractor processes input text
- **THEN** the raw theses are extracted using the same algorithm as before the refactoring

#### Scenario: Normalization preserved
- **WHEN** raw theses are normalized
- **THEN** lemmatization, lower-casing, whitespace collapsing, and negation preservation are applied as in the existing implementation

### Requirement: Extractor preserves statuses and error handling
The `Extractor` SHALL preserve the existing status values (`ok`, `repaired`,
`failed`, `skipped`, `not_attempted`) and their semantics. The `Extractor`
SHALL preserve the existing error-handling policy: parse failures, timeouts,
HTTP errors, and unexpected shapes SHALL be recorded in the corresponding
result fields without aborting the run. The `Extractor` SHALL preserve latency
measurement.

#### Scenario: Status values preserved
- **WHEN** extraction succeeds, is repaired, fails, or is skipped
- **THEN** the corresponding status value is set as in the existing implementation

#### Scenario: Error does not abort run
- **WHEN** extraction fails for one example
- **THEN** the error is recorded in the result fields and the run continues

#### Scenario: Latency measured
- **WHEN** extraction completes
- **THEN** `extract_latency_ms` is recorded as in the existing implementation

### Requirement: Extractor runs after classification on dev
On the dev split, the `Extractor` SHALL run after the classification stage has
completed. This ordering SHALL hold even when rules are absent: with a layerless
active prompt classification is set to `0` without calling the `Classifier`,
and with an empty rules list the `Classifier` runs as a zero-shot baseline; in
both cases the `Extractor` runs afterwards. On the holdout split, the
`Extractor` SHALL be skipped (`skip_extraction=True`), as in the current
implementation.

#### Scenario: Extraction follows classification on dev
- **WHEN** the runner processes an example on dev with rules present
- **THEN** the classifier is called first, then the extractor is called

#### Scenario: Extraction runs even without rules on dev
- **WHEN** the runner processes an example on dev with absent or empty rules
- **THEN** classification completes (skipped for a layerless prompt, zero-shot classifier call for empty rules), and then the extractor is called

#### Scenario: Extraction skipped on holdout
- **WHEN** the runner processes an example on holdout with `skip_extraction=True`
- **THEN** the extractor is not called and `extract_status` is `skipped`
