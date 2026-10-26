# Candidate Scoring Flow Specification

## Purpose

Define the application-level scoring and classification flow that turns
candidate judgments into a classification decision, keeping the scorer strictly
responsible for evaluation and the classification policy strictly responsible
for the decision.

## Requirements

### Requirement: Classification domain concept
A `Classification` SHALL be an immutable domain model representing the
application-level classification decision produced from a set of judgments. It
SHALL associate the selected `Candidate` with the full judgment set from which
the decision was derived, stored as a `tuple[Judgment, ...]` so that the
judgment sequence is deeply immutable. `Classification` SHALL NOT expose logits,
token IDs, tokenizer state, model objects, provider response objects, prompt
text, or transport metadata. `Classification` is the sole application-level
classification result; no generative `ClassificationResult` schema exists.

#### Scenario: Classification associates selected candidate and judgments
- **WHEN** a classification is created from a selected candidate and its judgments
- **THEN** it exposes that candidate and those judgments as a tuple

#### Scenario: Classification immutability
- **WHEN** a classification has been created
- **THEN** its selected candidate cannot be mutated and its judgments tuple cannot be mutated or appended to

#### Scenario: Classification free of model internals
- **WHEN** a classification is inspected
- **THEN** it exposes no logits, token IDs, tokenizer state, model objects, prompt text, or transport metadata

#### Scenario: Classification is the sole classification result
- **WHEN** the classification flow produces a decision
- **THEN** the result is a `Classification` and no `ClassificationResult` type exists in the codebase

### Requirement: Classification domain isolation
`Classification` SHALL be usable by importing only domain modules and the Python
standard library. It SHALL NOT import or indirectly require `torch`,
`transformers`, `vllm`, `sglang`, `mlx`, `openai`, `aiohttp`, provider SDKs,
tokenizer libraries, inference frameworks, application ports, infrastructure
modules, or interface modules.

#### Scenario: Classification imports only domain and stdlib
- **WHEN** `Classification` is imported
- **THEN** only domain modules and the Python standard library are required

### Requirement: Scoring flow dependency direction
The dependency graph SHALL remain `domain` ← `application` ← `infrastructure`.
`Classification` SHALL reside in `domain/models`. The `Classifier` outbound port
SHALL reside in `application/ports/outbound/`. The `Classifier` infrastructure
implementation SHALL reside in `infrastructure/llm/`. The application use case
SHALL import the `Classifier` outbound port, not concrete infrastructure
implementations. `ClassificationPolicy` as a separate application-internal
strategy SHALL be removed; the decision logic SHALL reside inside the
`Classifier` implementation. The domain SHALL NOT import application,
infrastructure, or interfaces.

#### Scenario: Classification in domain models
- **WHEN** `Classification` is located
- **THEN** it resides in `domain/models`

#### Scenario: Classifier port in application outbound
- **WHEN** the `Classifier` port is located
- **THEN** it resides in `application/ports/outbound/`

#### Scenario: Use case depends on port not concrete implementation
- **WHEN** the application use case is inspected
- **THEN** it imports the `Classifier` port, not concrete infrastructure

#### Scenario: Domain does not depend on outer layers
- **WHEN** the domain is inspected
- **THEN** it does not import application, infrastructure, or interfaces

### Requirement: Scoring flow architecture validation
The `application-ports-isolated` import-linter contract SHALL continue to pass.
`application.ports` SHALL NOT import `infrastructure` or `interfaces`, directly
or indirectly. The `Classifier` port resides in `application/ports/outbound/`
and SHALL NOT import `infrastructure` or `interfaces`. No exception to
`application-ports-isolated` SHALL be introduced for the classification use
case.

#### Scenario: Application ports isolated from infrastructure
- **WHEN** `lint-imports` is run
- **THEN** `application.ports` does not import `infrastructure` or `interfaces`

#### Scenario: Classifier port does not import infrastructure
- **WHEN** the `Classifier` port module is inspected
- **THEN** it does not import `infrastructure` or `interfaces`

#### Scenario: No port-boundary exceptions introduced
- **WHEN** the new `Classifier` port is added
- **THEN** no exception to `application-ports-isolated` is introduced
