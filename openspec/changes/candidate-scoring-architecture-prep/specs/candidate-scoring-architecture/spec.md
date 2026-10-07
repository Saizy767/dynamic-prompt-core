# Spec Delta

## Purpose

Define the target architecture for candidate-based scoring classification: the
scoring flow shape, the `CandidateScorer` port contract, layer responsibilities
in the new classification flow, component disposition, and the migration
boundary for introducing `Candidate`, `Judgment`, and `CandidateScorer`
incrementally in subsequent stages.

## ADDED Requirements

### Requirement: Candidate scoring flow
The classification flow SHALL be expressible as candidate scoring: a set of
candidates is scored by a `CandidateScorer` and a judgment is derived from the
scores. The flow SHALL NOT require classification results to be generated as
structured LLM output. The concept of a classification result is preserved;
only the generative `ClassificationResult` implementation is replaced.

#### Scenario: Scoring via port
- **WHEN** the classification flow is executed
- **THEN** scoring is performed through the `CandidateScorer` port

#### Scenario: No generative output required
- **WHEN** the `CandidateScorer` port contract is defined
- **THEN** it does not prescribe generative LLM structured output as the scoring mechanism

#### Scenario: Classification result concept preserved
- **WHEN** the scoring pipeline replaces the generative path
- **THEN** the concept of a classification result is preserved; only the generative `ClassificationResult` implementation is replaced

### Requirement: CandidateScorer port
`CandidateScorer` SHALL be an outbound port residing in
`application/ports/outbound/`. The application layer SHALL depend on this port;
a concrete implementation SHALL reside in `infrastructure`. The port SHALL NOT
depend on `infrastructure` or provider-specific types. Stage 1 fixes the port
**semantics** (scores candidates, returns judgments) but not the final Python
signature. The port SHALL NOT expose logits, token IDs, tokenizer APIs, or
model APIs.

#### Scenario: Port location
- **WHEN** the `CandidateScorer` port is defined
- **THEN** it resides under `application/ports/outbound/candidate_scorer`

#### Scenario: Application depends on port
- **WHEN** a use case scores candidates
- **THEN** it depends on `application.ports.outbound.candidate_scorer`, not on `infrastructure`

#### Scenario: Port free of provider types
- **WHEN** the `CandidateScorer` port is inspected
- **THEN** it contains no provider-specific abstractions

#### Scenario: Port hides LLM internals
- **WHEN** the `CandidateScorer` port contract is defined
- **THEN** it does not expose logits, token IDs, tokenizer APIs, or model APIs

#### Scenario: Semantics fixed, not signature
- **WHEN** Stage 1 is completed
- **THEN** the port semantics are fixed but the final Python signature is not prescribed

### Requirement: Layer responsibilities in scoring flow
`domain` SHALL define `Candidate` and `Judgment` as domain objects without
LLM-specific dependencies. `application` and `domain` SHALL own aggregation,
normalization policy, and the classification decision. `application` SHALL
orchestrate scoring through the `CandidateScorer` port and domain objects.
`infrastructure` SHALL own model inference mechanics and implement the
`CandidateScorer` port with all LLM-specific details. `interfaces` SHALL
assemble the concrete scorer and inject it into the application.

#### Scenario: Domain objects
- **WHEN** `Candidate` and `Judgment` are defined
- **THEN** they reside in `domain` and depend only on the standard library

#### Scenario: Application and domain own decision logic
- **WHEN** aggregation, normalization policy, or the classification decision is implemented
- **THEN** it resides in `application` or `domain`, not `infrastructure`

#### Scenario: Infrastructure owns inference mechanics
- **WHEN** model inference mechanics are implemented
- **THEN** they reside in `infrastructure`

#### Scenario: Application orchestration
- **WHEN** the scoring use case is executed
- **THEN** it uses the `CandidateScorer` port and domain objects

#### Scenario: Interfaces assembly
- **WHEN** the runtime is started
- **THEN** the composition root assembles the concrete scorer and injects it into the application

### Requirement: Component disposition
The migration SHALL classify existing classification components as remain,
migrate, or deprecate. `Candidate` and `Judgment` SHALL reside in `domain`.
`CandidateScorer` SHALL reside in `application/ports/outbound`. The generative
classification methods (`LLMClient.classify` path) SHALL become legacy/deprecated.
The generative `ClassificationResult` implementation SHALL be replaced; the
classification result concept is preserved. `BaselineRunner` disposition SHALL
be MIGRATE. The `LLMClient` port SHALL remain for non-classification calls
(e.g., thesis extraction).

#### Scenario: Generative classification legacy
- **WHEN** the scoring pipeline is introduced
- **THEN** the generative classification methods become legacy/deprecated

#### Scenario: BaselineRunner migrates
- **WHEN** the scoring pipeline is introduced
- **THEN** `BaselineRunner` migrates to use candidate scoring

#### Scenario: LLMClient retained
- **WHEN** non-classification LLM calls are needed
- **THEN** the `LLMClient` port remains available

### Requirement: Migration boundary
The architectural foundation SHALL allow `Candidate`, `Judgment`, and
`CandidateScorer` to be introduced in subsequent stages. Subsequent stages
SHALL preserve the established layer boundaries but need not be purely additive.
This stage SHALL NOT implement candidate scoring, logits extraction, batching,
or classification behavior changes.

#### Scenario: Boundary preservation
- **WHEN** subsequent stages introduce `Candidate`, `Judgment`, or `CandidateScorer`
- **THEN** the established layer boundaries are preserved

#### Scenario: Not purely additive
- **WHEN** subsequent stages modify existing code
- **THEN** they are not required to be purely additive, provided layer boundaries are preserved

#### Scenario: No implementation in this stage
- **WHEN** this stage is completed
- **THEN** no candidate scoring, logits extraction, batching, or classification behavior change is implemented

### Requirement: Existing behavior preserved
Existing public classification behavior SHALL be preserved. Stage 1 SHALL NOT
change runtime behavior.

#### Scenario: Behavior unchanged
- **WHEN** this stage is completed
- **THEN** existing runtime classification behavior is unchanged
