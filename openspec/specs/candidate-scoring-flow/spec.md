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
text, or transport metadata. `Classification` is distinct from the legacy
generative `ClassificationResult` schema and does not replace it.

#### Scenario: Classification associates selected candidate and judgments
- **WHEN** a classification is created from a selected candidate and its judgments
- **THEN** it exposes that candidate and those judgments as a tuple

#### Scenario: Classification immutability
- **WHEN** a classification has been created
- **THEN** its selected candidate cannot be mutated and its judgments tuple cannot be mutated or appended to

#### Scenario: Classification free of model internals
- **WHEN** a classification is inspected
- **THEN** it exposes no logits, token IDs, tokenizer state, model objects, prompt text, or transport metadata

#### Scenario: Classification distinct from legacy result
- **WHEN** the new classification flow produces a decision
- **THEN** the result is a `Classification`, not a legacy `ClassificationResult`

### Requirement: Classification domain isolation
`Classification` SHALL be usable by importing only domain modules and the Python
standard library. It SHALL NOT import or indirectly require `torch`,
`transformers`, `vllm`, `sglang`, `mlx`, `openai`, `aiohttp`, provider SDKs,
tokenizer libraries, inference frameworks, application ports, infrastructure
modules, or interface modules.

#### Scenario: Classification imports only domain and stdlib
- **WHEN** `Classification` is imported
- **THEN** only domain modules and the Python standard library are required

### Requirement: ClassificationPolicy as application service
`ClassificationPolicy` SHALL be an application-internal strategy abstraction
residing in `application/services/classification_policy.py`. It SHALL be defined
as a `typing.Protocol`. It is NOT an outbound port and NOT an external
dependency: it is a pure application-layer decision rule. Infrastructure SHALL
NOT implement this policy. The concrete `ArgmaxClassificationPolicy`
implementation SHALL reside alongside the protocol in the same module. The
`ClassifyInput` use case SHALL depend on the `ClassificationPolicy` protocol,
not the concrete implementation, so the decision rule is substitutable and
testable with doubles.

#### Scenario: Policy resides in application services
- **WHEN** the `ClassificationPolicy` is located
- **THEN** it is in `application/services/classification_policy.py`

#### Scenario: Policy is a typing.Protocol
- **WHEN** the `ClassificationPolicy` is defined
- **THEN** it is a `typing.Protocol`

#### Scenario: Policy not implemented by infrastructure
- **WHEN** the policy is inspected
- **THEN** no infrastructure module implements or owns the classification decision

#### Scenario: Concrete policy alongside protocol
- **WHEN** the concrete `ArgmaxClassificationPolicy` is located
- **THEN** it resides in the same `application/services/classification_policy.py` module as the protocol

### Requirement: ClassificationPolicy consumes judgments not model outputs
`ClassificationPolicy` SHALL accept a `list[Judgment]` and SHALL return a
`Classification`. The policy SHALL NOT accept raw LLM responses, logits, token
IDs, generated text, or provider response objects. The policy SHALL NOT perform
model inference, tokenization, logits extraction, prompt construction, or
provider API calls. The policy's sole inputs are the judgments produced by a
`CandidateScorer`.

#### Scenario: Policy accepts judgments
- **WHEN** the policy is invoked
- **THEN** it accepts a list of `Judgment` objects

#### Scenario: Policy returns classification
- **WHEN** the policy successfully produces a decision
- **THEN** it returns a `Classification`

#### Scenario: Policy rejects model internals
- **WHEN** the policy contract is inspected
- **THEN** it accepts no logits, token IDs, generated text, or provider response objects

#### Scenario: Policy performs no inference
- **WHEN** the policy is invoked
- **THEN** it performs no model inference, tokenization, logits extraction, prompt construction, or provider API calls

### Requirement: ClassificationPolicy responsibilities
`ClassificationPolicy` SHALL be responsible for semantic decisions: selecting
the highest-scoring candidate, applying thresholds, handling ties, deciding
whether no candidate is sufficiently supported, and converting judgments into
the application's `Classification` representation. The initial implementation
SHALL select the candidate with the maximum score. Tie behavior for equal
maximum scores SHALL be explicit and deterministic. The policy SHALL NOT
calibrate scores, normalize scores, or redefine the score scale.

#### Scenario: Highest scoring candidate selected
- **WHEN** judgments with scores `A=0.2`, `B=0.8`, `C=0.5` are classified
- **THEN** the selected candidate is `B`

#### Scenario: Single candidate selected
- **WHEN** a single judgment is classified
- **THEN** that candidate is selected

#### Scenario: Tie behavior is deterministic
- **WHEN** two or more candidates share the maximum score
- **THEN** the policy selects one deterministically according to its documented tie rule

#### Scenario: Policy does not calibrate or normalize
- **WHEN** the policy classifies judgments
- **THEN** it does not calibrate, normalize, or rescale the scores

### Requirement: ClassificationPolicy preserves judgment order
`ClassificationPolicy` SHALL preserve the scorer-provided judgment order in the
returned `Classification.judgments`. The judgment sequence in `Classification`
SHALL be in the same order as supplied to the policy. For the initial
`ArgmaxClassificationPolicy`, when two or more candidates share the maximum
score, the judgment appearing first in the input order SHALL win. Preserving
order makes tie-breaking deterministic and avoids hidden dependencies on
sort order.

#### Scenario: Classification judgments preserve input order
- **WHEN** the policy receives judgments in order `[A, B, C]`
- **THEN** `Classification.judgments` is `(A, B, C)` in that same order

#### Scenario: Tie breaks to first in input order
- **WHEN** judgments `[A=0.8, B=0.8, C=0.5]` are classified
- **THEN** the selected candidate is `A`, the first judgment with the maximum score

### Requirement: ClassificationPolicy empty input semantics
`ClassificationPolicy` SHALL define explicit behavior when invoked with an empty
judgment list. Empty judgments SHALL NOT produce a silent default
classification; the policy SHALL either raise an exception or return a
classification that explicitly denotes no candidate, according to its documented
contract. The behavior SHALL be documented and testable.

#### Scenario: Empty judgments handled explicitly
- **WHEN** the policy is invoked with an empty judgment list
- **THEN** the behavior is an explicit raise or an explicit no-candidate result, not a silent default

### Requirement: ClassifyInput use case orchestration
A `ClassifyInput` use case SHALL coordinate the scoring-classification flow:
validate candidate uniqueness at the application boundary, invoke
`CandidateScorer.score` to produce judgments, and invoke
`ClassificationPolicy.classify` to produce a `Classification`. The use case
SHALL receive its dependencies (`CandidateScorer`, `ClassificationPolicy`) from
outside via a typed dependency object. The use case SHALL NOT implement scoring
mechanics, inspect model outputs, or know whether the scorer uses logits,
embeddings, an API, a local model, or a mock.

#### Scenario: Use case validates candidates before scoring
- **WHEN** the use case is invoked with candidates
- **THEN** it validates candidate uniqueness before invoking the scorer

#### Scenario: Use case scores then classifies
- **WHEN** the use case is invoked with text and candidates
- **THEN** it invokes the scorer to produce judgments, then the policy to produce a classification

#### Scenario: Use case receives dependencies from outside
- **WHEN** the use case is constructed
- **THEN** the scorer and policy are passed in via a typed dependency object

#### Scenario: Use case does not inspect model outputs
- **WHEN** the use case orchestrates the flow
- **THEN** it does not inspect logits, token IDs, or model internals

#### Scenario: Use case is scorer-mechanism agnostic
- **WHEN** the use case is run with different scorer implementations
- **THEN** its orchestration is identical regardless of the scoring mechanism

### Requirement: Scoring classification separation invariant
`CandidateScorer` SHALL NOT select a winning candidate, rank candidates for
selection, or aggregate judgments into a classification decision.
`ClassificationPolicy` SHALL NOT perform model inference or produce scores. The
flow `score → classify` SHALL be the only supported composition; a combined
`scorer.classify(...)` operation SHALL NOT exist. This separation permits
multiple classification strategies, offline re-ranking, threshold experiments,
deterministic testing, non-LLM scorers, and score inspection.

#### Scenario: Scorer does not classify
- **WHEN** the scorer port contract is inspected
- **THEN** it exposes no classification, selection, ranking, or aggregation operation

#### Scenario: Policy does not score
- **WHEN** the policy contract is inspected
- **THEN** it exposes no scoring, model inference, or candidate evaluation operation

#### Scenario: No combined score-and-classify operation
- **WHEN** the application flow is inspected
- **THEN** scoring and classification are separate steps composed as `score → classify`

### Requirement: Candidate validation at application boundary
Candidate request validation SHALL be performed at the application boundary
before the scorer is invoked. The use case SHALL reject empty candidate lists
(when classification requires at least one candidate) and duplicate candidates
before any scoring occurs. The infrastructure scorer SHALL NOT be responsible
for enforcing application-level candidate-set semantics.

#### Scenario: Empty candidate list rejected
- **WHEN** the use case is invoked with an empty candidate list
- **THEN** the request is rejected before the scorer is invoked

#### Scenario: Duplicate candidates rejected before scoring
- **WHEN** the use case is invoked with duplicate candidates
- **THEN** the request is rejected before the scorer is invoked

#### Scenario: Infrastructure not responsible for candidate-set semantics
- **WHEN** the scorer is invoked
- **THEN** it is not responsible for application-level candidate-set validation

### Requirement: Existing generative classification preserved
The existing generative classification pipeline SHALL continue to operate
unchanged. `LLMClient.classify`, `AsyncTask.classify_detailed`,
`ClassificationResult`, `BaselineRunner`, existing parsing and validation
behavior, and existing classification tests SHALL continue to work unchanged.
No existing caller SHALL be migrated to the new scoring-classification flow in
this change. The new flow is additive and structurally distinct from the
generative path; no compatibility layer SHALL make the new scorer produce a
legacy `ClassificationResult`. The new flow SHALL be implemented and
independently testable but SHALL NOT be wired into any existing production
caller; existing production callers remain on the legacy generative path.

#### Scenario: Generative pipeline unchanged
- **WHEN** this change is completed
- **THEN** the generative classification pipeline continues to operate unchanged

#### Scenario: No caller migrated
- **WHEN** this change is completed
- **THEN** no existing caller is migrated to the new scoring-classification flow

#### Scenario: New flow not wired into existing callers
- **WHEN** this change is completed
- **THEN** the new flow is implemented and independently testable but no existing production caller invokes it

#### Scenario: No compatibility layer to legacy result
- **WHEN** the new flow is inspected
- **THEN** no adapter makes the scorer or policy produce a legacy `ClassificationResult`

#### Scenario: Existing tests pass
- **WHEN** the existing classification test suite is run
- **THEN** all tests pass

### Requirement: Scoring flow dependency direction
The dependency graph SHALL remain `domain` ← `application` ← `infrastructure`.
`Classification` SHALL reside in `domain/models`. `ClassificationPolicy` and
its concrete `ArgmaxClassificationPolicy` implementation SHALL reside in
`application/services/classification_policy.py`. `ClassifyInput` SHALL reside in
`application/use_cases`. The application use case SHALL import the
`CandidateScorer` outbound port and the `ClassificationPolicy` application
service protocol, not concrete infrastructure implementations. The domain SHALL
NOT import application, infrastructure, or interfaces.

#### Scenario: Classification in domain models
- **WHEN** `Classification` is located
- **THEN** it resides in `domain/models`

#### Scenario: Policy in application services
- **WHEN** `ClassificationPolicy` is located
- **THEN** it resides in `application/services/classification_policy.py`

#### Scenario: Use case depends on abstractions not concrete implementations
- **WHEN** the `ClassifyInput` use case is inspected
- **THEN** it imports the `CandidateScorer` port and the `ClassificationPolicy` protocol, not concrete infrastructure

#### Scenario: Domain does not depend on outer layers
- **WHEN** the domain is inspected
- **THEN** it does not import application, infrastructure, or interfaces

### Requirement: Scoring flow architecture validation
The `application-ports-isolated` import-linter contract SHALL continue to pass.
`application.ports` SHALL NOT import `infrastructure` or `interfaces`, directly
or indirectly. The `ClassificationPolicy` resides in `application/services`, not
`application/ports`, because it is an application-internal strategy, not an
infrastructure-implemented outbound port; it SHALL NOT import `infrastructure` or
`interfaces`. No exception to `application-ports-isolated` SHALL be introduced
for the `ClassifyInput` use case.

#### Scenario: Application ports isolated from infrastructure
- **WHEN** `lint-imports` is run
- **THEN** `application.ports` does not import `infrastructure` or `interfaces`

#### Scenario: Policy does not import infrastructure
- **WHEN** the `ClassificationPolicy` module is inspected
- **THEN** it does not import `infrastructure` or `interfaces`

#### Scenario: No port-boundary exceptions introduced
- **WHEN** the new policy service and use case are added
- **THEN** no exception to `application-ports-isolated` is introduced
