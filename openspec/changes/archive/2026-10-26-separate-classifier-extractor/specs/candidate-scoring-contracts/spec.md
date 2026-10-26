# Spec Delta

## MODIFIED Requirements

### Requirement: Dependency direction
The dependency direction SHALL be `domain` ← `application` ←
`infrastructure`, with interfaces remaining outermost. `domain` SHALL NOT
import `application`, `infrastructure`, or `interfaces`. `application` MAY
import `domain` and SHALL NOT import concrete infrastructure classifier
implementations. `infrastructure` MAY import `domain` and `application.ports`
and SHALL NOT import application use cases or `interfaces`. `Classifier` SHALL
be the dependency inversion boundary between application orchestration and
infrastructure classification. `Candidate` and `Judgment` SHALL remain domain
models usable by the `Classifier` implementation internally.

#### Scenario: Domain does not depend on application
- **WHEN** `domain` is inspected
- **THEN** it does not import `application`

#### Scenario: Domain does not depend on infrastructure
- **WHEN** `domain` is inspected
- **THEN** it does not import `infrastructure`

#### Scenario: Application depends on port not concrete classifier
- **WHEN** an application use case classifies input
- **THEN** it imports the `Classifier` port, not a concrete infrastructure classifier

#### Scenario: Infrastructure depends on ports not use cases
- **WHEN** `infrastructure` is inspected
- **THEN** it may import `application.ports` but does not import application use cases or `interfaces`

## REMOVED Requirements

### Requirement: CandidateScorer port location
**Reason**: The `CandidateScorer` port is replaced by the `Classifier` port,
which encapsulates the full classification process including scoring and
decision. The `Classifier` port is defined in the `classifier` capability.
**Migration**: Use the `Classifier` port from `application/ports/outbound/`
instead of `CandidateScorer` from `application/ports/outbound/candidate_scorer`.

### Requirement: Active classification rules as scoring input
**Reason**: Rules are now passed as `ClassificationRules` to the `Classifier`
port, not as a `Sequence[str]` to `CandidateScorer.score()`. The
`Classifier` receives `ClassificationRules` as a structured domain model.
**Migration**: Convert `PromptArtifact.layers.rules` to `ClassificationRules`
and pass to `Classifier.classify()`.

### Requirement: CandidateScorer contract
**Reason**: The `CandidateScorer` contract (accepts text + candidates + rules,
returns `list[Judgment]`) is replaced by the `Classifier` contract (accepts
text + `ClassificationRules`, returns `Classification`). The `Classifier`
absorbs candidate handling, scoring, and decision into a single call.
**Migration**: Replace `CandidateScorer.score()` + `ClassificationPolicy.classify()`
with a single `Classifier.classify()` call.

### Requirement: Multiple candidates per invocation
**Reason**: Candidate handling is now internal to the `Classifier` implementation.
The application no longer invokes the scorer with a candidate list; candidates
are supplied to the `Classifier` at construction time.
**Migration**: Supply candidates to the `Classifier` at construction, not per
invocation.

### Requirement: Candidate-judgment association
**Reason**: Judgment association is now internal to the `Classifier`. The
application receives a `Classification` with the selected candidate and full
judgment set, without needing to associate judgments with candidates externally.
**Migration**: Read `Classification.selected` and `Classification.judgments`
from the `Classifier` result.

### Requirement: Duplicate candidate validation at application boundary
**Reason**: Candidate validation moves inside the `Classifier` implementation.
The application no longer validates candidates before scoring; the `Classifier`
validates at its boundary.
**Migration**: The `Classifier` handles candidate validation internally.

### Requirement: Scoring semantics
**Reason**: The `CandidateScorer` SHALL NOT decide invariant is removed. The
`Classifier` SHALL decide the final classification — that is its purpose. The
evaluation-vs-decision boundary is redefined: the `Classifier` port returns
`Classification` (decision), while the scoring mechanism remains internal.
**Migration**: The `Classifier` port replaces both `CandidateScorer` and
`ClassificationPolicy`. The decision is produced inside the `Classifier`.

### Requirement: Port information boundary
**Reason**: The `CandidateScorer` port information boundary is replaced by the
`Classifier` port information boundary, defined in the `classifier` capability.
**Migration**: The `Classifier` port exposes no logits, token IDs, tokenizer
APIs, model objects, or provider-specific types.

### Requirement: Port boundary architecture validation
**Reason**: The `CandidateScorer`-specific import-linter contract is replaced by
the `Classifier` port boundary contract. The general `application-ports-isolated`
contract continues to enforce that `application.ports` does not import
`infrastructure` or `interfaces`.
**Migration**: The `Classifier` port module is covered by the existing
`application-ports-isolated` import-linter contract.
