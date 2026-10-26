# Spec Delta

## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: ClassificationPolicy as application service
**Reason**: `ClassificationPolicy` as a separate application-internal strategy
abstraction is removed. The decision logic (selecting the highest-scoring
candidate, tie-breaking by input order) moves inside the `Classifier`
implementation. The `Classifier` port returns `Classification` directly,
eliminating the need for a separate policy step.
**Migration**: The `Classifier` port replaces the
`CandidateScorer` + `ClassificationPolicy` two-step flow with a single
`Classifier.classify()` call.

### Requirement: ClassificationPolicy consumes judgments not model outputs
**Reason**: `ClassificationPolicy` is removed. The `Classifier` implementation
internally consumes judgments produced by its own scoring mechanism and
produces a `Classification`. The application no longer interacts with
judgments directly.
**Migration**: The `Classifier` handles judgment consumption internally.

### Requirement: ClassificationPolicy responsibilities
**Reason**: The policy responsibilities (selecting highest-scoring candidate,
tie-breaking, converting judgments to `Classification`) are absorbed by the
`Classifier` implementation. The decision logic is preserved but no longer
exposed as a separate abstraction.
**Migration**: The `Classifier` preserves the existing decision logic
internally.

### Requirement: ClassificationPolicy preserves judgment order
**Reason**: Judgment order preservation is now an internal concern of the
`Classifier` implementation. The `Classifier` returns a `Classification` with
judgments in the same order as the candidate input, preserving the existing
tie-breaking semantics.
**Migration**: The `Classifier` preserves judgment order internally.

### Requirement: ClassificationPolicy empty input semantics
**Reason**: The empty-judgment-list case is no longer an application-level
concern. The `Classifier` receives `ClassificationRules`; absent rules are
handled at the orchestration level (classification set to `0` without calling
the `Classifier`). The `Classifier` is only invoked with non-empty, valid rules.
**Migration**: Absent rules are handled by `BaselineRunner` before the
`Classifier` is called.

### Requirement: Classification rules as explicit use case input
**Reason**: The `classify_input` use case is replaced by the `Classifier` port.
Rules are passed as `ClassificationRules` to the `Classifier`, not as an
explicit argument to a use case.
**Migration**: Pass `ClassificationRules` to `Classifier.classify()`.

### Requirement: ClassifyInput use case orchestration
**Reason**: The `ClassifyInput` use case (validate candidates, invoke
`CandidateScorer.score`, invoke `ClassificationPolicy.classify`) is replaced by
the `Classifier` port, which encapsulates the full flow. The use case is
removed or adapted to call the `Classifier` directly.
**Migration**: Replace `ClassifyInput` orchestration with a single
`Classifier.classify()` call.

### Requirement: Scoring classification separation invariant
**Reason**: The `score → classify` two-step separation is replaced by a single
`Classifier.classify()` call that returns `Classification` directly. The
separation between scoring and decision becomes an internal implementation
detail of the `Classifier`, not an architectural boundary. The
`CandidateScorer` SHALL NOT decide invariant is removed because the
`Classifier` SHALL decide.
**Migration**: The `Classifier` port replaces the two-step flow. Scoring and
decision remain separated internally within the `Classifier` implementation.

### Requirement: Candidate validation at application boundary
**Reason**: Candidate validation moves inside the `Classifier` implementation.
The application no longer validates candidates before scoring; the
`Classifier` validates at its boundary.
**Migration**: The `Classifier` handles candidate validation internally.
