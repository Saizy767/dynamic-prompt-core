# Spec Delta

## ADDED Requirements

### Requirement: Domain isolation via stdlib-only allowlist
`domain` SHALL depend only on the Python standard library plus an explicit
allowlist of permitted third-party modules. LLM providers, tokenizers, logits,
inference engines, and transport protocols SHALL NOT be on the allowlist. This
policy replaces a per-library blacklist: any third-party import in `domain` not
on the allowlist is a violation, so the check does not grow with every new LLM
library. The policy SHALL be checked by the linter.

#### Scenario: Domain imports stdlib
- **WHEN** `domain` imports `dataclasses` and `typing`
- **THEN** the check passes

#### Scenario: Domain imports non-allowlisted third-party
- **WHEN** `domain` imports a third-party module not on the explicit allowlist
- **THEN** the import is forbidden by the linter

#### Scenario: Domain imports tokenizer
- **WHEN** `domain` imports a tokenizer library (e.g., `transformers`)
- **THEN** the import is forbidden by the linter

#### Scenario: Domain imports transport
- **WHEN** `domain` imports a transport protocol library (e.g., `aiohttp`)
- **THEN** the import is forbidden by the linter

### Requirement: Infrastructure ownership of model inference mechanics
`infrastructure` SHALL own model inference mechanics: logits extraction,
tokenization, batching, and provider adapters. These details SHALL NOT leak
into `domain` or `application`.

#### Scenario: Logits extraction in infrastructure
- **WHEN** logits extraction is implemented
- **THEN** it resides in `infrastructure`

#### Scenario: Tokenization in infrastructure
- **WHEN** tokenization for scoring is implemented
- **THEN** it resides in `infrastructure`

#### Scenario: Inference mechanics do not leak
- **WHEN** `domain` or `application` is inspected
- **THEN** it contains no logits extraction, tokenization, provider adapters, or model inference mechanics

### Requirement: CandidateScorer port boundary
`CandidateScorer` SHALL reside in `application/ports/outbound/`. `application`
SHALL depend on this port; `infrastructure` SHALL implement it. The
import-linter contract for this boundary SHALL be added together with the port
in the stage that creates it, not in this stage.

#### Scenario: Port in application ports
- **WHEN** the `CandidateScorer` port is located
- **THEN** it is under `application/ports/outbound/candidate_scorer`

#### Scenario: Application depends on port not infrastructure
- **WHEN** an application use case scores candidates
- **THEN** it imports the port, not the infrastructure implementation

#### Scenario: Port contract added with port
- **WHEN** the `CandidateScorer` port is created in a subsequent stage
- **THEN** its import-linter boundary contract is added in the same stage

### Requirement: Architecture validation for candidate scoring
import-linter SHALL validate existing dependency boundaries in this stage:
domain stdlib-only allowlist enforcement. The `CandidateScorer` port-boundary
contract SHALL be added in the stage that creates the port. All contracts SHALL
reside in `pyproject.toml` and SHALL fail CI on violation.

#### Scenario: Domain allowlist validated
- **WHEN** `lint-imports` is run
- **THEN** the domain stdlib-only allowlist policy is checked

#### Scenario: Port boundary deferred
- **WHEN** this stage is completed
- **THEN** no import-linter contract for the not-yet-existing `CandidateScorer` port is present

#### Scenario: Violation fails CI
- **WHEN** a validated dependency boundary violation is detected
- **THEN** CI fails
