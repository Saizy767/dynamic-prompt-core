# Design

## Context

The layered architecture is in place with import-linter contracts enforcing
layer direction and domain isolation. The `candidate-scoring-architecture-prep`
change fixed the `CandidateScorer` port location
(`application/ports/outbound/candidate_scorer`), the layer responsibilities, and
the component disposition; it explicitly deferred creation of `Candidate`,
`Judgment`, and `CandidateScorer` to this stage, and deferred the port-boundary
import-linter contract to "the stage that creates the port."

Existing conventions this design follows:
- Domain models are `@dataclass(frozen=True)` in `domain/models/` (e.g.,
  `PromptLayer`, `PromptArtifact` in `domain/prompts/base.py`), with
  `__post_init__` validation using `object.__setattr__` for frozen fields.
- Domain errors are `ValueError` subclasses in `domain/errors/`.
- Outbound ports are `@runtime_checkable typing.Protocol` classes in flat files
  under `application/ports/outbound/` (e.g., `llm_client.py`, `normalizer.py`),
  async where I/O is involved.
- Public `__init__.py` files re-export public types with `__all__`.
- `domain` is stdlib-only (enforced by the `domain-stdlib-only-allowlist`
  import-linter contract).

One constraint shapes the port-boundary contract: `application` use cases
currently import `infrastructure` directly (e.g., `run_baseline/runner.py`
imports `infrastructure.llm` and `infrastructure.nlp` at module level). A broad
`application → infrastructure` forbidden contract would fail immediately. The
dependency-inversion boundary is therefore enforced at the **ports** layer, not
the whole application.

## Goals / Non-Goals

**Goals:**
- Introduce `Candidate` and `Judgment` as immutable, stdlib-only domain models
  matching existing domain conventions.
- Introduce `CandidateScorer` as an async `typing.Protocol` outbound port
  matching existing port conventions, with a strict all-or-nothing contract:
  a successful call returns exactly one judgment per candidate.
- Provide a duplicate-candidate validation helper at the application boundary so
  invalid requests are rejected before the scorer is invoked.
- Fix the evaluation-vs-decision boundary: the scorer evaluates candidates and
  does not decide, rank, select, or aggregate.
- Add an import-linter contract that enforces the port boundary (ports must not
  depend on infrastructure, directly or indirectly) and is green immediately.
- Expose the new types through the established public `__init__.py` surfaces.

**Non-Goals:**
- Enforce `application → infrastructure` isolation broadly (use cases currently
  import infrastructure; that migration is out of scope).
- Fix ownership of score normalization or calibration policy. That decision is
  deferred until the logits backend produces real score semantics that can be
  evaluated experimentally.
- Provide a concrete scorer or test double beyond what is needed to verify the
  domain contracts, the validation helper, and port typing.
- Migrate any existing caller to `CandidateScorer`.
- Finalize the application exception hierarchy for scoring failures (deferred to
  the concrete-scorer stage).

## Decisions

### Decision: `Candidate` is a frozen string-backed dataclass in `domain/models/candidate.py`
`Candidate` holds a single `value: str` field and is decorated
`@dataclass(frozen=True)`, giving immutability, value-based equality, and
hashing for free — all derived from the semantic value, not object identity.
`__post_init__` rejects empty/blank values. The value is stored verbatim: no
`strip`, `lower`, `casefold`, or whitespace collapsing is applied. The candidate
is a semantic value; normalization is the caller's responsibility or a future
candidate-generation layer's responsibility. Imports only `dataclasses` and the
stdlib, satisfying domain isolation.
- **Alternative considered**: a multi-field value object (`CandidateId`,
  `CandidateName`, `CandidateMetadata`). Rejected: the spec requires a simple,
  domain-oriented identity; a single semantic value is sufficient and keeps the
  contract minimal.
- **Alternative considered**: auto-normalize the value (`strip`/`casefold`).
  Rejected: the candidate is a semantic value; automatic normalization would
  silently change identity and hide caller mistakes. Normalization belongs to
  the candidate-generation layer.
- **Alternative considered**: `pydantic` for validation. Rejected: `domain` is
  stdlib-only (enforced by import-linter); `pydantic` is on the forbidden list.

### Decision: `Judgment` is a frozen dataclass in `domain/models/judgment.py` associating `Candidate` and `float`
`Judgment` holds `candidate: Candidate` and `score: float`, decorated
`@dataclass(frozen=True)` for immutability and value-based equality.
`__post_init__` validates the score: `math.isnan(score)` or `math.isinf(score)`
raises a domain `ValueError`. No `[0, 1]` constraint and no clamping. `Judgment`
is an immutable domain model; no architectural claim is made about it being a
DDD value object, leaving its structure free to evolve without that
categorization constraint. Imports only `dataclasses`, `math`, and the
`Candidate` domain type.
- **Alternative considered**: a `Score` value object wrapping the float.
  Rejected: adds a layer without behavioral benefit at this stage; the validity
  rules are simple enough to enforce in `Judgment.__post_init__`. A `Score` type
  can be extracted later if normalization/calibration policies need it.
- **Alternative considered**: allow `NaN`/`inf` and flag them. Rejected: the spec
  requires rejection, not silent acceptance.

### Decision: Score is an opaque ranking signal, not a probability
The score is documented as an opaque numeric ranking signal where higher means
stronger model-backed support for the candidate. The spec forbids assuming it is
a probability, confidence value, calibrated likelihood, or threshold-ready
value. No `[0, 1]` constraint is imposed. This keeps the contract open for
future logits-based backends whose score semantics are established separately
from probability calibration.
- **Alternative considered**: constrain scores to `[0, 1]` as probabilities.
  Rejected: future logits backends may expose unnormalized scores; imposing the
  constraint now would force premature calibration.

### Decision: `CandidateScorer` is a flat-file `@runtime_checkable typing.Protocol` in `application/ports/outbound/candidate_scorer.py`
The port is a `typing.Protocol` with a single async method:
`async def score(self, text: str, candidates: list[Candidate]) -> list[Judgment]`.
This matches every existing outbound port (`LLMClient`, `Normalizer`,
`TeacherLLMClient` are flat-file `@runtime_checkable` Protocols). The module
path `application.ports.outbound.candidate_scorer` satisfies the location fixed
by the prep stage. A flat file is chosen over a directory package to match the
existing convention; the module path is identical either way.
`@runtime_checkable` is an implementation convention matching existing ports,
not an architectural contract — it is not part of the spec. Its only purpose is
structural typing for dependency injection.
- **Alternative considered**: a `candidate_scorer/` directory package. Rejected:
  all existing ports are flat files; introducing a directory for one port breaks
  the convention without benefit.
- **Alternative considered**: `abc.ABC`. Rejected: the architecture spec defaults
  to `typing.Protocol` for ports; no runtime inheritance is required.

### Decision: All-or-nothing score contract — exactly one judgment per candidate
A successful `score()` call SHALL return exactly one `Judgment` per supplied
candidate (`len(result) == len(candidates)`). If the scorer cannot evaluate any
candidate, the call raises an exception. Partial-result semantics (returning
fewer judgments than candidates) are deliberately excluded. This keeps the
contract simple and unambiguous, and simplifies the next stage: the application
does not need to reconcile missing judgments or half-failed evaluations.
- **Alternative considered**: partial results with per-candidate failure
  markers. Rejected: it complicates the contract and the next stage's
  aggregation logic without demonstrated need; all-or-nothing is simpler and
  sufficient now.

### Decision: Classification input is `str`
The `score` method takes `text: str` as the classification input, consistent
with `LLMClient.classify(text: str, ...)`. The input to classification in this
codebase is the text to classify. Using `str` keeps the port small and
domain-oriented.
- **Alternative considered**: a dedicated `ClassificationInput` domain type.
  Rejected: no existing classification input type exists; introducing one now
  adds a concept the spec does not require. Can be introduced later if the
  scoring pipeline needs richer input.

### Decision: Duplicate candidate validation helper at the application boundary, in this stage
Duplicate candidates constitute an invalid request, not a scorer-specific
concern. A `validate_unique_candidates(candidates)` helper is provided in the
port module (`candidate_scorer.py`) and rejects duplicates by raising
`ValueError`. This is an input-contract validation, not an implementation
detail, so it is introduced in this stage rather than deferred. The helper uses
`Candidate` equality/hashing to detect duplicates. The concrete scorer is not
responsible for deduplication.
- **Alternative considered**: defer the helper to the concrete-scorer stage.
  Rejected: duplicate rejection is an input-contract concern, not an LLM
  implementation detail; deferring it blurs the boundary between invalid
  requests and scorer behavior.
- **Alternative considered**: a `CandidateSet` domain type that enforces
  uniqueness by construction. Rejected: a list plus a validation helper is
  simpler and sufficient; a `CandidateSet` adds a concept the spec does not
  require.

### Decision: Scoring semantics — evaluation, not decision
The scorer evaluates candidates independently and produces one judgment per
candidate. It does not decide the final classification, select a winning
candidate, rank candidates for selection, or aggregate judgments. Score
calibration, candidate ranking, candidate selection, and judgment aggregation
are all outside this change. This boundary is fixed in the spec to prevent the
next stage from turning `CandidateScorer` into a `Classifier` and mixing
evaluation with decision.
- **Alternative considered**: let the scorer also rank/select. Rejected: it
  would couple the scoring mechanism to the decision policy, making the decision
  untestable without LLM dependencies and violating the layer responsibility
  split.

### Decision: Port-boundary import-linter contract targets `application.ports`, not all of `application`
A new `application-ports-isolated` contract forbids `application.ports` from
importing `infrastructure` and `interfaces`, directly or indirectly. This
enforces that every port (including `CandidateScorer`) remains a pure
abstraction — the dependency-inversion boundary. It is green immediately because
all existing ports are pure `typing.Protocol` definitions. A broad
`application → infrastructure` forbidden contract is rejected because
application use cases currently import `infrastructure` directly (e.g.,
`run_baseline/runner.py`); that migration is a later stage.
- **Alternative considered**: forbid `application → infrastructure` broadly.
  Rejected: fails immediately against existing use-case imports.
- **Alternative considered**: a contract referencing a future
  `infrastructure.scoring` module. Rejected: the prep stage explicitly rejected
  forward-looking contracts for nonexistent modules.

### Decision: Public API exposure via existing `__init__.py` surfaces
`Candidate` and `Judgment` are re-exported from `domain/models/__init__.py`;
`CandidateScorer` and `validate_unique_candidates` are re-exported from
`application/ports/outbound/__init__.py`. This matches the existing pattern
(`Dataset`, `Record`, `LLMClient`, etc.). The `__init__.py` files remain API
surfaces only — no implementation logic.
- **Alternative considered**: new top-level `domain/__init__.py` exports.
  Rejected: `domain/__init__.py` is currently empty; the existing public surface
  is `domain/models/__init__.py` and `domain/errors/__init__.py`.

## Risks / Trade-offs

- **[Score semantics left open]** Treating the score as an unbounded opaque
  ranking signal means future code must not assume probability semantics.
  → Mitigation: the spec explicitly documents the score as an opaque ranking
  signal and forbids probability/confidence/calibrated-likelihood assumptions.
  Normalization and calibration ownership are deferred until the logits backend
  produces real score semantics.
- **[All-or-nothing may be too strict]** If one candidate out of many fails
  evaluation, the entire call fails and no judgments are returned.
  → Mitigation: this is intentional and simplifies the next stage. If
  partial-result semantics are later needed, the contract can be revisited
  without changing the domain types.
- **[Port-boundary contract is narrower than full dependency inversion]**
  `application-ports-isolated` only forbids `application.ports` →
  `infrastructure`, not `application` → `infrastructure` broadly. Use cases can
  still import infrastructure directly.
  → Mitigation: this reflects the current state of the codebase; broadening the
  contract is a later migration stage that moves use-case infrastructure imports
  behind ports. The contract added here is immediately green and enforces the
  boundary that matters for `CandidateScorer`.
- **[Flat file vs directory for the port]** The prep stage's design wrote the
  path with a trailing slash, suggesting a directory; this design uses a flat
  file.
  → Mitigation: the module path is identical; the flat file matches every
  existing port. If a port package later needs multiple modules, it can be
  promoted to a directory without changing import paths.

## Migration Plan

This stage introduces new domain and application contracts without modifying
existing runtime behavior. The additions are new domain modules, a new port
module with a validation helper, new tests, and one new import-linter contract.
Adding the import-linter contract is a change to boundary enforcement, not just
an additive runtime file, but no existing runtime behavior changes and no
rollback is needed. Subsequent stages, in order: (1) concrete `CandidateScorer`
implementation in `infrastructure` with logits extraction, (2) the scoring use
case, (3) `BaselineRunner` migration to the scoring pipeline, (4)
deprecation/removal of the generative `ClassificationResult` path. Each
subsequent stage preserves the layer boundaries fixed here and in the prep
stage.

## Open Questions

- **Scoring failure exception hierarchy**: the spec distinguishes invalid domain
  data, invalid scoring requests, failed evaluation, and infrastructure/model
  failures, but defers the exact exception hierarchy to the concrete-scorer
  stage. Deferrable — no concrete scorer is introduced here, so no exception
  type is needed yet.
- **`ClassificationInput` type**: whether the `text: str` input should later
  become a richer domain type. Deferrable — `str` is sufficient now and matches
  the existing `LLMClient.classify` convention; a richer type can be introduced
  when the scoring pipeline needs it without changing the port's semantics.
- **Score normalization and calibration ownership**: whether normalization or
  calibration policy belongs to `application` or `domain`. Deferred until the
  logits backend produces real score semantics that can be evaluated
  experimentally — fixing ownership now would architecturally lock in a
  decision that has not yet been validated.
