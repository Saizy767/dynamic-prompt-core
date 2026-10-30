# Design

## Context

The layered architecture is in place: `domain`, `application`,
`infrastructure`, `interfaces` under `src/dynamic_prompt_core/`, with
import-linter contracts in `pyproject.toml` enforcing layer direction, domain
isolation, and infrastructure restrictions. Classification today flows through
`BaselineRunner` (`application/use_cases/run_baseline/runner.py`), which calls
`AsyncTask.classify_detailed` (`infrastructure/llm/client.py`) to get a
generative structured `ClassificationResult` (`application/schemas/classification.py`,
Pydantic: `decision` + `confidence`). The `LLMClient` port
(`application/ports/outbound/llm_client.py`) abstracts the LLM but its contract
is built around generative structured output (`classify` → `T | None` parsed
Pydantic model). `run_cycle` orchestrates the optimization loop via
`RunCycleDeps`, a typed dependency object holding outbound ports.

The candidate-based scoring migration replaces generative classification with a
`CandidateScorer` that scores discrete candidates from model internals (logits,
tokenization). This stage prepares the architecture so later stages can introduce
`Candidate`, `Judgment`, and `CandidateScorer` without further boundary changes.

## Goals / Non-Goals

**Goals:**
- Fix the `CandidateScorer` port location and **semantics** (not the final
  Python signature) at the application/infrastructure boundary.
- Fix the port's information boundary: it hides logits, token IDs, tokenizer,
  and model APIs.
- Strengthen domain isolation via a **stdlib-only allowlist policy** (not a
  growing blacklist of LLM libraries) and add an import-linter contract that
  enforces it.
- Fix the responsibility split: `infrastructure` owns model inference mechanics;
  `application`/`domain` own aggregation, normalization policy, and the
  classification decision.
- Define the component disposition (remain / migrate / deprecate) for existing
  classification code.
- Define the migration boundary: subsequent stages preserve the established
  layer boundaries but need not be purely additive.
- Preserve existing classification behavior — no runtime behavior changes.

**Non-Goals:**
- Implement `Candidate`, `Judgment`, or `CandidateScorer` (subsequent stages).
- Implement logits extraction, batching, or the scoring pipeline.
- Change classification behavior or outputs.
- Introduce provider-specific abstractions into `domain`.
- Add the `CandidateScorer` port-boundary import-linter contract (added with
  the port in a later stage).
- Refactor unrelated code.

## Decisions

### Decision: `CandidateScorer` is a `typing.Protocol` in `application/ports/outbound/candidate_scorer/`
The port follows the existing port convention (`LLMClient`, `TeacherLLMClient`,
`Normalizer`, etc. are `typing.Protocol` in `application/ports/outbound/`).
`CandidateScorer` lives alongside them. The application layer depends on the
port; `infrastructure` provides the concrete scorer. Stage 1 fixes the port
**semantics** (scores candidates, returns judgments) but not the final Python
signature — the signature is settled when `Candidate` and `Judgment` are
concrete in a later stage. The port SHALL NOT expose logits, token IDs,
tokenizer APIs, or model APIs; those are infrastructure internals behind the
port boundary. No `.py` file is created in this stage.
- **Alternative considered**: define `CandidateScorer` as an `abc.ABC`.
  Rejected: the `architecture` spec defaults to `typing.Protocol` for new ports;
  no runtime inheritance is required.
- **Alternative considered**: place the port under `application/ports/inbound/`.
  Rejected: the scorer is an outbound dependency (the application calls out to
  it), not an inbound entry point.

### Decision: Domain isolation via stdlib-only allowlist, not LLM-library blacklist
The existing `domain-isolated` contract forbids `domain` from importing
`application`, `infrastructure`, `interfaces`. For LLM internals, instead of
maintaining a growing blacklist of forbidden LLM libraries (`transformers`,
`aiohttp`, `torch`, `sentence_transformers`, …), `domain` SHALL depend only on
the standard library plus an explicit allowlist of permitted third-party modules
(currently empty). Any third-party import not on the allowlist is a violation.
This is forward-compatible: new LLM libraries are automatically forbidden without
adding them to a blacklist. The allowlist is the single place to grant an
exception.
- **Alternative considered**: a blacklist of known LLM libraries. Rejected: it
  grows with every new library and silently admits anything not yet listed.
- **Alternative considered**: a single combined contract with the existing
  layer-direction check. Rejected: separating layer-direction from the
  stdlib-only policy keeps each contract focused and produces clearer failure
  messages.

### Decision: Infrastructure owns model inference mechanics; application/domain own the decision
`infrastructure` owns model inference mechanics: logits extraction, tokenization,
batching, and provider adapters. `application` and `domain` own aggregation,
normalization policy, and the classification decision. This keeps the decision
logic testable without LLM dependencies and keeps provider specifics behind the
port. The existing `infrastructure-isolated` contract already prevents
`infrastructure` from importing `application.use_cases` or `interfaces`, so no
new contract is needed for that direction.
- **Alternative considered**: put aggregation in `infrastructure`. Rejected: it
  would make the classification decision untestable without LLM dependencies and
  violate the domain/application ownership of decision logic.

### Decision: `ClassificationResult` generative path is legacy/deprecated, not removed
The generative `ClassificationResult` implementation and `LLMClient.classify`
path are marked legacy/deprecated in the component disposition. The
**concept** of a classification result is preserved; only the generative
implementation is replaced. They are not removed in this stage — removal happens
when the scoring pipeline replaces the generative path in a later stage.
`LLMClient` itself remains for non-classification calls (thesis extraction).
- **Alternative considered**: remove `ClassificationResult` now. Rejected: it
  would change runtime behavior, violating the preservation requirement.

### Decision: No `Candidate`/`Judgment` domain types created in this stage
The spec defines that `Candidate` and `Judgment` SHALL be domain objects, but
creating them now is an implementation step. This stage fixes where they will
live (`domain`) and what they must not depend on, without creating the files.
- **Alternative considered**: create empty `Candidate`/`Judgment` stubs now.
  Rejected: stubs with no behavior add noise and blur the planning/implementation
  boundary.

### Decision: import-linter validates existing boundaries only; port contract added with the port
This stage adds the domain stdlib-only allowlist contract to `pyproject.toml`,
enforceable immediately against existing code. The `CandidateScorer` port-boundary
contract is **not** added now — it is added together with the port in the stage
that creates it, so import-linter never references a not-yet-existing module.
- **Alternative considered**: add a forward-compatible port contract now.
  Rejected: a contract referencing a non-existent module is inert and clutters
  the configuration; adding it with the port keeps contracts grounded in real
  modules.

## Risks / Trade-offs

- **[Port semantics fixed before signature]** The final `CandidateScorer`
  Python signature may shift when `Candidate`/`Judgment` are concrete.
  → Mitigation: Stage 1 fixes semantics (scores candidates, returns judgments)
  and the information boundary (no logits/token IDs/tokenizer/model APIs), both
  stable; the signature is intentionally left open.
- **[Allowlist too strict]** A stdlib-only allowlist may reject a legitimate
  `domain` dependency.
  → Mitigation: the allowlist is the single, explicit place to grant an
  exception; additions are deliberate and visible.
- **[Component disposition may shift]** Whether `BaselineRunner` migrates or is
  replaced depends on later-stage findings.
  → Mitigation: the disposition is recorded as intent in the spec; later stages
  can revise the delta if needed.

## Migration Plan

This stage is planning-only: spec deltas + the domain stdlib-only allowlist
import-linter contract. No production code changes, no rollback needed.
Subsequent stages introduce, in order: (1) `Candidate` and `Judgment` domain
types, (2) the `CandidateScorer` port + its import-linter boundary contract +
infrastructure implementation with logits extraction, (3) the scoring use case,
(4) `BaselineRunner` migration to the scoring pipeline, (5) deprecation/removal
of the generative `ClassificationResult` path. Each subsequent stage preserves
the layer boundaries fixed here but need not be purely additive.

## Open Questions

- **Scoring subpackage placement**: should the infrastructure scorer live in
  `infrastructure/llm/` (alongside `AsyncTask`) or a new
  `infrastructure/scoring/`? Deferrable — decided when the implementation is
  written; the boundary contract is the same either way.
- **Allowlist enforcement mechanism**: import-linter's `forbidden` type is
  blacklist-oriented; a stdlib-only allowlist may need a custom check or a
  `forbidden` contract listing non-allowlisted third-party roots. Deferrable —
  the policy is fixed in the spec; the exact tool mechanism is settled in
  implementation.
