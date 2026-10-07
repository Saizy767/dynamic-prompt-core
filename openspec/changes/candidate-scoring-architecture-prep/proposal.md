# Proposal

## Why

Classification today assumes the LLM produces a structured `ClassificationResult`
(`decision` + `confidence`) via generative output. The candidate-based scoring
migration replaces that assumption with a `CandidateScorer` that scores discrete
candidates from model internals (logits, tokenization). Before any scoring code
is written, the architecture must fix the dependency boundaries for the new flow
so that `domain` never touches LLM providers, tokenizers, logits, inference
engines, or transport protocols, and `infrastructure` owns all LLM-specific
details. Without this preparation, later stages would leak scoring internals
into `domain` or `application` and the import-linter contracts could not catch it.

## What Changes

- Define the target architecture for candidate-based scoring as a delta over the
  existing layered architecture: the scoring flow shape, layer responsibilities,
  and the `CandidateScorer` port location.
- Establish `CandidateScorer` as an application-level outbound port
  (`application/ports/outbound/candidate_scorer`) that the application layer
  depends on; infrastructure provides the concrete scorer. Stage 1 fixes the
  port **semantics** (scores candidates, returns judgments) but not the final
  Python signature. The port SHALL NOT expose logits, token IDs, tokenizer, or
  model APIs.
- Strengthen domain isolation via a **stdlib-only allowlist policy**: `domain`
  SHALL depend only on the standard library plus an explicit allowlist — not a
  growing blacklist of LLM libraries. LLM providers, tokenizers, logits,
  inference engines, and transport protocols are not on the allowlist.
- Require `infrastructure` to own **model inference mechanics** (logits
  extraction, tokenization, batching, provider adapters).
- Require `application` and `domain` to own **aggregation, normalization
  policy, and the classification decision**; `application` depends only on
  `domain` objects and `application` ports — no direct infrastructure imports
  for scoring.
- Remove the architectural assumption that classification results must be
  generated as structured LLM output: the `CandidateScorer` port contract is
  defined without prescribing generative output. The **concept** of a
  classification result is preserved; only the generative `ClassificationResult`
  implementation is replaced.
- Define the component disposition: `Candidate`/`Judgment` → `domain`;
  `CandidateScorer` → `application/ports/outbound`; `LLMClient` **remains**;
  generative classification methods become **legacy/deprecated**; `BaselineRunner`
  disposition is **MIGRATE**.
- Define the migration boundary: subsequent stages SHALL preserve the
  established layer boundaries but need not be purely additive.
- Add import-linter contracts validating **existing** dependency boundaries
  (domain stdlib-only allowlist). The import-linter contract for the
  not-yet-existing `CandidateScorer` port is added **together with the port** in
  a later stage, not now.
- Preserve existing public classification behavior — Stage 1 does not change
  runtime behavior.

## Capabilities

### New Capabilities
- `candidate-scoring-architecture`: Target architecture for candidate-based
  scoring — the scoring flow shape, `CandidateScorer` port contract, layer
  responsibilities in the new classification flow, component disposition
  (remain/migrate/deprecate), and the migration boundary for incremental
  introduction of `Candidate`, `Judgment`, and `CandidateScorer`.

### Modified Capabilities
- `architecture`: Add candidate-scoring-specific dependency boundary
  requirements — domain isolation via a stdlib-only allowlist policy (not a
  growing blacklist of LLM libraries), infrastructure ownership of model
  inference mechanics, and the `CandidateScorer` port as an application-level
  dependency boundary. Import-linter validation covers existing boundaries now;
  the port-boundary contract is added with the port in a later stage.

## Impact

- **Code**: No production code changes in this stage. The change defines
  architectural contracts and adds import-linter contracts to `pyproject.toml`
  that validate existing boundaries (domain stdlib-only allowlist). Existing
  classification behavior is preserved — runtime behavior is unchanged.
- **Specs**: `architecture` gains candidate-scoring boundary requirements; a
  new `candidate-scoring-architecture` capability spec is introduced.
- **Dependencies**: No new runtime dependencies. `import-linter` configuration
  in `pyproject.toml` gains a contract for domain stdlib-only allowlist
  enforcement. The `CandidateScorer` port-boundary contract is deferred to the
  stage that creates the port.
- **Public API**: Unchanged. `CandidateScorer` is defined as a port contract
  in the spec but not implemented in this stage.
- **Migration**: Establishes the stable architectural foundation for
  subsequent stages that introduce `Candidate`, `Judgment`, and
  `CandidateScorer` implementations, logits extraction, batching, and the
  scoring pipeline. Subsequent stages preserve the layer boundaries fixed here
  but need not be purely additive.
