# Proposal

## Why

The candidate-scoring migration replaces generative LLM classification with
per-candidate scoring, but the domain currently has no representation of a
classification candidate or its evaluation result, and no application port
through which orchestration can request judgments without coupling to an LLM
implementation. The architecture preparation stage fixed the layer boundaries
and port location; this stage introduces the concrete domain contracts
(`Candidate`, `Judgment`) and the `CandidateScorer` outbound port so subsequent
stages can implement a scorer and migrate `BaselineRunner` without further
boundary changes. Introducing these contracts now keeps the scoring mechanism
replaceable behind a single dependency-inversion boundary.

## What Changes

- Introduce `Candidate` as an immutable domain model representing one
  classification hypothesis, with a single string-backed semantic value stored
  verbatim (no auto-normalization: no `strip`/`lower`/`casefold`), stable
  identity (equality, hashing, deterministic comparison), and no
  LLM/infrastructure dependencies.
- Introduce `Judgment` as an immutable domain model associating one `Candidate`
  with a validated numeric score. The score is an opaque ranking signal where
  higher means stronger support — not a probability, confidence, or calibrated
  likelihood. Invalid floats (`NaN`, `+inf`, `-inf`) are rejected; scores are
  not silently clamped and no `[0, 1]` constraint is imposed.
- Introduce `CandidateScorer` as an application outbound port
  (`application/ports/outbound/candidate_scorer`) defined as a
  `typing.Protocol`. The port accepts a classification input plus a collection
  of candidates and returns `Judgment` objects. It is asynchronous (matching
  existing outbound ports), supports multiple candidates per invocation, and
  exposes no logits, token IDs, tokenizers, model objects, or provider types.
  The contract is all-or-nothing: a successful call returns exactly one judgment
  per candidate; if any candidate cannot be evaluated, the call raises an
  exception (no partial results).
- Fix the evaluation-vs-decision boundary: the scorer evaluates candidates
  independently and does not decide, rank, select, or aggregate. Score
  calibration, candidate ranking, candidate selection, and judgment aggregation
  are all outside this change.
- Provide a `validate_unique_candidates` helper at the application boundary that
  rejects duplicate candidates before the scorer is invoked. Duplicate candidates
  are an invalid request, not a scorer-specific concern.
- Add an import-linter contract validating the `CandidateScorer` port boundary
  (`application.ports` must not import `infrastructure` or `interfaces`, directly
  or indirectly) together with the port module, per the architecture preparation
  stage.
- Expose the new domain types and port through the established public
  `__init__.py` API surfaces only.
- Preserve the existing generative classification pipeline unchanged:
  `BaselineRunner`, `AsyncTask.classify_detailed`, `LLMClient.classify`,
  `ClassificationResult`, `run_cycle`, and existing tests are not modified.

## Capabilities

### New Capabilities
- `candidate-scoring-contracts`: Domain model for candidate-based scoring
  (`Candidate`, `Judgment` with score validation) and the `CandidateScorer`
  application outbound port contract through which the application requests
  candidate judgments without depending on an LLM implementation.

### Modified Capabilities
<!-- None. Existing classification behavior is preserved; the architecture
     boundary requirements for the port are already specified by the
     candidate-scoring-architecture-prep change. This change introduces new
     concepts and fulfills the prep stage's "port contract added with port"
     requirement as an implementation task, not a spec-level modification. -->

## Impact

- **Code**: New domain modules under `src/dynamic_prompt_core/domain/models/`
  for `Candidate` and `Judgment`; a new application port module
  `src/dynamic_prompt_core/application/ports/outbound/candidate_scorer.py`
  containing the `CandidateScorer` protocol and the
  `validate_unique_candidates` helper; new tests for the domain contracts and
  port; a new import-linter contract in `pyproject.toml` for the port boundary.
  No existing `src/` files are modified.
- **APIs**: New public domain types (`Candidate`, `Judgment`) and a new
  application port (`CandidateScorer`) are exposed through their packages'
  `__init__.py` surfaces. No existing public API changes.
- **Dependencies**: No new runtime dependencies. The domain remains stdlib-only.
  `import-linter` configuration gains the port-boundary contract.
- **Specs**: A new `candidate-scoring-contracts` capability spec is introduced.
- **Migration**: Establishes the stable domain contracts and dependency-inversion
  boundary for subsequent stages that implement a concrete scorer, introduce
  logits-based inference, and migrate `BaselineRunner` — all without changing
  the domain contracts or application dependency boundary.
