# Proposal

## Why

The candidate-scoring contracts (`Candidate`, `Judgment`, `CandidateScorer` port)
are in place, but there is no application flow that turns judgments into a
classification decision, and no concrete scorer that derives scores from a model.
Classification still flows through the generative path
(`LLMClient.classify` → `ClassificationResult`). The migration needs the next
architectural step: an application-level classification flow that consumes
`Judgment` objects, and an infrastructure `CandidateScorer` backed by LLM logits,
so the LLM evaluates candidates while the application decides what those
evaluations mean — without leaking tokenizer, model, or logit internals across
the port boundary.

## What Changes

- Introduce a `ClassificationPolicy` application-internal strategy abstraction
  (in `application/services/`, not `application/ports/`) that consumes
  `list[Judgment]` and produces a `Classification` decision. The policy owns
  selection (argmax for the initial implementation), threshold/tie handling, and
  conversion to the application's classification representation. It performs no
  model inference and is not an outbound port or external dependency.
- Introduce a `Classification` domain model representing the application-level
  classification decision (the selected candidate plus the judgments it was
  derived from, stored as an immutable `tuple[Judgment, ...]`). Distinct from
  the legacy generative `ClassificationResult` schema, which remains unchanged.
- Introduce a `ClassifyInput` application use case that orchestrates:
  `validate_unique_candidates` → `CandidateScorer.score` →
  `ClassificationPolicy.classify`. The use case owns orchestration only; it does
  not inspect model outputs or know the scoring mechanism.
- Implement `LLMLogitCandidateScorer`, a concrete `CandidateScorer` in
  infrastructure that derives candidate scores from model logits. It owns prompt
  construction, tokenizer/model invocation, logit extraction, the
  logit-to-score algorithm (mean candidate-token log-probability with formal
  causal logit alignment), multi-token candidate handling, inference-mode
  enforcement, and error translation — all behind the port boundary.
- Introduce narrow infrastructure-facing seams (`CausalLanguageModel`,
  tokenizer adapter, prompt builder, logit scorer) so the scoring algorithm is
  testable with deterministic doubles and decoupled from a concrete model
  library.
- Introduce an application-facing `CandidateScoringError` in
  `application/errors/` and an infrastructure-internal `LLMScoringError` in
  `infrastructure/llm/scoring/errors.py`; the scorer translates infrastructure
  failures into `CandidateScoringError` so the application never imports
  infrastructure error types.
- Remove the `candidate-scoring-contracts` requirement "No concrete scorer or
  inference introduced" — that was a deferral marker for the previous stage;
  this change introduces the concrete scorer and classification flow.
- Preserve the existing generative classification path (`LLMClient.classify`,
  `AsyncTask.classify_detailed`, `ClassificationResult`, `BaselineRunner`).
  The new scoring path is additive; no existing caller is migrated or wired to
  the new flow.

## Capabilities

### New Capabilities
- `candidate-scoring-flow`: Application-level scoring and classification flow —
  `ClassificationPolicy` (consumes judgments, produces a classification
  decision), `Classification` domain model, `ClassifyInput` use case
  orchestrating validation → scoring → classification, and the
  evaluation-vs-decision separation that keeps the scorer from becoming a
  classifier.
- `llm-logit-scorer`: Infrastructure `CandidateScorer` implementation backed by
  LLM logits — prompt construction, tokenizer/model seams, logit extraction,
  logit-to-score algorithm with multi-token candidate handling, inference-mode
  enforcement, numerical validation, and error translation, all contained inside
  infrastructure.

### Modified Capabilities
- `candidate-scoring-contracts`: Remove the "No concrete scorer or inference
  introduced" requirement. That prohibition was a deferral marker for the
  contract-introduction stage; this change introduces the concrete scorer and
  the classification flow, so the prohibition no longer holds. All other
  contract requirements (port location, information boundary, scoring semantics,
  domain isolation, dependency direction, existing-path preservation) remain
  unchanged.

## Impact

- **Code**: New domain model (`Classification`); new application components
  (`ClassificationPolicy` service, `ClassifyInput` use case, `CandidateScoringError`);
  new infrastructure (`LLMLogitCandidateScorer` plus tokenizer/model/logit-scoring
  seams and `LLMScoringError`); new tests. No existing production code is modified
  and no existing caller is wired to the new flow — the generative path remains
  intact and is the only production classification path.
- **Specs**: Two new capability specs (`candidate-scoring-flow`,
  `llm-logit-scorer`); one modified spec (`candidate-scoring-contracts`) removing
  the deferral requirement.
- **Dependencies**: No new runtime dependencies. `transformers` and `torch` (if
  used by the model backend) remain infrastructure-only and are already project
  dependencies. No domain or application module gains an inference-library
  import.
- **Public API**: Additive. The new scoring/classification flow is introduced
  alongside the legacy generative path. No existing public API is changed.
- **Migration**: Establishes the first concrete scoring + classification flow,
  implemented and independently testable but not wired into any existing
  production caller. Caller-by-caller migration of existing generative
  classification callers is a separate follow-up, explicitly out of scope.
