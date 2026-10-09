# Proposal

## Why

The candidate-scoring migration is functionally complete: `CandidateScorer` is
the outbound port, `LLMLogitCandidateScorer` is the infrastructure implementation,
`ArgmaxClassificationPolicy` selects from `Judgment` objects, `BaselineRunner`
classifies through scoring, and `RunCycleDeps` wires the scorer/policy/candidates
through all three runner construction sites via the shared `build_candidate_scorer`
factory. Existing tests cover individual components, but the migration introduces
cross-layer invariants — port isolation, candidate source of truth, scorer
lifetime, no generative fallback, metric boundaries — that are not yet fully
protected by executable tests and architectural checks. This change makes those
invariants executable so future changes cannot silently reintroduce the
generative classification architecture.

## What Changes

- Add architectural tests verifying application code depends on the
  `CandidateScorer` port rather than `LLMLogitCandidateScorer`. Layer-direction
  boundaries already enforced by `lint-imports` are not duplicated; symbol-level
  AST checks are added only where import-linter permits an import path that is
  still architecturally forbidden.
- Add architectural tests verifying concrete scorer construction has one
  production source (both composition paths route through
  `build_candidate_scorer`), and that the scorer is constructed once per process
  and reused across all optimization rounds.
- Add architectural tests verifying the candidate set has one source of truth
  (`RunCycleDeps.candidates`): no `BaselineRunner` construction in `run_cycle`
  supplies a `candidates=` argument other than `deps.candidates`.
- Add `BaselineRunner` unit tests verifying `Classification.selected.value`
  maps to `predicted_decision`, including the `int()` conversion and `None`
  fallback semantics.
- Add an integration test verifying the metric-boundary invariant: changing
  `Judgment.score` without changing the selected candidate does not change the
  optimization metric.
- Add/extend optimization-cycle tests verifying all three `BaselineRunner`
  construction paths receive the same scorer/policy instances (object identity)
  and the same candidate ordering.
- Add/extend LLM logit scorer tests verifying causal next-token alignment,
  multi-token mean log-probability, candidate token boundaries in the actual
  model input, and batched scoring semantics (one judgment per candidate, no
  partial results on failure).
- Add/extend error-boundary tests verifying scoring failures propagate as
  `CandidateScoringError` and map to `classify_status = "failed"` without
  producing fake classifications.
- Preserve and extend the no-generative-classification contract tests
  (`ClassificationResult`, `classify_detailed`, `classify_many` remain absent)
  and add legacy structured-generation dependency validation targeting concrete
  obsolete artifacts, not vocabulary.
- Add a non-blocking async boundary test using the real
  `LLMLogitCandidateScorer` with a deliberately blocking fake model adapter,
  verifying a concurrent `asyncio` task makes progress while scoring is awaited.
- Add a no-infrastructure-state-escapes validation verifying that `Judgment`,
  `Classification`, `ResultRow`, and optimization state carry no tensors, token
  IDs, logits, attention masks, or model output objects.
- Run the full static-analysis suite (`pytest`, `mypy`, `ruff check`,
  `lint-imports`) and fix architectural violations rather than weakening tests.

## Capabilities

### New Capabilities

- `candidate-scoring-validation`: Executable test and architectural-validation
  requirements that enforce the candidate-scoring migration's cross-layer
  invariants — port isolation, policy ownership, candidate source of truth,
  scorer lifetime, metric boundaries, predicted-decision conversion semantics,
  no generative fallback, candidate ordering, no partial scoring results, no
  infrastructure state escapes, non-blocking async boundary, and
  import-boundary enforcement.

### Modified Capabilities

(None — no existing spec-level behavior changes. The validation requirements are
new; existing capabilities' production behavior is unchanged.)

## Impact

- **Code**: New and extended test files under `tests/unit/`, `tests/integration/`,
  and `tests/contract/`. No production code changes unless an architectural
  violation is discovered, in which case the production dependency direction or
  wiring is fixed rather than the test weakened.
- **Specs**: New `candidate-scoring-validation` delta spec defining the
  executable invariant requirements.
- **Dependencies**: No new runtime or dev dependencies. Tests use existing
  fakes/stubs (`FakeCandidateScorer`, stub model/tokenizer adapters) already
  established in the test suite.
- **Public API**: No change. The test suite is not a public API.
- **Migration**: No production migration. This is a validation stage that
  confirms the completed migration is architecturally complete, not just
  functionally correct.
