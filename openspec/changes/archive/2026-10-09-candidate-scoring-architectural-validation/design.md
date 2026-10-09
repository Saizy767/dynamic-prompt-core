# Design

## Context

The candidate-scoring migration is functionally complete. The architecture is:

- `CandidateScorer` (`application/ports/outbound/candidate_scorer.py`) is the
  outbound port — a `runtime_checkable Protocol` with
  `async def score(text, candidates) -> list[Judgment]`.
- `LLMLogitCandidateScorer` (`infrastructure/llm/scoring/candidate_scorer.py`) is
  the sole infrastructure implementation. It batches all candidates through one
  `forward_batch` call and translates `LLMScoringError`/`Exception` to
  `CandidateScoringError` at the port boundary (all-or-nothing, no partial
  results).
- `ClassificationPolicy` (`application/services/classification_policy.py`) is an
  application-internal strategy protocol. `ArgmaxClassificationPolicy` selects
  the highest-scoring candidate, breaking ties by original order (`>`, not
  `>=`).
- `Classification` (`domain/models/classification.py`) is the frozen domain
  result: `selected: Candidate` + `judgments: tuple[Judgment, ...]`.
- `ClassifyInput` (`application/use_cases/classify_input/`) composes
  `scorer.score` → `policy.classify`.
- `BaselineRunner` (`run_baseline/runner.py`) classifies per-example via
  `scorer.score` → `policy.classify` and maps
  `classification.selected.value` → `predicted_decision` (with `int()` cast and
  `None` fallback). On `CandidateScoringError`, it sets `classify_status =
  "failed"`.
- `RunCycleDeps` (`run_cycle_deps.py`) carries `candidate_scorer`,
  `classification_policy`, and `candidates: tuple[Candidate, ...]`.
- `steps.py` threads these to all three `BaselineRunner` construction sites
  (`run_active_on_dev`, `run_new_on_dev`, `run_holdout`).
- `build_candidate_scorer(model_path)` (`infrastructure/llm/scoring/factory.py`)
  is the shared composition-root factory, called by both `build_cli_deps` and
  the runner CLI `_main_async`.
- The generative classification path (`ClassificationResult`, `classify_detailed`,
  `classify_many`) is removed. Contract tests in
  `tests/contract/test_no_generative_classification.py` verify absence.

Existing test coverage is substantial:
- `tests/unit/candidate_scoring/` (12 files) covers domain models, the port,
  policy, `classify_input`, logit scorer, scoring seams, the LLM scorer contract,
  and batch scoring.
- `tests/unit/run_cycle/` (8 files + conftest) covers steps, queue, state,
  report, error handling, counters, and stop criteria. `conftest.py` already
  provides `FakeCandidateScorer` + real `ArgmaxClassificationPolicy` +
  `FAKE_CANDIDATES`. Three tests in `test_steps.py` verify each construction
  site threads scorer/policy/candidates, plus one identity test.
- `tests/unit/run_baseline/test_runner.py` covers `_process_example`
  classification, error mapping, and `ResultRow` fields.
- `tests/contract/test_no_generative_classification.py` covers symbol absence,
  port method absence, and no runtime fallback.

The remaining work is to fill gaps in cross-layer invariant coverage and add
architectural tests that import-linter cannot express. This is a **testing and
architectural-validation stage**, not a production architecture stage.

## Goals / Non-Goals

**Goals:**

- Make every cross-layer invariant from the migration executable through tests
  or static checks.
- Fill gaps in existing coverage: import-boundary tests for application →
  infrastructure scorer, candidate source-of-truth validation, scorer lifetime
  identity, metric-boundary validation, causal token alignment, candidate token
  boundaries, non-blocking async boundary, and legacy structured-generation
  dependency absence.
- Preserve and extend the no-generative-classification contract tests.
- Preserve all existing optimization regression coverage.
- Run the full static-analysis suite (`pytest`, `mypy`, `ruff check`,
  `lint-imports`).

**Non-Goals:**

- Introduce new production abstractions solely for testing.
- Change candidate-scoring behavior, optimization metrics, stopping conditions,
  rollback behavior, or reporting.
- Add optimization-specific test doubles to production code.
- Require network access or a real model for unit tests.
- Reintroduce the removed generative classification path for compatibility
  testing.
- Create a new top-level test framework solely for this migration.

## Decisions

### Decision: Test through the port, not the implementation

Application tests SHALL depend on `CandidateScorer` and use `FakeCandidateScorer`
(deterministic `Judgment` objects). This proves application behavior is
independent of the concrete model implementation. The existing
`FakeCandidateScorer` in `tests/unit/run_cycle/conftest.py` already follows this
pattern; tests in `tests/unit/candidate_scoring/` use `StubScorer` similarly.

- **Alternative considered**: use `LLMLogitCandidateScorer` with a stub model.
  Rejected: that couples application tests to infrastructure and would fail if
  the scorer implementation changes, even though the port contract is
  preserved.

### Decision: Use the real `ArgmaxClassificationPolicy` in application tests

The policy is a pure function with no external dependencies. Using the real
implementation exercises actual selection behavior (tie-breaking, max-score
selection) at no cost. A mock policy is used only when a test needs a
predetermined `Classification` result rather than testing classification itself.
The existing `conftest.py` already follows this pattern.

### Decision: Organize tests by existing repository structure

Tests SHALL follow the existing layout (`tests/unit/`, `tests/integration/`,
`tests/contract/`) rather than introducing a new generic architecture-testing
framework. New architectural tests go in `tests/contract/` or alongside existing
unit tests in the relevant subdirectory. This avoids a parallel test
organization that would diverge from production code structure.

### Decision: Architectural tests use AST and source inspection, without duplicating lint-imports

`lint-imports` already enforces layer-direction contracts (`domain-isolated`,
`application-isolated`, `application-ports-isolated`, `infrastructure-isolated`,
`no-cyclic-imports`). Architectural tests SHALL NOT duplicate those checks.
Symbol-level AST checks SHALL be added only where an existing import-linter
contract permits an import path that is still architecturally forbidden — for
example, if a future contract allowed `application` to import a specific
`infrastructure` module but not `LLMLogitCandidateScorer` by name.

The invariants that currently need AST or source inspection because
import-linter cannot express them:
- no `BaselineRunner` construction in `run_cycle` supplies a `candidates=`
  argument other than `deps.candidates` (a wiring invariant, not an import
  boundary);
- concrete scorer construction has one production source (both composition roots
  route through `build_candidate_scorer`; neither constructs
  `LLMLogitCandidateScorer` directly);
- removed generative symbols remaining absent (already partially covered by
  `test_no_generative_classification.py`).

These tests use `ast.walk` or `pathlib` source scanning, consistent with the
existing contract test style. If `lint-imports` already forbids
`application → infrastructure` entirely, no additional AST test for that
boundary is needed.

### Decision: Test candidate ordering and tie-breaking explicitly

`ArgmaxClassificationPolicy` resolves ties by original order (`>`, not `>=`).
Candidate ordering is part of the classification contract. Tests SHALL verify
`candidates[i] == judgments[i].candidate` and that equal scores resolve to the
first-in-order candidate. The existing `test_classification_policy.py` covers
tie-breaking; the gap is the end-to-end ordering preservation through the
scorer → policy path, which SHALL be added as an integration test.

### Decision: Validate `RunCycleDeps` wiring with object identity

Optimization-cycle tests SHALL verify `runner.scorer is deps.candidate_scorer`
and `runner.policy is deps.classification_policy` (identity, not equality). For
candidates, `BaselineRunner` accepts a `list` while `RunCycleDeps` stores a
`tuple`, so tests verify value/order equality (`list(deps.candidates) ==
passed_candidates`). The existing `test_steps.py` already has three construction
tests and one identity test; these SHALL be preserved and extended if gaps are
found.

### Decision: Separate runner result-mapping from metric-boundary integration

The `BaselineRunner` unit test SHALL verify that `classification.selected.value`
maps to `predicted_decision` (including the `int()` conversion and `None`
fallback semantics). This is a narrow unit test of the runner's result mapping.

The metric-boundary invariant — that `Judgment.score` magnitude does not
independently influence `compute_metrics` — crosses scorer → policy → runner →
metrics. This SHALL be an integration/application pipeline test, not a runner
unit test. The integration test constructs two scoring scenarios where the
selected candidate is the same but `Judgment.score` values differ, then verifies
the optimization metric (computed from `predicted_decision` vs `true_label`) is
unchanged. Separating these gives clearer failure diagnosis: a runner unit test
failure points at the mapping, while an integration test failure points at the
metric boundary.

### Decision: Test the LLM logit scorer with fake adapters

The infrastructure scorer tests SHALL use deterministic fake model/tokenizer
adapters (already established as `StubTokenizer`, `StubModel`, `FailingModel`,
`PartialFailingModel` in `test_llm_logit_candidate_scorer.py` and
`test_batch_scoring.py`). Tests SHALL verify:
- causal next-token alignment (`logit[prefix_last] → c0`, not `logit[c0] → c0`);
- multi-token mean log-probability;
- candidate token IDs matching model input positions (not independently
  tokenized and concatenated);
- batched scoring semantics (one judgment per candidate, same order);
- no partial results on failure;
- model failure propagation.

The existing `test_logit_scorer.py` covers causal alignment and multi-token
scoring; the gap is the candidate-token-boundary invariant (token IDs used for
scoring exactly occupy candidate continuation positions in the actual model
input), which SHALL be added.

### Decision: Test non-blocking async boundary with real scorer and blocking fake model

The test SHALL exercise the real `LLMLogitCandidateScorer` with deterministic
fake tokenizer/model adapters. The fake model adapter deliberately blocks on a
synchronization primitive (e.g., `threading.Event`) while a concurrent
`asyncio` task demonstrates event-loop progress. This proves the real
infrastructure boundary offloads blocking work correctly — a fake async scorer
using `await event.wait()` would be naturally non-blocking even if the real
scorer accidentally ran blocking inference directly on the event loop.

The specific executor mechanism (`asyncio.to_thread` in `LLMLogitCandidateScorer`)
is an infrastructure implementation detail and SHALL NOT be asserted. The test
focuses on the externally observable contract: a concurrent task makes progress
while scoring is awaited.

### Decision: Extend contract tests with concrete artifact targeting

The existing `tests/contract/test_no_generative_classification.py` SHALL remain.
It SHALL be extended to verify no legacy structured-generation classification
dependency remains. The test SHALL target concrete obsolete runtime artifacts
— specific symbols (`ClassificationResult`, `classify_detailed`, `classify_many`),
specific generated-output model types, specific classification JSON parsers or
helpers — rather than searching for vague textual concepts like `"decision"` or
`"output_contract"` globally. The rule is: ban obsolete runtime artifacts, not
vocabulary. Terms like `"classify"`, `"confidence"`, and `"classification"`
remain valid in the candidate-scoring architecture and SHALL NOT be broadly
forbidden. The test targets classification-specific legacy dependencies, not all
LLM generation infrastructure.

### Decision: Fix violations, not weaken tests

If an architectural test fails, the preferred response is to fix the production
dependency direction or wiring. The implementation SHALL NOT weaken the port,
introduce a generative fallback, expose logits to application code, use
`Judgment.score` as a metric, duplicate scorer construction, or mock away the
policy when its real behavior is the subject of the test.

### Decision: No infrastructure state escapes into domain or application

The migration's most important architectural outcome is that tensors, token IDs,
logits, attention masks, and model output objects stay inside infrastructure.
The test suite SHALL explicitly verify that `Judgment`, `Classification`,
`ResultRow`, and optimization state carry only domain/application-safe
information. No `torch.Tensor`, `token_ids`, `raw_logits`, `attention_mask`, or
model output object SHALL appear as a field on these types. This is verified by
inspecting the field types of these dataclasses (or equivalent structures) and
confirming they reference only `Candidate`, `Judgment`, `Classification`,
`str`, `int`, `float`, `bool`, `None`, or collections thereof.

### Decision: Explicitly test predicted_decision conversion semantics

`BaselineRunner` maps `classification.selected.value` to `predicted_decision`
via `int()` cast with `None` fallback on `ValueError`/`TypeError`. This is a
meaningful boundary with potentially hidden behavior. The test suite SHALL
explicitly verify:
- `Candidate("1")` → `predicted_decision == 1` (integer conversion succeeds);
- a non-integer selected value (e.g., `Candidate("yes")`) →
  `predicted_decision == None` (conversion fails gracefully).

This preserves the conversion semantics as an explicit contract rather than an
undocumented implementation detail.

## Risks / Trade-offs

- **[Discovering a violation during validation]** An architectural test may
  reveal a production dependency-direction violation not caught by existing
  tooling.
  → Mitigation: fix the production code rather than weakening the test. This is
  the intended outcome of the validation stage.

- **[Test duplication with import-linter]** Some import-boundary tests may
  overlap with `lint-imports` contracts.
  → Mitigation: architectural tests are added only where import-linter cannot
  express the invariant (wiring, symbol-level, factory usage, candidate
  source). Layer-direction checks already enforced by `lint-imports` are not
  duplicated.

- **[Candidate token boundary test fragility]** The candidate-token-boundary
  invariant (token IDs in model input match scoring positions) depends on
  tokenizer behavior that may vary across tokenizer implementations.
  → Mitigation: test with the existing `StubTokenizer` which controls token IDs
  deterministically. The test verifies the scorer's contract, not a specific
  tokenizer's behavior.

- **[Non-blocking async test timing sensitivity]** The non-blocking boundary
  test relies on concurrent task progress while the real scorer awaits a
  deliberately blocking fake model adapter.
  → Mitigation: use the real `LLMLogitCandidateScorer` with deterministic fake
  tokenizer/model adapters. The fake model blocks on a `threading.Event` while
  a concurrent `asyncio` task demonstrates event-loop progress. The test is
  deterministic and does not depend on real model inference timing.

## Migration Plan

1. Inventory existing tests covering classification, `BaselineRunner`, and
   `run_cycle` to identify coverage gaps.
2. Add/extend deterministic fake scorer fixtures where needed.
3. Add classification composition tests using the real
   `ArgmaxClassificationPolicy` (ordering, tie-breaking, candidate identity).
4. Add/extend `BaselineRunner` unit tests for result mapping
   (`classification.selected.value` → `predicted_decision`, including `int()`
   conversion and `None` fallback) and error-status mapping.
5. Add an integration test for the metric-boundary invariant (score magnitude
   does not independently influence `compute_metrics`).
6. Add/extend optimization-cycle dependency-wiring and scorer-lifetime tests.
7. Add candidate-source-of-truth validation targeting `BaselineRunner`
   construction calls specifically.
8. Add/complete LLM scorer unit/contract tests (causal alignment, token
   boundaries, multi-token, batched semantics).
9. Add batch-scoring failure-boundary and error-propagation tests.
10. Add non-blocking async boundary test using real `LLMLogitCandidateScorer`
    with a deliberately blocking fake model adapter.
11. Add no-infrastructure-state-escapes validation (`Judgment`, `Classification`,
    `ResultRow`, optimization state carry no tensors/logits/token IDs).
12. Preserve and extend the no-generative-classification contract tests with
    concrete artifact targeting and add legacy structured-generation dependency
    validation.
13. Add symbol-level AST checks only where import-linter permits an import path
    that is still architecturally forbidden.
14. Run the complete test and static-analysis suite.
15. Fix architectural violations rather than weakening tests.
16. Confirm that no production behavior outside the migration changed.

No rollback strategy is required — this stage adds tests and architectural
checks. If a test reveals a production violation, the fix is a production code
change that corrects the dependency direction or wiring.

## Acceptance Criteria

- [ ] Existing candidate-scoring tests remain green.
- [ ] End-to-end scorer → policy ordering preservation is tested.
- [ ] Candidate continuation-token boundary is tested.
- [ ] Real `LLMLogitCandidateScorer` is verified not to block the event loop
      using blocking fake model adapters.
- [ ] All optimization `BaselineRunner` constructions receive the same
      scorer/policy instances.
- [ ] Every optimization runner receives candidates derived from
      `RunCycleDeps`.
- [ ] `BaselineRunner` unit test verifies `classification.selected.value` →
      `predicted_decision`, including `int()` conversion and `None` fallback.
- [ ] Integration test proves raw score magnitude is not consumed directly as a
      metric.
- [ ] No production application module imports concrete scoring infrastructure
      (enforced by `lint-imports`; supplemented by AST only where needed).
- [ ] Concrete scorer construction has one production composition path.
- [ ] Removed generative classification symbols remain absent.
- [ ] No tensors/logits/token IDs/attention masks escape infrastructure into
      domain, application, or optimization state.
- [ ] `pytest`, `mypy`, `ruff check`, and `lint-imports` pass.

## Open Questions

(None — the testing strategy, test organization, and invariant set are fully
determined by the migration's architectural boundaries and the existing test
conventions.)
