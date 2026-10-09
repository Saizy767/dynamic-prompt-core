# Tasks

## 1. Inventory existing test coverage

- [x] 1.1 Review `tests/unit/candidate_scoring/` (12 files), `tests/unit/run_cycle/`
  (8 files + conftest), `tests/unit/run_baseline/test_runner.py`, and
  `tests/contract/test_no_generative_classification.py`. Document which
  invariants from `design.md` are already covered and which are gaps. Verify by
  producing a written gap list mapping each acceptance criterion to existing
  test(s) or "gap".

## 2. Classification composition tests

- [x] 2.1 Add or extend tests in `tests/unit/candidate_scoring/test_classify_input.py`
  verifying the end-to-end classification path: `text → scorer.score() →
  Judgment[] → policy.classify() → Classification` using a deterministic fake
  scorer. Verify the selected candidate is derived from candidate scores, not
  generated text. Run
  `pytest tests/unit/candidate_scoring/test_classify_input.py -x --tb=short`.
- [x] 2.2 Add or extend tests in
  `tests/unit/candidate_scoring/test_classification_policy.py` verifying
  candidate ordering preservation: `candidates[i] == judgments[i].candidate`
  holds for all candidates after scoring and classification. Verify the test
  passes.
- [x] 2.3 Add or extend an integration test in `tests/integration/` exercising
  the full classification path with a fake scorer where candidates `["0", "1"]`
  receive scores `0.2` and `0.8` respectively, and the expected classification
  selects `Candidate("1")`. Verify no generated JSON response is involved. Run
  `pytest tests/integration/ -x --tb=short`.
- [x] 2.4 Add a test verifying that application classification tests do not
  import `transformers`, PyTorch, tokenizer adapters, logit adapters, or
  `LLMLogitCandidateScorer`. Use `ast.walk` or `importlib` inspection on the
  test module. Verify the test passes.
- [x] 2.5 Add an integration test in `tests/integration/` verifying the
  metric-boundary invariant: changing `Judgment.score` values without changing
  the selected candidate does not change `predicted_decision` or the resulting
  classification metric. Construct two scoring scenarios with the same
  selection but different scores and assert metric equality. This crosses
  scorer → policy → runner → metrics, so it is an integration test, not a
  runner unit test. Verify by running
  `pytest tests/integration/ -k "metric" -x --tb=short`.

## 3. BaselineRunner result-mapping and error-boundary tests

- [x] 3.1 Add or extend tests in `tests/unit/run_baseline/test_runner.py`
  verifying that `Classification.selected.value` maps to
  `predicted_decision` and that `predicted_decision` is not derived from raw
  logits, `Judgment.score`, generated text, a confidence field, or a legacy
  `ClassificationResult`. Verify by running
  `pytest tests/unit/run_baseline/test_runner.py -x --tb=short`.
- [x] 3.2 Add or extend tests in `tests/unit/run_baseline/test_runner.py`
  explicitly verifying the `predicted_decision` conversion semantics:
  `Candidate("1")` → `predicted_decision == 1` (integer conversion succeeds),
  and a non-integer value such as `Candidate("yes")` →
  `predicted_decision == None` (conversion fails gracefully). Verify the test
  passes.
- [x] 3.3 Add or extend a test verifying that `BaselineRunner` maps
  `CandidateScoringError` to `classify_status = "failed"` with
  `predicted_decision = None` and `selected_candidate = None`, and that no
  fake classification is produced. Verify by running
  `pytest tests/unit/run_baseline/test_runner.py -k "fail" -x --tb=short`.
- [x] 3.4 Add or extend a test verifying that `ResultRow` does not contain
  `confidence` or `raw_classify` fields and that `judgment_scores` is populated
  from `Classification.judgments`. Verify the test passes.

## 4. Optimization-cycle wiring and scorer-lifetime tests

- [x] 4.1 Verify or extend tests in `tests/unit/run_cycle/test_steps.py` that
  all three `BaselineRunner` construction sites (`run_active_on_dev`,
  `run_new_on_dev`, `run_holdout`) pass `deps.candidate_scorer`,
  `deps.classification_policy`, and `list(deps.candidates)`. Use the existing
  `_CapturingRunner` pattern. Verify by running
  `pytest tests/unit/run_cycle/test_steps.py -x --tb=short`.
- [x] 4.2 Verify or extend the test in `tests/unit/run_cycle/test_steps.py`
  confirming all three runners from a single round share scorer and policy
  object identity (`runner.scorer is deps.candidate_scorer`,
  `runner.policy is deps.classification_policy`) and receive the same candidate
  ordering (`list(deps.candidates)`). Verify the test passes.
- [x] 4.3 Add a test verifying `RunCycleDeps` requires `candidate_scorer`,
  `classification_policy`, and `candidates` (construction without them raises
  `TypeError`). Verify by running
  `pytest tests/unit/run_cycle/ -k "deps" -x --tb=short`.
- [x] 4.4 Add a test verifying the candidate source-of-truth invariant: no
  `BaselineRunner` construction in `run_cycle` supplies a `candidates=` argument
  other than `deps.candidates`. Inspect `BaselineRunner(...)` calls in `steps.py`
  specifically and validate the `candidates=` argument — do not ban all
  `Candidate(...)` construction in `run_cycle`. Verify the test passes.
- [x] 4.5 Verify or extend a test confirming the optimization cycle runs with
  `FakeCandidateScorer` and real `ArgmaxClassificationPolicy` without loading a
  model. Run `pytest tests/unit/run_cycle/ -x --tb=short` and confirm no model
  is loaded.

## 5. LLM logit scorer unit and contract tests

- [x] 5.1 Verify or extend tests in
  `tests/unit/candidate_scoring/test_logit_scorer.py` covering causal
  next-token alignment: `logit[prefix_last] → c0`, not `logit[c0] → c0`.
  Include a test that catches an off-by-one implementation. Verify by running
  `pytest tests/unit/candidate_scoring/test_logit_scorer.py -x --tb=short`.
- [x] 5.2 Verify or extend tests covering multi-token candidate scoring:
  the score is the arithmetic mean of per-token log-probabilities, not the sum.
  Verify the test passes.
- [x] 5.3 Add a test verifying candidate token IDs used for scoring exactly
  occupy the candidate continuation positions in the actual model input (not
  independently tokenized and concatenated). Use `StubTokenizer` with
  controllable token IDs. Verify the test passes.
- [x] 5.4 Verify or extend tests in
  `tests/unit/candidate_scoring/test_batch_scoring.py` covering batched
  scoring semantics: one logical `score(text, candidates)` call produces one
  judgment per candidate in the same order (`len(judgments) == len(candidates)`
  and `judgments[i].candidate == candidates[i]`). Verify the test passes.
- [x] 5.5 Add or extend a test verifying that a batched model call failure
  propagates `CandidateScoringError` and produces no partial judgments. Use
  `FailingModel` or `PartialFailingModel`. Verify the test passes.
- [x] 5.6 Verify or extend tests covering model adapter failure propagation:
  `LLMScoringError` is translated to `CandidateScoringError` at the port
  boundary with the original exception available as the cause. Verify the test
  passes.

## 6. Non-blocking async boundary test

- [x] 6.1 Add a test verifying that the real `LLMLogitCandidateScorer` does not
  synchronously block the event loop on model inference. Use deterministic fake
  tokenizer/model adapters where the fake model deliberately blocks on a
  synchronization primitive (e.g., `threading.Event`). While scoring is awaited,
  a concurrent `asyncio` task must make progress, proving blocking inference is
  offloaded off the event loop. Do not assert the specific executor mechanism
  (e.g., `asyncio.to_thread`). Verify by running the test with
  `pytest -x --tb=short`.

## 7. Architectural and contract tests

- [x] 7.1 Verify that `lint-imports` already enforces application →
  infrastructure isolation (`application-isolated`, `application-ports-isolated`
  contracts). Do not duplicate this check in an AST test. Add a symbol-level
  AST check only if an existing import-linter contract permits an import path
  that is still architecturally forbidden. Verify by running `lint-imports` and
  confirming the contracts pass.
- [x] 7.2 Add an architectural test verifying the semantic invariant: concrete
  scorer construction has one production source. Validate that both composition
  roots (`interfaces/cli/main.py:build_cli_deps` and
  `run_baseline/runner.py:_main_async`) reference `build_candidate_scorer` and
  neither constructs `LLMLogitCandidateScorer` directly. Use source inspection
  or `ast.walk`. The invariant is semantic ("one production construction path");
  the AST assertion is its current validation. Verify the test passes.
- [x] 7.3 Add an architectural test verifying the scorer is constructed once
  per process in the composition root and passed into `RunCycleDeps` (not
  reconstructed per round or per runner). Inspect `build_cli_deps` to confirm
  a single `build_candidate_scorer` call. Verify the test passes.
- [x] 7.4 Preserve and extend `tests/contract/test_no_generative_classification.py`
  to verify `ClassificationResult`, `classify_detailed`, and `classify_many`
  remain absent from production code. Avoid broad forbidden-word checks on
  `"classify"` or `"confidence"`. Verify by running
  `pytest tests/contract/test_no_generative_classification.py -x --tb=short`.
- [x] 7.5 Add a contract test verifying the classification path does not depend
  on legacy structured-generation mechanisms. Target concrete obsolete runtime
  artifacts — specific symbols, generated-output model types, classification
  JSON parsers or helpers — not vague textual concepts like `"decision"` or
  `"output_contract"`. Ban obsolete runtime artifacts, not vocabulary. Verify
  the test passes.
- [x] 7.6 Add or verify an architectural test confirming the composition root
  is allowed to import infrastructure (positive assertion: `interfaces/cli/main.py`
  imports `build_candidate_scorer` from `infrastructure`). Verify the test
  passes.
- [x] 7.7 Add a test verifying no infrastructure state escapes into domain or
  application: inspect the field types of `Judgment`, `Classification`,
  `ResultRow`, and optimization state and confirm they reference only
  domain/application-safe types (`Candidate`, `Judgment`, `Classification`,
  `str`, `int`, `float`, `bool`, `None`, `dict`, or collections thereof). No
  `torch.Tensor`, `token_ids`, `raw_logits`, `attention_mask`, or model output
  object SHALL appear as a field. Verify the test passes.

## 8. Regression and full validation

- [x] 8.1 Verify existing optimization regression tests remain unchanged and
  passing: `pytest tests/unit/run_cycle/ -v` covers iteration, candidate queue,
  active/new/holdout evaluation, rollback, reporting, state dump, resumability,
  stopping conditions, and version comparison. Confirm no behavior change.
- [x] 8.2 Run `pytest` and verify the full test suite passes, including all
  new architectural and contract tests. Confirm no model is loaded by ordinary
  unit tests.
- [x] 8.3 Run `mypy` and verify type checking passes with no regressions.
- [x] 8.4 Run `ruff check` and verify linting passes with no regressions.
- [x] 8.5 Run `lint-imports` and verify all import-linter contracts pass,
  including `domain-isolated`, `domain-stdlib-only-allowlist`,
  `application-isolated`, `application-ports-isolated`,
  `infrastructure-isolated`, and `no-cyclic-imports`.
- [x] 8.6 Run
  `openspec validate --changes candidate-scoring-architectural-validation`
  and verify the spec delta is valid: every requirement has at least one
  scenario, the `## ADDED Requirements` format is correct, and the `## Purpose`
  section is present.
- [x] 8.7 Verify the acceptance criteria from `design.md` are met: inspect
  test coverage for each invariant, confirm no production behavior outside the
  migration changed, and confirm all architectural violations were fixed rather
  than tests weakened.
