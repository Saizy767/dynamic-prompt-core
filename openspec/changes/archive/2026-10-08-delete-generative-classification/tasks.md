# Tasks

## 1. Search and classify all references

- [x] 1.1 Search `src/` and `tests/` for legacy APIs (`classify_detailed`, `classify`, `classify_many`, `classify_many_detailed`, `ClassificationResult`) and affected `ResultRow` fields (`predicted_decision`, `confidence`, `raw_classify`). Classify each occurrence as production dependency, test dependency, unrelated use, documentation, or dead code. Record findings for use in subsequent tasks. Verify: a written inventory of all occurrences exists with classification labels.

## 2. Migrate BaselineRunner to candidate scoring

- [x] 2.1 Add a `[classification].candidates` field to `config.toml` defaulting to `["0", "1"]` for binary classification. Add a `RunnerConfig` field to parse it. Verify: `RunnerConfig.from_config()` loads the candidate values.
- [x] 2.2 Add `CandidateScorer`, `ClassificationPolicy`, and `candidates: list[Candidate]` dependencies to `BaselineRunner.__init__` alongside the existing `AsyncTask` dependency. Verify: `BaselineRunner` accepts scorer, policy, and candidates without breaking construction.
- [x] 2.3 Replace the `classify_detailed` call in `_process_example` with `CandidateScorer.score(text, candidates)` → `ClassificationPolicy.classify(judgments)` → `Classification`. Populate `predicted_decision` from `int(Classification.selected.value)`. Catch `CandidateScoringError` and map to `classify_status=failed`. Verify: the classification step uses candidate scoring and does not call `classify_detailed`.
- [x] 2.4 Update `ResultRow` schema: keep `predicted_decision` (populated from `Classification.selected`), remove `confidence`, remove `raw_classify`, add `selected_candidate` (string), add `judgment_scores` (dict mapping candidate values to scores). Update `to_dict` accordingly. Verify: `ResultRow` fields match the delta spec for `stage1-baseline-runner`.
- [x] 2.5 Update the runner's CLI (`_main_async`, `main`) and any composition root to wire `CandidateScorer` and `ClassificationPolicy` into the runner. Construct `Candidate` objects from the configured candidate values. Verify: the CLI can construct the runner with all required dependencies.
- [x] 2.6 Update or add unit tests for `BaselineRunner` covering: normal classification via candidate scoring, multiple candidates, scoring failure mapping, candidate ordering preservation. Verify: runner tests pass with the new classification flow.

## 3. Migrate metrics module

- [x] 3.1 Remove `confidence` from `REQUIRED_FIELDS` in `application/services/metrics/metrics.py`. Verify: `load_results` no longer rejects result rows without `confidence`.
- [x] 3.2 Update `tests/unit/run_cycle/test_steps.py` synthetic result rows to omit `confidence`. Verify: step tests pass without `confidence` in synthetic data.
- [x] 3.3 Verify `predicted_decision` remains unchanged in the metrics module — it is populated from candidate scoring instead of generative classification, but the field name and semantics are preserved. Verify: metrics computation tests pass.

## 4. Remove ClassificationResult

- [x] 4.1 Delete `application/schemas/classification.py`. Verify: the file no longer exists.
- [x] 4.2 Update `application/schemas/__init__.py` to remove the `ClassificationResult` import and `__all__` entry. Verify: `from dynamic_prompt_core.application.schemas import ClassificationResult` fails with `ImportError`.
- [x] 4.3 Update `domain/models/classification.py` docstring only if it references `ClassificationResult` or generative classification terminology. Do not alter the domain semantics of `Classification`. Verify: no docstring references `ClassificationResult`.
- [x] 4.4 Grep the entire `src/` tree for remaining `ClassificationResult` references and remove or update them. Verify: `rg "ClassificationResult" src/` returns no results.

## 5. Remove LLMClient classification methods

- [x] 5.1 Remove `classify`, `classify_detailed`, `classify_many`, and `classify_many_detailed` methods from `AsyncTask` in `infrastructure/llm/client.py`. Preserve `extract_theses`, `extract_theses_detailed`, `extract_theses_many`, `extract_theses_many_detailed`, `analyze`, `analyze_raw`, `analyze_many`. Verify: `rg "async def classify" src/dynamic_prompt_core/infrastructure/llm/client.py` returns no results.
- [x] 5.2 Remove `classify` and `classify_many` from the `LLMClient` port in `application/ports/outbound/llm_client.py`. Preserve `extract_theses` and `extract_theses_many`. Verify: the port defines only extraction methods.
- [x] 5.3 Update `infrastructure/llm/__init__.py` and any `__all__` exports if classification-specific symbols were exported. Verify: `python -c "from dynamic_prompt_core.infrastructure.llm import AsyncTask"` succeeds.
- [x] 5.4 Update or add tests verifying that `extract_theses_detailed` and `analyze` still work after classification method removal. Verify: extraction tests pass.

## 6. Remove dead code and cleanup

- [x] 6.1 Remove dead imports of `ClassificationResult`, `classify_detailed`, `classify`, `classify_many` across `src/`. Verify: `rg "ClassificationResult|classify_detailed|classify_many" src/` returns no results.
- [x] 6.2 Remove classification-specific JSON parsing, response models, and structured-output schemas that existed solely for generative classification. Verify: no classification-specific structured-output code remains.
- [x] 6.3 Remove generative-specific error types with no remaining consumers. Verify: `rg` for removed error types returns no results.
- [x] 6.4 Remove obsolete mocks, fixtures, and test helpers for generative classification. Verify: no test file references deleted symbols.
- [x] 6.5 Update documentation (README, architecture docs, inline docstrings) to describe candidate scoring as the sole classification mechanism. Remove references to generative classification as a production path. Verify: documentation does not present generative classification as supported.

## 7. Forbidden classification-path references

- [x] 7.1 Add a forbidden-reference test that verifies no production code under `src/` imports or references the following specific symbols: `classify_detailed`, `ClassificationResult`, `classify_many`, `classify_many_detailed`. These are forbidden because they are deleted generative-classification symbols. Verify: the test passes and would fail if a reference were reintroduced.
- [x] 7.2 Add a forbidden-reference test that verifies the `LLMClient` port (`application/ports/outbound/llm_client.py`) does not define `classify` or `classify_many` methods. Verify: the test passes.
- [x] 7.3 Add a forbidden-reference test that verifies no runtime fallback to generative classification exists (no `try: candidate_scoring except: generative` pattern). Verify: the test passes.
- [x] 7.4 Verify the following symbols are explicitly allowed and not flagged by the forbidden-reference tests: `classify_input` (the new use case), `ClassificationPolicy` (the new policy), `Classification` (the new domain model), `CLASSIFICATION_PROMPT_V0` (retained as prompt versioning seed), `classification_policy.py` (the new service module). Verify: the allowlist is documented in the test.

## 8. Semantic migration validation

- [x] 8.1 Verify that `ScoringPromptBuilder._SEMANTIC_PREFIX` contains all semantic classification criteria from `CLASSIFICATION_PROMPT_V0` (role, task, rules) in candidate-evaluation language. The existing `TestJudgmentPromptSemantics` tests in `tests/unit/candidate_scoring/test_scoring_seams.py` serve as this verification. Verify: these tests pass after the generative path is removed, confirming the semantic criteria are not lost when the generative code is deleted.

## 9. Integration verification

- [x] 9.1 Run the full test suite (`pytest`) and verify all tests pass. Verify: exit code 0.
- [x] 9.2 Run `lint-imports` and verify architectural boundary contracts pass. Verify: exit code 0.
- [x] 9.3 Run `mypy` and verify type checking passes. Verify: no type errors.
- [x] 9.4 Run `ruff check` and verify linting passes. Verify: no lint errors.
- [x] 9.5 Verify the candidate-scoring flow (`CandidateScorer` → `ClassificationPolicy` → `Classification`) is the only supported classification path by inspecting the codebase. Verify: no generative classification path exists in production code.
