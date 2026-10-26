# Tasks

## 1. Audit current interfaces and dependencies

- [x] 1.1 Grep all imports and usages of `CandidateScorer`, `LLMLogitCandidateScorer`, `ScoringPromptBuilder`, `ClassificationPolicy`, `ArgmaxClassificationPolicy`, `CandidateScoringError`, `ClassifyInput`, and `EXTRACTION_PROMPT` across `src/` and `tests/`. Record every file and line. Verify the output is a complete dependency map.
- [x] 1.2 Read `RunCycleDeps` (`application/use_cases/run_cycle/run_cycle_deps.py`) and all three composition roots (`interfaces/cli/main.py`, `runner.py:_main_async`, `compare_versions/comparator.py:_run_version_async`) to document how `scorer`, `policy`, and `candidates` are currently wired. Verify the map covers every construction site.
- [x] 1.3 Check `pyproject.toml` for existing `package-data` or `include-package-data` configuration. Verify the current state has no packaged resource files.

## 2. Add ClassificationRules domain model

- [x] 2.1 Create `src/dynamic_prompt_core/domain/models/classification_rules.py` with a frozen dataclass `ClassificationRules` having `rules: tuple[str, ...]`. Add it to `domain/models/__init__.py` exports. Verify `from dynamic_prompt_core.domain.models import ClassificationRules` succeeds.
- [x] 2.2 Add a conversion function `classification_rules_from_artifact(artifact: PromptArtifact) -> ClassificationRules` that handles `None` layers and empty `rules` list by returning an empty `ClassificationRules`, preserves rule order and text, and raises on non-string or malformed elements. Verify with a unit test covering: valid rules, `None` layers, empty list, non-string element.
- [x] 2.3 Write unit tests in `tests/unit/domain/test_classification_rules.py` covering: immutability, order preservation, text preservation, `None` layers → empty, empty list → empty, non-string element → error, 3-5 constraint bypass for empty rules. Verify `pytest tests/unit/domain/test_classification_rules.py` passes.

## 3. Add system prompt resource files and packaging

- [x] 3.1 Create `src/dynamic_prompt_core/infrastructure/llm/prompts/classifier.txt` with the classifier system prompt text (adapted from `ScoringPromptBuilder._FIXED_INSTRUCTIONS`). Verify the file exists and contains non-empty text.
- [x] 3.2 Create `src/dynamic_prompt_core/infrastructure/llm/prompts/extractor.txt` with the extractor system prompt text (adapted from `EXTRACTION_PROMPT.text` in `domain/prompts/fixed.py`). Verify the file exists and contains non-empty text.
- [x] 3.3 Add `[tool.setuptools.package-data]` configuration in `pyproject.toml` to include `infrastructure/llm/prompts/*.txt`. Verify `pip install -e .` succeeds and the files are accessible via `importlib.resources`.
- [x] 3.4 Add a resource-loading utility (e.g., `load_prompt(name: str) -> str` in `infrastructure/llm/prompts/__init__.py`) that reads a prompt file by name, with clear error on missing/unreadable file. Verify with a unit test that loading `classifier` and `extractor` returns non-empty strings and loading a nonexistent name raises a clear error.

## 4. Implement Classifier port and infrastructure

- [x] 4.1 Create the `Classifier` outbound port in `application/ports/outbound/classifier.py` as a `typing.Protocol` with `async def classify(self, text: str, rules: ClassificationRules) -> Classification`. Verify the port is importable from `application.ports.outbound` and does not import `infrastructure`.
- [x] 4.2 Create the `Classifier` infrastructure implementation in `infrastructure/llm/classifier.py` that: loads the classifier system prompt at init, validates `ClassificationRules` at the boundary, builds the scoring prefix (system prompt + rules + text + candidate), runs the existing logit-based scoring (tokenizer → model forward → `LogitScorer` → softmax), applies the argmax decision with input-order tie-breaking, and returns `Classification`. Verify the implementation imports `application.ports.outbound.classifier` and `domain.models` but not `application.use_cases`.
- [x] 4.3 Adapt the existing `ScoringPromptBuilder` (rename to `ClassifierPromptBuilder` or similar) to accept the loaded system prompt as the initial prefix instead of `_FIXED_INSTRUCTIONS`. Preserve the rules formatting, candidate building, and prefix structure. Verify the adapted builder produces the same prefix format as before when given the same system prompt text.
- [x] 4.4 Create a `build_classifier(config: ScorerBackendConfig, candidates: list[Candidate]) -> Classifier` factory in `infrastructure/llm/` that wires the tokenizer adapter, model adapter, `LogitScorer`, adapted prompt builder, and system prompt. Verify the factory returns a `Classifier` instance and reuses the existing backend dispatch (gguf, llama_server, huggingface).
- [x] 4.5 Write unit tests for the `Classifier` in `tests/unit/infrastructure/llm/test_classifier.py` covering: returns `Classification` with correct selected candidate, preserves judgment order, uses loaded system prompt as prefix, logit scoring mechanism preserved (fake adapters), error propagation on model failure, no partial results, non-blocking async. Verify `pytest tests/unit/infrastructure/llm/test_classifier.py` passes.

## 5. Implement Extractor component

- [x] 5.1 Create the `Extractor` component in `infrastructure/llm/extractor.py` (or `application/` if it wraps a port) that: loads the extractor system prompt at init, calls `AsyncTask.extract_theses_detailed()` with the loaded prompt, applies `_try_repair()` and `_map_call_status()`, runs `normalize_theses()`, and returns `(theses_raw, theses_norm, status, latency_ms, raw_extract)`. Verify the component does not receive classification results.
- [x] 5.2 Write unit tests for the `Extractor` in `tests/unit/infrastructure/llm/test_extractor.py` covering: loads dedicated system prompt, calls transport with same inputs as before, preserves normalization, preserves status values (`ok`, `repaired`, `failed`, `skipped`), preserves latency measurement, error handling does not abort. Verify `pytest tests/unit/infrastructure/llm/test_extractor.py` passes.

## 6. Restructure BaselineRunner

- [x] 6.1 Change `BaselineRunner.__init__` to accept `classifier: Classifier` and `extractor: Extractor` instead of `scorer`, `policy`, and `task`. Keep `candidates`, `classify_prompt`, `skip_extraction`, and other existing params. Verify the constructor signature is updated and old params are removed.
- [x] 6.2 Rewrite `_process_example` to run classification first: convert `classify_prompt` to `ClassificationRules` via `classification_rules_from_artifact()`. If rules are empty, set `predicted_decision=0`, `selected_candidate="0"`, `classify_status="ok"`, `judgment_scores={}`, `classify_latency_ms=0` without calling `Classifier`. If rules are non-empty, call `classifier.classify(text, rules)` and populate fields from the returned `Classification`. Verify with a unit test that absent rules produce `predicted_decision=0` without calling the classifier.
- [x] 6.3 After classification, run extraction (if not `skip_extraction`): call `extractor.extract(text, label, language)` and populate `theses_raw`, `theses_norm`, `extract_status`, `extract_latency_ms`, `raw_extract`. If `skip_extraction`, set skipped values as today. Verify with a unit test that extraction runs after classification and `skip_extraction=True` skips the extractor.
- [x] 6.4 Preserve `ResultRow` structure: all 13 fields, their names, types, and semantics unchanged. Verify `test_result_row_schema_has_no_confidence_or_raw_classify` still passes and no fields are added or removed.
- [x] 6.5 Update error handling: classification errors set `classify_status="failed"`, `predicted_decision=None`, `selected_candidate=None`, `judgment_scores={}`. Classification errors are not treated as absent rules. Verify with a unit test that a classifier error produces `classify_status="failed"` and does not silently set `predicted_decision=0`.
- [x] 6.6 Write integration tests for `BaselineRunner` in `tests/unit/run_baseline/test_runner.py` covering: rules present → classifier called first, then extractor; absent layers → classification=0, classifier not called, extractor called; empty rules → same as absent; invalid rules → existing error policy; `skip_extraction=True` → extractor not called; `ResultRow` format preserved. Verify `pytest tests/unit/run_baseline/test_runner.py` passes.

## 7. Update run_cycle integration

- [x] 7.1 Update `RunCycleDeps` to replace `candidate_scorer` and `classification_policy` with `classifier` and `extractor`. Keep `candidates` and all other fields. Verify the dataclass is updated and frozen.
- [x] 7.2 Update `run_cycle/steps.py` to pass `classifier` and `extractor` from `RunCycleDeps` to `BaselineRunner` at all three construction sites (`run_active_on_dev`, `run_new_on_dev`, `run_holdout`). Do not change the cycle sequence, policy, acceptance criteria, or rollback logic. Verify with `pytest tests/unit/run_cycle/test_steps.py` that the step sequence and dependency wiring are correct.
- [x] 7.3 Update `tests/unit/run_cycle/conftest.py`: replace `FakeCandidateScorer` with `FakeClassifier` (returns a predetermined `Classification`), add `FakeExtractor`, and update `build_mock_deps` to construct them. Verify all run_cycle tests pass with the updated fixtures.

## 8. Update composition roots

- [x] 8.1 Update `interfaces/cli/main.py` `build_cli_deps()` to construct `Classifier` via `build_classifier()` and `Extractor`, and pass them to `RunCycleDeps` instead of `scorer` and `policy`. Verify the composition root references `build_classifier` and does not construct the concrete classifier directly.
- [x] 8.2 Update `runner.py:_main_async()` to construct `Classifier` and `Extractor` and pass to `BaselineRunner`. Verify the runner CLI constructs dependencies through the shared factory.
- [x] 8.3 Update `compare_versions/comparator.py:_run_version_async()` to construct `Classifier` and `Extractor` and pass to `BaselineRunner`. Verify the comparator CLI constructs dependencies through the shared factory.
- [x] 8.4 Add or update the contract test in `tests/contract/test_architectural_invariants.py` to verify both composition roots route through `build_classifier` and neither constructs the concrete classifier directly. Verify `pytest tests/contract/test_architectural_invariants.py` passes.

## 9. Remove old abstractions and clean up

- [x] 9.1 Grep for all remaining references to `CandidateScorer`, `candidate_scorer`, `LLMLogitCandidateScorer`, `ClassificationPolicy`, `classification_policy`, `ArgmaxClassificationPolicy`, `ClassifyInput`, `build_candidate_scorer`, and `CandidateScoringError` in `src/`. Remove or rename each. Verify `grep -r` finds zero matches in `src/` for the old names.
- [x] 9.2 Remove `application/ports/outbound/candidate_scorer.py` after confirming no imports remain. Remove or adapt `application/use_cases/classify_input/` (remove if `BaselineRunner` is the only caller, otherwise adapt to use `Classifier`). Verify no import errors.
- [x] 9.3 Remove `application/services/classification_policy.py` after confirming no imports remain. Preserve the decision logic inside the `Classifier` implementation. Verify no import errors.
- [x] 9.4 Remove `EXTRACTION_PROMPT` from `domain/prompts/fixed.py` only if no code references it after the `Extractor` loads from a resource file. Preserve `CLASSIFICATION_PROMPT_V0` and other entities. Verify `grep -r EXTRACTION_PROMPT src/` finds zero matches (or document why it is retained).
- [x] 9.5 Rename or remove old scoring modules (`infrastructure/llm/scoring/candidate_scorer.py`, `scoring/prompt_builder.py`) after confirming their logic is fully transferred to the `Classifier` implementation. Preserve `LogitScorer`, tokenizer/model adapters, and other reusable low-level algorithms. Verify no import errors.

## 10. Update and add tests

- [x] 10.1 Update `tests/unit/run_baseline/test_runner.py`: replace `FakeCandidateScorer` mocks with `FakeClassifier` mocks, update tests that asserted `ValueError` on `None` layers to assert `predicted_decision=0` instead, update `test_runner_skip_extraction` to verify `Extractor` is not called. Verify all updated tests pass.
- [x] 10.2 Update `tests/unit/candidate_scoring/` tests: adapt or remove tests that reference `CandidateScorer`, `ClassificationPolicy`, `ClassifyInput`. Move preserved tests (logit scoring, causal alignment, non-blocking async) to reference the `Classifier` implementation. Verify `pytest tests/unit/candidate_scoring/` passes (or the renamed directory).
- [x] 10.3 Update `tests/contract/test_no_generative_classification.py` to verify classification goes through the `Classifier` port, not generative methods. Verify the contract test passes.
- [x] 10.4 Update `tests/contract/test_architectural_invariants.py` to verify the `Classifier` port boundary, single construction path, and layer dependencies. Verify the contract test passes.
- [x] 10.5 Add regression tests in `tests/regression/` verifying: rule discovery only on dev, holdout evaluates full rule set, rules unchanged during holdout evaluation, acceptance/rejection criteria preserved, `run_new_on_dev()` conditions preserved. Verify `pytest tests/regression/` passes.
- [x] 10.6 Add a test verifying system prompt resource files are accessible via `importlib.resources` after installation (not just from repo root). Verify the test passes.

## 11. Final validation

- [x] 11.1 Run `pytest` and verify all tests pass with no regressions.
- [x] 11.2 Run `mypy` and verify type checking passes with no regressions.
- [x] 11.3 Run `ruff check` and verify linting passes with no regressions.
- [x] 11.4 Run `lint-imports` and verify all import-linter contracts pass, including domain isolation, application isolation, port isolation, and the `Classifier` port boundary.
- [x] 11.5 Grep `src/` for any remaining references to removed names (`CandidateScorer`, `LLMLogitCandidateScorer`, `ScoringPromptBuilder`, `ClassificationPolicy`, `ArgmaxClassificationPolicy`, `ClassifyInput`, `build_candidate_scorer`, `CandidateScoringError`). Verify zero matches.
- [x] 11.6 Verify JSONL output format is unchanged by running a smoke test and comparing output fields against the existing `ResultRow` schema.
