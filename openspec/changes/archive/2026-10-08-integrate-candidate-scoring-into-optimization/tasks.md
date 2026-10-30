# Tasks

## 1. Extract shared `build_candidate_scorer` factory

- [x] 1.1 Create a `build_candidate_scorer(model_path: str) -> CandidateScorer`
  helper that constructs `LLMLogitCandidateScorer` with
  `ScoringPromptBuilder`, `HuggingFaceBatchTokenizerAdapter`,
  `TorchBatchedCausalLMAdapter`, and `LogitScorer`, loading the model via
  `AutoModelForCausalLM.from_pretrained(model_path)`. Place it outside the
  application layer — `interfaces/` or `infrastructure/` are both
  acceptable; choose the location that best fits the project's composition
  conventions. The factory may import `transformers` and scoring modules.
  Verify the module imports cleanly and the function returns a
  `CandidateScorer`-compatible object.
- [x] 1.2 Refactor `runner.py:_main_async` to call `build_candidate_scorer`
  instead of constructing the scorer inline. Verify the baseline runner CLI
  still produces the same scorer by running
  `pytest tests/unit/run_baseline/ -x --tb=short`.
- [x] 1.3 Run `mypy` and `ruff check` on the new factory module and the
  modified `runner.py`. Run `lint-imports` and verify no new contract
  violations — the factory is outside `application`, so it may import
  `application.ports`, `domain`, and `infrastructure`.

## 2. Add scorer, policy, and candidates to `RunCycleDeps`

- [x] 2.1 Add `candidate_scorer: CandidateScorer`,
  `classification_policy: ClassificationPolicy`, and
  `candidates: tuple[Candidate, ...]` as required fields (no defaults) to
  `RunCycleDeps` in
  `src/dynamic_prompt_core/application/use_cases/run_cycle/run_cycle_deps.py`.
  Import `CandidateScorer` from
  `application.ports.outbound.candidate_scorer`, `ClassificationPolicy` from
  `application.services.classification_policy`, and `Candidate` from
  `domain.models.candidate`. Verify with `mypy` that the dataclass
  type-checks and that `build_cli_deps` and `build_mock_deps` now produce a
  `TypeError` until updated.
- [x] 2.2 Run `ruff check src/dynamic_prompt_core/application/use_cases/run_cycle/run_cycle_deps.py`
  and verify no lint errors.

## 3. Thread dependencies through `steps.py`

- [x] 3.1 In `run_active_on_dev` (`steps.py`), add
  `scorer=deps.candidate_scorer`, `policy=deps.classification_policy`, and
  `candidates=list(deps.candidates)` to the `BaselineRunner(...)` constructor
  call. Verify by reading the constructor call and confirming all three
  keyword arguments are present.
- [x] 3.2 In `run_new_on_dev` (`steps.py`), add the same three keyword
  arguments to the `BaselineRunner(...)` constructor call. Verify by reading
  the constructor call.
- [x] 3.3 In `run_holdout` (`steps.py`), add the same three keyword
  arguments to the `BaselineRunner(...)` constructor call. Verify by reading
  the constructor call.
- [x] 3.4 Run `mypy src/dynamic_prompt_core/application/use_cases/run_cycle/steps.py`
  and `ruff check src/dynamic_prompt_core/application/use_cases/run_cycle/steps.py`
  and verify no errors. Confirm no other `BaselineRunner` construction exists
  in `steps.py` by searching for `BaselineRunner(` in the file.

## 4. Wire scorer, policy, and candidates in the composition root

- [x] 4.1 In `build_cli_deps` (`interfaces/cli/main.py`), call
  `build_candidate_scorer(model_path)` where `model_path` comes from
  `[llm].model_path` in `config.toml`. Construct
  `ArgmaxClassificationPolicy()` and
  `tuple(Candidate(v) for v in config.get("classification", {}).get("candidates", ["0", "1"]))`.
  Defer all infrastructure imports to the function body. Verify by reading
  `build_cli_deps` and confirming the scorer, policy, and candidates are
  constructed via the shared factory and passed to `RunCycleDeps`.
- [x] 4.2 Pass `candidate_scorer=scorer`, `classification_policy=policy`, and
  `candidates=candidates` to the `RunCycleDeps(...)` constructor in
  `build_cli_deps`. Verify by reading the `RunCycleDeps(...)` call and
  confirming all three keyword arguments are present.
- [x] 4.3 Verify that `build_cli_deps` is only reached by CLI commands whose
  dependency graph requires the classification scorer, by reading `main()`
  in `interfaces/cli/main.py` — the `refine` command exits before
  `build_cli_deps` is reached. Confirm the scorer is not constructed for
  commands that do not require classification.
- [x] 4.4 Run `mypy src/dynamic_prompt_core/interfaces/cli/main.py` and
  `ruff check src/dynamic_prompt_core/interfaces/cli/main.py` and verify no
  errors. Run `lint-imports` and verify the `application-ports-isolated`
  contract still passes (the composition root is in `interfaces`, which is
  allowed to import `infrastructure`).

## 5. Update test fixtures

- [x] 5.1 Add a `FakeCandidateScorer` class to
  `tests/unit/run_cycle/conftest.py` that implements the `CandidateScorer`
  protocol by returning deterministic `Judgment` objects (e.g., one judgment
  per candidate with a fixed score). Verify the fake scorer satisfies the
  `CandidateScorer` runtime-checkable protocol.
- [x] 5.2 Update the `build_mock_deps` fixture in `conftest.py` to pass
  `candidate_scorer=FakeCandidateScorer()`,
  `classification_policy=ArgmaxClassificationPolicy()` (the real policy, not
  a mock), and `candidates=(Candidate("0"), Candidate("1"))` into
  `RunCycleDeps`. Verify by running
  `pytest tests/unit/run_cycle/ -x --tb=short` and confirming no
  `TypeError` from missing `RunCycleDeps` fields.

## 6. Add integration tests for scorer/policy threading

- [x] 6.1 Add a test in `tests/unit/run_cycle/test_steps.py` that verifies
  `run_active_on_dev` passes `deps.candidate_scorer`,
  `deps.classification_policy`, and `deps.candidates` to the
  `BaselineRunner` constructor. Patch `BaselineRunner` to capture constructor
  kwargs and assert the scorer, policy, and candidates match the deps.
  Include a scorer identity assertion (`runner.scorer is deps.candidate_scorer`)
  and a candidate propagation assertion
  (`list(deps.candidates) == candidates passed to runner`). Verify the test
  passes.
- [x] 6.2 Add a test in `tests/unit/run_cycle/test_steps.py` that verifies
  `run_new_on_dev` and `run_holdout` similarly pass the scorer, policy, and
  candidates from deps to `BaselineRunner`. Verify the tests pass.
- [x] 6.3 Add a test that verifies all three `BaselineRunner` constructions
  during a single round receive the same scorer and policy object instances
  (identity check: `runner.scorer is deps.candidate_scorer`, not equality).
  Also assert that all three receive the same candidate list derived from
  `deps.candidates`. Verify the test passes.
- [x] 6.4 Run `pytest tests/unit/run_cycle/ -v` and verify all run_cycle unit
  tests pass, including the existing
  `test_steps.py::test_run_round_executes_steps_and_writes_report`.

## 7. Full validation

- [x] 7.1 Run `pytest` and verify the full test suite passes, including the
  contract tests in `tests/contract/test_no_generative_classification.py`
  that verify no generative classification symbols exist in production code.
- [x] 7.2 Run `mypy` and `ruff check` and verify they pass with no
  regressions.
- [x] 7.3 Run `lint-imports` and verify all import-linter contracts pass,
  including `application-ports-isolated` and the domain stdlib-only allowlist.
- [x] 7.4 Run
  `openspec validate --changes integrate-candidate-scoring-into-optimization`
  and verify the spec delta is valid: every requirement has at least one
  scenario, the modified `stage3-cycle-orchestrator` delta uses
  `## MODIFIED Requirements` with full updated content, and the added
  requirements use `## ADDED Requirements`.
- [x] 7.5 Verify the acceptance criteria from `design.md` are met: inspect
  `RunCycleDeps` for the three required fields with correct types, confirm
  `build_candidate_scorer` is shared between both composition roots, confirm
  no `BaselineRunner` construction in `run_cycle` omits scoring dependencies,
  and confirm `Judgment.score` is not used as an optimization metric.
