# Tasks

## 1. Port contract and schemas

- [x] 1.1 Redefine `TeacherLLMClient` in
  `application/ports/outbound/teacher_llm_client.py` as a `typing.Protocol` with
  a single `review_theses` method: `async def review_theses(self, text: str,
  theses: list[str], *, system_prompt: str) -> RefinementReview`. Define
  `RefinementReview` as a typed result (plan, latency_ms, usage) in the port
  module. Remove the old `classify` / `extract_theses` methods. Verify the module
  imports only `typing` and `application.schemas.refinement` (no `infrastructure`),
  and that `isinstance(mock, TeacherLLMClient)` works for a matching mock via
  `@runtime_checkable`.
- [x] 1.2 Add `RefinementPlan`, `ReformulateItem`, and `DropItem` Pydantic models
  to `application/schemas/refinement.py` (design D7), moving them from
  `refiner.py:104-119`. Keep `RuleFormulation` intact. Verify
  `RefinementPlan.model_validate_json('{"keep":[],"reformulate":[],"drop":[],"add":[]}')`
  succeeds and that a missing `"from"` alias on a `reformulate` item raises
  `ValidationError`.

## 2. Domain service — pure refinement logic

- [x] 2.1 Create `domain/services/thesis_refinement.py` with a framework-free
  `RefinementPlanData` dataclass (keep, reformulate, drop, add as plain lists of
  plain dataclasses / dicts) and an `apply_refinement(theses_raw, plan, *,
  filter_noisy, filter_interpretive, allow_additions) -> (theses_refined,
  filtered_out, added)` function (design D3). Move the logic from
  `refiner.py:277-341` (`_apply_refinement`). Verify the module imports only
  `dataclasses` and `typing` (no `pydantic`, `aiohttp`, or `infrastructure`).
- [x] 2.2 Verify `apply_refinement` produces correct output for: (a) all-keep plan
  → `theses_refined` equals input, `filtered_out` empty; (b) a drop plan → thesis
  in `filtered_out` with reason, not in `theses_refined`; (c) a reformulate where
  `len(to) <= len(from)` → `to` in `theses_refined`; (d) a reformulate where
  `len(to) > len(from)` → original kept, `to` discarded; (e) an add plan with
  `allow_additions=True` → thesis in `added` with `source="teacher"`; (f) an add
  plan with `allow_additions=False` → nothing added.

## 3. Teacher client implementation

- [x] 3.1 Create `infrastructure/llm/teacher_client.py` with a `TeacherClient`
  class implementing `TeacherLLMClient` (design D2). Constructor takes
  `endpoint`, `model_name`, `temperature`, `max_tokens`, `timeout`,
  `max_retries`, and `config_path`. It builds an `AsyncTask` pointed at the
  teacher endpoint and overrides `_served_model_name` to `model_name`. The
  `review_theses` method builds the user message, calls `AsyncTask.analyze_raw`
  with `RefinementPlan`, validates the response, and returns a `RefinementReview`.
  Verify the class does not import `application.use_cases` and that
  `isinstance(TeacherClient(...), TeacherLLMClient)` is `True`.
- [x] 3.2 Move `build_teacher_prompt(config)` from `refiner.py:125` into
  `TeacherClient` (or a module-level helper in `teacher_client.py`) so prompt
  assembly lives with the transport, not the use case. Verify the prompt includes
  the noisy-filter, interpretive-filter, and addition instructions conditionally
  on the config flags, matching the existing output.
- [x] 3.3 Verify `TeacherClient.review_theses` with a mocked `AsyncTask`
  returning a valid `RefinementPlan` JSON yields a `RefinementReview` with the
  parsed plan, `latency_ms` from the raw response, and `usage` from the raw
  response. Verify a mocked `AsyncTask` returning `ResponseStatus.ERROR` raises
  an exception (to be caught by the use case's failure isolation).

## 4. Dependency object and use case refactor

- [x] 4.1 Create
  `application/use_cases/refine_theses/refine_theses_deps.py` with a frozen
  dataclass `RefineThesesDeps(teacher_llm_client: TeacherLLMClient,
  run_repository: RunRepository)` (design D4), mirroring `RunCycleDeps`. Verify
  it imports only the two ports and that constructing it with mock ports
  succeeds.
- [x] 4.2 Define `RefineThesesResult` dataclass (processed, teacher_errors,
  filtered, reformulated, added, skipped, total_prompt_tokens,
  total_completion_tokens, total_latency_ms, artifact_path) in
  `application/use_cases/refine_theses/refiner.py` or a `result.py` module
  (design D6). Verify it is a frozen dataclass with all fields defaulting to
  zero / empty.
- [x] 4.3 Refactor `refiner.py` so the use case function
  `refine_theses(deps: RefineThesesDeps, input: RefineThesesInput) ->
  RefineThesesResult` receives dependencies via the deps object (design D4). The
  use case calls `deps.teacher_llm_client.review_theses(...)` instead of
  `_call_teacher(...)` directly, and delegates pure logic to
  `domain.services.thesis_refinement.apply_refinement`. Remove the
  `_build_teacher_task` function and the direct `AsyncTask` import. Verify the
  module does not import `infrastructure.llm` or `aiohttp`.
- [x] 4.4 Preserve the per-example failure isolation (design D7 from the archived
  change): wrap each `review_theses` call in try/except; on failure, record the
  example with `theses_refined = theses_raw_tiny`, increment `teacher_errors`,
  log the error, and continue. Verify with a mock port that raises on one
  example: that example keeps its originals, processing continues, and
  `teacher_errors` is incremented.
- [x] 4.5 Preserve cost and latency accounting: extract `latency_ms` and `usage`
  from `RefinementReview`, accumulate into the result totals, and log per-call.
  Verify with a mock port returning `usage={"prompt_tokens": 10,
  "completion_tokens": 5}` and `latency_ms=42.0`: the result totals reflect
  these values and the log entry contains them.

## 5. Composition root wiring and CLI

- [x] 5.1 Add a `build_refine_deps(config) -> RefineThesesDeps` function to
  `interfaces/cli/main.py` (design D5) that constructs `TeacherClient` from the
  `[teacher]` config section and a `RunRepository` implementation, and returns
  `RefineThesesDeps(teacher_llm_client=..., run_repository=...)`. Verify
  `TeacherClient` is not instantiated in any other module (search the codebase
  for `TeacherClient(` outside `main.py` and `teacher_client.py`).
- [x] 5.2 Add a `--refine` subcommand (or argparse dispatch) to `main.py` that
  loads the `[teacher]` config, calls `build_refine_deps`, constructs
  `RefineThesesInput`, invokes `refine_theses(deps, input)`, and prints the
  summary. Verify `configure_logging()` is called in `main.py` and that no
  refinement module calls `logging.basicConfig`.
- [x] 5.3 Remove the standalone `main()` from `refiner.py` (or reduce it to a
  thin wrapper that delegates to the composition root). Verify `refiner.py` no
  longer has `if __name__ == "__main__"` that constructs `AsyncTask` directly.

## 6. Artifact I/O and reload

- [x] 6.1 Keep `write_refined_artifact` and `load_refined` in the use case module
  (or move to an infrastructure storage helper if `RunRepository` is extended).
  Verify the artifact filename is
  `theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl`, each record has
  `id`, `theses_raw_tiny`, `theses_refined`, `filtered_out`, `added`, and
  `load_refined(write_refined_artifact(records, ...))` round-trips the records.
- [x] 6.2 Verify the source results artifact is not modified: after refinement,
  the input `results_*.jsonl` file is byte-identical to its pre-refinement state.

## 7. Static checks and integration

- [x] 7.1 Run `lint-imports` and verify it passes: the use case
  (`application.use_cases.refine_theses`) depends on
  `application.ports.outbound.teacher_llm_client` and
  `domain.services.thesis_refinement`, not on `infrastructure`. The domain
  service has no `infrastructure` / `pydantic` / `aiohttp` imports.
  `infrastructure.llm.teacher_client` depends on
  `application.ports.outbound.teacher_llm_client` (allowed) but not on
  `application.use_cases`.
- [x] 7.2 Run `mypy` and verify it passes with no errors on the new and modified
  modules. Pay attention to the `RefinementReview` type, the
  `TeacherLLMClient` protocol conformance, and the `RefineThesesDeps` fields.
- [x] 7.3 Run an end-to-end test with a mocked teacher port: load a small
  `results_*.jsonl` fixture, run `refine_theses(deps, input)`, verify the
  written artifact has the expected records, the summary counters are correct,
  and the refinement log contains entries for `load_candidates`,
  `teacher_call`, `filter` / `reformulate` / `add`, and `write_artifact`.
