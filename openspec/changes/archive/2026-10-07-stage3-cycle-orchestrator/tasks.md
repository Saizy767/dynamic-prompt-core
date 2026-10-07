# Tasks

## 1. Dependency object and configuration

- [x] 1.1 Expand `RunCycleDeps` in
  `application/use_cases/run_cycle/run_cycle_deps.py` to a frozen dataclass
  with fields: `llm_client: LLMClient`, `prompt_repository: PromptRepository`,
  `run_repository: RunRepository`, `dataset_repository: DatasetRepository`,
  `embedding_client: EmbeddingClient`, `normalizer: Normalizer`,
  `teacher_llm_client: TeacherLLMClient | None`. Remove the `task_store` field.
  Import all ports from `application.ports.outbound`. Verify the dataclass
  constructs with all ports and that `teacher_llm_client` accepts `None`.
- [x] 1.2 Create `application/use_cases/run_cycle/config.py` with a frozen
  `CycleConfig` dataclass: `max_rounds: int = 5`,
  `max_consecutive_rollbacks: int = 2`, `decision_metric: str = "macro_f1"`,
  `tie_breaker_metric: str = "minority_f1"`, `dev_split: str = "dev"`,
  `holdout_split: str = "holdout"`, `run_id: str = ""`, `output_dir: str =
  "data/results"`, `prompt_store_path: str = ""`, `thesis_bank_path: str = ""`,
  `candidate_queue_path: str = ""`, `stop_on_first_error: bool = False`,
  `dump_state_after_each_round: bool = True`, `use_teacher_refinement: bool =
  False`. Add a `from_toml(config_path: str) -> CycleConfig` classmethod that
  reads `[cycle_orchestrator]` via `tomllib`. Verify `from_toml` returns
  configured values and falls back to defaults for missing keys.
- [x] 1.3 Update `RunCycleInput` in
  `application/ports/inbound/run_cycle_input.py` to carry a
  `config: CycleConfig | None = None` field (in addition to the existing
  `config_path`, `max_rounds`, `dev_split`, `holdout_split`, `resume_from`).
  Verify the dataclass constructs with and without the `config` field.
- [x] 1.4 Update `application/use_cases/run_cycle/__init__.py` and `__all__` to
  export `run_cycle`, `RunCycleDeps`, `RunCycleInput`, `RunCycleResult`, and
  `CycleConfig`. Do not export internal modules (`steps`, `state`, `report`,
  `config` module itself — only the `CycleConfig` type). Verify `__all__`
  contains only the public symbols.

## 2. State module

- [x] 2.1 Create `application/use_cases/run_cycle/state.py` with a
  `CycleState` dataclass (mutable, not frozen) holding: `round_counter: int`,
  `active_version: PromptArtifact`, `active_version_path: str`,
  `thesis_bank_path: str`, `clusters_path: str`,
  `candidate_queue: list[dict[str, Any]]`, `rollback_counter: int`,
  `latest_report_path: str`, `accepted_history: list[str]`,
  `rollback_history: list[dict[str, Any]]`, `run_id: str`. Verify it
  constructs with defaults and serializes via `dataclasses.asdict`.
- [x] 2.2 Implement `dump_state(state: CycleState, run_repository: RunRepository,
  output_dir: str) -> str` that serializes `CycleState` to a JSON file
  `state_{run_id}_{timestamp}.json` via `run_repository`. Verify the file is
  valid JSON and the filename contains `run_id` and a timestamp.
- [x] 2.3 Implement `load_state(path: str) -> CycleState` that restores a
  `CycleState` from a dumped JSON file, rebuilding the `PromptArtifact` from
  the stored version dict. Verify that loading a file written by `dump_state`
  reproduces `round_counter`, `active_version`, `candidate_queue`, and
  `rollback_counter`.
- [x] 2.4 Implement `update_in_prompt_flags(thesis_bank_path: str,
  accepted_cluster_ids: set[str]) -> None` that loads the thesis bank, sets
  `in_prompt=true` for theses whose `cluster_id` is in
  `accepted_cluster_ids` and `false` for all others, and persists the updated
  bank. Verify theses in accepted clusters get `in_prompt=true` and others get
  `false` on reload.

## 3. Report and logging module

- [x] 3.1 Create `application/use_cases/run_cycle/report.py` with
  `write_report(round_number, active_version_at_start, new_version, decision,
  metrics_dev_active, metrics_dev_new, metrics_holdout_active,
  metrics_holdout_new, rollback_counter, changed_decisions, next_action,
  run_id, run_repository, output_dir) -> str` that serializes to
  `report_{run_id}_round{N}_{timestamp}.json` via `run_repository`. Verify the
  file is valid JSON, the filename matches the pattern, and all fields
  round-trip on load.
- [x] 3.2 Implement `load_report(path: str) -> dict[str, Any]` that restores a
  per-round report without recomputation. Verify loading a file written by
  `write_report` reproduces every field.
- [x] 3.3 Implement `write_summary(state, final_dev_metrics,
  final_holdout_metrics, stop_reason, all_changed_decisions, run_id,
  run_repository, output_dir) -> str` that serializes to
  `summary_{run_id}_{timestamp}.json` with `total_rounds`, `stop_reason`,
  `final_active_version`, `final_dev_metrics`, `final_holdout_metrics`,
  `accepted_history`, `rollback_history`, `all_changed_decisions`. Verify the
  file is valid JSON and the stop reason is one of the four enumerated values.
  Reject other stop reasons with a `CycleOrchestratorError`.
- [x] 3.4 Implement `log_event(event: str, round: int, details: dict, log_path:
  str) -> None` that appends one JSON object (`{timestamp, round, event,
  details}`) to `cycle_log_{run_id}_{timestamp}.jsonl`. Use
  `logging.getLogger(__name__)` for runtime logging; do not call
  `logging.basicConfig`. Verify the file is append-only (two calls produce two
  lines) and each line is valid JSON with timestamp and round.
- [x] 3.5 Implement `read_cycle_log(path: str, round: int | None = None) ->
  list[dict[str, Any]]` that reads the JSONL log and returns all events, or
  only events for a given round. Verify: no filter returns all events;
  filtering by round N returns only that round's events.

## 4. Round execution — the nine-step sequence

- [x] 4.1 Create `application/use_cases/run_cycle/steps.py` with
  `run_active_on_dev(state, deps, config, config_path) -> str` (step 1) that
  instantiates `BaselineRunner` with the active `PromptArtifact` on the dev
  split and returns the results artifact path. Verify it produces a
  `results_*_{version}_dev_*.jsonl` artifact.
- [x] 4.2 Implement `compute_dev_metrics(results_path, config_path) -> dict`
  (step 2) that calls `compute_metrics` from `application.services.metrics`
  and returns the metrics dict. Verify it returns `accuracy`, `macro_f1`,
  `minority_f1`, and confusion matrix for an existing results artifact.
- [x] 4.3 Implement `update_thesis_collection(results_path, config_path,
  refined_path=None) -> tuple[str, str]` (step 3) that runs
  `analyze_theses.analyze` on the active results (or refined theses when
  `refined_path` is provided) and returns the updated thesis bank path and
  clusters path. Verify it produces a `thesis_bank_*.json` dump and a clusters
  artifact.
- [x] 4.4 Implement `select_candidates(thesis_bank_path, config_path) ->
  tuple[str, list[dict]]` (step 4) that runs
  `select_candidates.select_candidates` on the thesis bank and returns the
  candidate artifact path plus the candidate queue populated from the
  artifact. Verify it produces a `rule_candidates_*.json` artifact and a
  non-empty queue.
- [x] 4.5 Implement `compose_new_version(candidate, active_prompt, config_path,
  deps) -> tuple[PromptArtifact, str]` (step 5) that calls
  `compose_prompt.compose` on the candidate and the active prompt's base
  layers, writes the prompt version via `write_prompt_version`, and returns
  the new `PromptArtifact` plus its artifact path. Verify with a mocked
  composer that it returns a `PromptArtifact` with an incremented version.
- [x] 4.6 Implement `run_new_on_dev(new_prompt, config, config_path) -> str`
  (step 6) that runs the new prompt version on dev via `BaselineRunner` and
  returns the results artifact path. Verify it produces a results artifact
  carrying the new version in its filename.
- [x] 4.7 Implement `decide_and_update(state, new_prompt, metrics_new,
  metrics_active, config, config_path) -> tuple[str, str | None, list[dict]]`
  (step 7–8) that calls `compare_versions.decide` with
  `decision_metric` and `tie_breaker_metric`, updates the rollback counter via
  `update_rollback_count`, and on accept updates the active version via
  `deps.prompt_repository.activate` and calls `update_in_prompt_flags`; on
  rollback pops the next candidate via `next_candidate`. Return
  `(decision, reason, changed)`. Verify: accept resets the rollback counter
  and activates the version; rollback increments the counter and pops a
  candidate.
- [x] 4.8 Implement `run_holdout(prompt_artifact, config, config_path) -> dict`
  that runs a prompt on the holdout split via `BaselineRunner` and computes
  metrics. Verify it produces holdout metrics with the same keys as dev
  metrics. This is logged only, not used in the decision.
- [x] 4.9 Implement `run_round(state, deps, config, config_path) -> CycleState`
  that chains steps 1–9: run active on dev, compute metrics, optionally
  refine theses (when `config.use_teacher_refinement`), update thesis
  collection, select candidates, compose, run new on dev, decide, update
  state, run holdout for both versions, and write the per-round report. On
  rollback within the round, consume the candidate queue (steps 5–7) until a
  candidate is accepted or the queue is exhausted. Log `round_start` and
  `round_end` events. Return the updated `CycleState`. Verify with mocked
  sub-steps that all nine steps execute in order and the round counter
  increments by one.

## 5. Cycle loop — counters, queue, rollback, stop

- [x] 5.1 Implement `check_stop(state, config) -> tuple[bool, str | None]` in
  `run_cycle.py` that returns `(True, "max_rounds_reached")` when
  `state.round_counter >= config.max_rounds`; `(True,
  "max_rollbacks_reached")` when
  `state.rollback_counter >= config.max_consecutive_rollbacks`; `(True,
  "candidate_queue_exhausted")` when a rollback left the queue empty;
  `(False, None)` otherwise. Verify each stop reason fires at the right
  threshold and a healthy state returns `(False, None)`.
- [x] 5.2 Implement `load_initial_state(deps, config, config_path) ->
  CycleState` that loads the active prompt via `deps.prompt_repository`
  (or `CLASSIFICATION_PROMPT_V0` when no active version exists), the dataset
  via `deps.dataset_repository`, and returns a `CycleState` with empty thesis
  collection, round counter 0, rollback counter 0, and empty histories. Raise
  `CycleOrchestratorError` naming the missing artifact when the active version
  or dataset is not found. Verify it returns v0 active with zero counters, and
  raises on missing active prompt and missing dataset.
- [x] 5.3 Implement the `run_cycle(deps, run_input) -> RunCycleResult` body in
  `run_cycle.py`: load or resume `CycleState`, loop while
  `check_stop` returns `(False, _)`, call `run_round`, increment the round
  counter, dump state when `config.dump_state_after_each_round`, log events.
  On unrecoverable exception, stop with `unrecoverable_error`. On stop, write
  the final summary via `write_summary` and return `RunCycleResult`. Verify
  with mocked deps that the loop runs `max_rounds` rounds and stops with
  `max_rounds_reached`.
- [x] 5.4 Implement resume: when `run_input.resume_from` is set, load state via
  `load_state`, set the round counter to the dumped value, and continue from
  the next round. Verify with a mocked state at round 3 that resume starts at
  round 4 and does not re-run rounds 1–3, and that the thesis collection,
  clusters, active version, and rollback counter match the dumped state.

## 6. Optional teacher refinement integration

- [x] 6.1 Implement `refine_theses_step(results_path, run_id, prompt_version,
  deps, config_path) -> str` in `steps.py` that calls
  `refine_theses(deps.teacher_llm_client, ...)` when
  `deps.teacher_llm_client is not None` and returns the refined artifact path.
  When `deps.teacher_llm_client is None`, return `results_path` unchanged.
  Verify: with a mock teacher client it returns a refined path; with `None` it
  returns the original path.
- [x] 6.2 Wire the refinement step into `run_round` between step 1 (baseline
  run) and step 3 (thesis analysis) when `config.use_teacher_refinement` is
  `True`. Pass the refined path to `update_thesis_collection`. Verify: when
  enabled, the analyzer receives refined theses; when disabled, it receives
  raw theses.

## 7. Composition root wiring

- [x] 7.1 Expand `build_cli_deps` in `interfaces/cli/main.py` to assemble all
  ports: `AsyncTask` (llm_client), `PromptStore` (prompt_repository),
  `FileRunRepository` (run_repository), `normalize_theses` (normalizer), and
  `TeacherClient` (teacher_llm_client, when `[teacher].endpoint` is present,
  else `None`). For `dataset_repository` and `embedding_client`, pass the
  existing module-level functions or a thin adapter wrapping the existing
  use-case modules. Return the full `RunCycleDeps`. Verify the returned deps
  object has all fields populated and `teacher_llm_client` is `None` when the
  teacher endpoint is absent.
- [x] 7.2 Build `CycleConfig` in the composition root from
  `CycleConfig.from_toml(config_path)` and pass it through `RunCycleInput`.
  Override `max_rounds` from the CLI `--rounds` flag when provided. Verify
  the `RunCycleInput` carries the fully populated `CycleConfig`.
- [x] 7.3 Remove the `except NotImplementedError` catch at `main.py:177` and
  the associated stub message. Let real exceptions propagate to the existing
  `except Exception` handler. Verify the CLI no longer prints the stub
  message and that `run_cycle` is called with the full deps object.
- [x] 7.4 Add `decision_metric`, `tie_breaker_metric`, and
  `use_teacher_refinement` keys to the `[cycle_orchestrator]` section of
  `config.toml` with defaults (`macro_f1`, `minority_f1`, `false`) and inline
  comments. Verify the file parses with `tomllib.load` and
  `CycleConfig.from_toml` reads the new keys.

## 8. Unit tests with mock ports

- [x] 8.1 Create `tests/unit/run_cycle/__init__.py` and
  `tests/unit/run_cycle/conftest.py` with mock implementations of every port
  (`MockLLMClient`, `MockPromptRepository`, `MockRunRepository`,
  `MockDatasetRepository`, `MockEmbeddingClient`, `MockNormalizer`,
  `MockTeacherLLMClient`) and a `build_mock_deps()` fixture returning a
  `RunCycleDeps` with all mocks. Verify the fixture constructs without error
  and all mocks satisfy their port protocols.
- [x] 8.2 Write `tests/unit/run_cycle/test_steps.py` verifying the nine-step
  sequence executes in order with mocked sub-steps. Assert: step 1 produces a
  dev results path, step 2 returns metrics, step 3 returns thesis + clusters
  paths, step 4 returns candidates, step 5 returns a new `PromptArtifact`,
  step 6 produces a new dev results path, step 7 returns a decision, and
  `run_round` increments the round counter by one.
- [x] 8.3 Write `tests/unit/run_cycle/test_counters.py` verifying: the round
  counter increments on completed rounds and stops at `max_rounds`; the
  rollback counter increments on rollback, resets on accept, and stops at
  `max_consecutive_rollbacks`; `check_stop` returns the correct reason for
  each threshold.
- [x] 8.4 Write `tests/unit/run_cycle/test_queue.py` verifying: the candidate
  queue is consumed in order; the queue is refilled at the start of each
  round; exhaustion on rollback stops the cycle with
  `candidate_queue_exhausted`.
- [x] 8.5 Write `tests/unit/run_cycle/test_report.py` verifying: `write_report`
  produces a valid JSON file with all fields; `load_report` round-trips;
  `write_summary` produces a valid summary with an enumerated stop reason and
  rejects invalid reasons; `log_event` is append-only with timestamp and
  round; `read_cycle_log` filters by round.
- [x] 8.6 Write `tests/unit/run_cycle/test_state.py` verifying: `dump_state`
  produces a valid JSON file; `load_state` reproduces all fields;
  `update_in_prompt_flags` sets flags correctly; resume from a dumped state
  at round 3 starts at round 4 and preserves thesis collection, clusters,
  active version, and rollback counter.
- [x] 8.7 Write `tests/unit/run_cycle/test_error_handling.py` verifying: a
  single example failure (mock returns invalid JSON) is recorded as failed
  and the cycle continues; an unrecoverable exception stops the cycle with
  `unrecoverable_error`; `stop_on_first_error=true` stops on any step failure.

## 9. Integration and static checks

- [x] 9.1 Run `lint-imports` and verify it passes: `run_cycle` and its internal
  modules import only from `application` (ports, use cases, services) and
  `domain`, not from `infrastructure` or `interfaces`.
- [x] 9.2 Run `mypy` on the `run_cycle` package and `interfaces/cli/main.py`
  and verify it passes with no new errors. Address the `AsyncTask` /
  `LLMClient` signature mismatch with the existing `# type: ignore[arg-type]`
  convention in the composition root only.
- [x] 9.3 Run `ruff check` and `ruff format --check` on the new and modified
  files and verify they pass.
- [x] 9.4 Run the full unit test suite (`pytest tests/unit/run_cycle/`) and
  verify all tests pass with no inference server running.
- [x] 9.5 Run the orchestrator end-to-end for `max_rounds=1` against a running
  LLM server on the small prepared dataset. Verify: one round report is
  written with all fields, a state dump is written, a final summary is
  written with a valid stop reason, the cycle log contains `round_start` and
  `round_end` events, and the active version after the round is either v0
  (rollback) or a composed version (accept).
- [x] 9.6 Run the orchestrator for `max_rounds=5` end-to-end on the small
  dataset. Verify the cycle completes, five round reports are written, the
  final summary's `total_rounds` is 5, and the stop reason is
  `max_rounds_reached`. Then resume from the round-3 state dump and verify
  the resumed cycle runs rounds 4–5 only and produces a summary with
  `total_rounds=5`.
