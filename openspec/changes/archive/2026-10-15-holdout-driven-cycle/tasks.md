# Tasks

## 1. Decision logic: strict improvement

- [x]  1.1 Modify `decide()` in `compare_versions/comparator.py` to accept only when the decision metric is strictly greater (`new > active`). Equal → rollback. Remove tie-breaker from the decision path; keep it in the reason string for logging. Verify: `pytest tests/unit/ -k "decide or comparator" -v` passes with updated assertions.
- [x] 1.2 Remove `update_rollback_count()` and `check_stop()` from `comparator.py` (rollback counter no longer used). Verify: `ruff check src/dynamic_prompt_core/application/use_cases/compare_versions/comparator.py` passes, no dangling imports.

## 2. Versioning: `vN_k` scheme

- [x] 2.1 Add `intermediate_version(cycle: int, cluster: int) -> str` returning `f"v{cycle}_{cluster}"` and `finalized_version(cycle: int) -> str` returning `f"v{cycle}"` to `compose_prompt/composer.py`. Replace `increment_version()` usage in `assemble_prompt()` to accept an explicit version string parameter. Verify: unit test that `intermediate_version(2, 3) == "v2_3"` and `finalized_version(2) == "v2"`.
- [x] 2.2 Update `compose()` and `assemble_prompt()` signatures to accept `version_str: str` instead of deriving it from `base_version`. Update `write_prompt_version()` to store the cycle/cluster info in metadata. Verify: `pytest tests/unit/ -k "composer or version" -v` passes.

## 3. State: new fields and renames

- [x] 3.1 In `run_cycle/state.py`: rename `round_counter` → `cycle_counter`, add `cycle_number: int = 0` and `cluster_index: int = 0`, remove `rollback_counter`, rename `last_round_new_candidates` → `last_cycle_new_candidates`. Update `to_dict()` and `load_state()` accordingly. `load_state()` SHALL use `.get()` with defaults for backward-compat. Verify: `pytest tests/unit/run_cycle/test_state.py -v` passes.
- [x] 3.2 Update `update_in_prompt_flags()` — no changes needed, but verify it still works with the updated state. Verify: existing test for in_prompt flags passes.

## 4. Config: renames and cleanup

- [x] 4.1 In `run_cycle/config.py`: rename `max_rounds` → `max_cycles`, remove `max_consecutive_rollbacks`, rename `dump_state_after_each_round` → `dump_state_after_each_cycle`, rename `duration_for_round()` → `duration_for_cycle()`. Update `from_toml()` to read `max_cycles` from `[cycle_orchestrator]`. Verify: `CycleConfig.from_toml("config.toml")` loads without error.
- [x] 4.2 In `ports/inbound/run_cycle_input.py`: rename `max_rounds` → `max_cycles`. Verify: `mypy src/dynamic_prompt_core/application/ports/inbound/run_cycle_input.py` passes.
- [x] 4.3 Update `config.toml` `[cycle_orchestrator]`: rename `max_rounds` → `max_cycles`, remove `max_consecutive_rollbacks`. Update `[version_comparator]`: remove `max_consecutive_rollbacks`. Verify: `tomllib.load` parses the file without error.

## 5. Cycle flow: holdout-driven accumulation

- [x] 5.1 In `run_cycle/steps.py`: add a `run_active_on_holdout()` step that runs the active version on holdout with `skip_extraction=True` and returns holdout metrics. This is the cached baseline for the accumulation loop. Verify: unit test that the step produces holdout metrics.
- [x] 5.2 Rewrite the accumulation loop in `run_round()` (rename to `run_cycle_step()`): each candidate composes `vN_k` using `intermediate_version(cycle_number, cluster_index)`, runs **only on holdout** (`skip_extraction=True`), and decides via `decide(holdout_new, holdout_best)`. Remove per-candidate dev runs (`run_new_on_dev` calls in the loop). Verify: `pytest tests/unit/run_cycle/test_steps.py -v` passes with updated mocks.
- [x] 5.3 After the accumulation loop: finalize `best_prompt` by renaming its version to `finalized_version(cycle_number)` via `assemble_prompt` with the explicit version string. Update `state.active_version` to the finalized version. Verify: unit test that the finalized version has the correct `vN` name.
- [x] 5.4 Track `cluster_index` in the accumulation loop (increment on each candidate attempt, reset to 0 at cycle start). Track `cycle_number` (increment after each cycle). Update `state.cycle_number` and `state.cluster_index`. Verify: unit test that cycle_number and cluster_index are correct after a cycle.

## 6. Stop logic: `max_cycles` + `no_improvement`

- [x] 6.1 In `run_cycle/run_cycle.py`: rewrite `check_stop()` to check `cycle_counter >= max_cycles` → `max_cycles_reached`. Remove `max_consecutive_rollbacks` and `candidate_queue_exhausted` checks. Verify: `pytest tests/unit/run_cycle/test_counters.py -v` passes with updated assertions.
- [x] 6.2 Add `no_improvement` detection: after `run_cycle_step()`, if `not any_accepted` (no candidate was accepted in the cycle), stop with `no_improvement`. Track `any_accepted` via a return value or state field from `run_cycle_step()`. Verify: unit test that a cycle with no acceptance triggers `no_improvement`.
- [x] 6.3 Update `RunCycleResult`: rename `total_rounds` → `total_cycles`. Update `write_summary()` call. Verify: `mypy src/dynamic_prompt_core/application/use_cases/run_cycle/run_cycle.py` passes.

## 7. Report: cycle/cluster format

- [x] 7.1 In `run_cycle/report.py`: update `write_report()` — rename `round_number` → `cycle_number`, add `base_version`, `finalized_version`, `attempts` (list of `{version, cluster_id, holdout_metric, decision}`), `accepted_cluster_ids`. Keep `metrics_dev_active` (from step 1) and holdout metrics. Verify: unit test that the report contains the new fields.
- [x] 7.2 Update `VALID_STOP_REASONS` to `{max_cycles_reached, no_improvement, unrecoverable_error, plateau_detected}`. Remove `max_rounds_reached`, `max_rollbacks_reached`, `candidate_queue_exhausted`. Verify: `pytest tests/unit/run_cycle/test_report.py -v` passes.
- [x] 7.3 Update `write_summary()`: rename `total_rounds` → `total_cycles`. Verify: summary test passes.

## 8. CLI and composition root

- [x] 8.1 In `interfaces/cli/main.py`: rename `--rounds` → `--cycles`, update `RunCycleInput` construction to use `max_cycles`, update log messages. Verify: `python -m dynamic_prompt_core.interfaces.cli.main cycle --help` shows `--cycles`.
- [x] 8.2 Update `run_cycle.py:load_initial_state()` and `run_cycle()` to use `cycle_counter` instead of `round_counter`, and `config.max_cycles` instead of `config.max_rounds`. Verify: `mypy src/dynamic_prompt_core/application/use_cases/run_cycle/run_cycle.py` passes.

## 9. Tests: update affected test files

- [x] 9.1 Update `tests/unit/run_cycle/test_counters.py`: replace `max_rounds` with `max_cycles`, remove `max_consecutive_rollbacks` tests, add `no_improvement` test. Verify: `pytest tests/unit/run_cycle/test_counters.py -v` passes.
- [x] 9.2 Update `tests/unit/run_cycle/test_steps.py`: update `run_round` → `run_cycle_step` references, mock holdout runs instead of dev runs for candidates, assert strict improvement, assert no per-candidate dev runs. Verify: `pytest tests/unit/run_cycle/test_steps.py -v` passes.
- [x] 9.3 Update `tests/unit/run_cycle/test_state.py`: use `cycle_counter` instead of `round_counter`, add `cycle_number` and `cluster_index` assertions. Verify: `pytest tests/unit/run_cycle/test_state.py -v` passes.
- [x] 9.4 Update `tests/unit/run_cycle/test_report.py` and `test_stop_criteria.py`: update stop reasons, field names. Verify: `pytest tests/unit/run_cycle/test_report.py tests/unit/run_cycle/test_stop_criteria.py -v` passes.
- [x] 9.5 Update `tests/unit/run_cycle/test_queue.py` and `test_error_handling.py`: rename `max_rounds` → `max_cycles` in fixtures. Verify: `pytest tests/unit/run_cycle/test_queue.py tests/unit/run_cycle/test_error_handling.py -v` passes.
- [x] 9.6 Update `tests/unit/run_cycle/conftest.py`: replace `max_rounds` with `max_cycles` in `CycleConfig` fixture. Verify: `pytest tests/unit/run_cycle/ -v` collects without errors.

## 10. Verification

- [x] 10.1 Run `ruff check src/` and `ruff format --check src/` — zero errors. Fix any issues.
- [x] 10.2 Run `mypy src/` — zero errors. Fix any type issues from the renames and signature changes.
- [x] 10.3 Run `lint-imports` — zero contract violations. Verify no new cross-layer imports.
- [x] 10.4 Run `pytest tests/unit/ -v` — all unit tests pass. Fix any remaining failures from the refactor.
- [x] 10.5 Run `pytest tests/integration/ -v -k "not gguf"` — integration tests pass (skip GGUF tests without the optional dependency).
