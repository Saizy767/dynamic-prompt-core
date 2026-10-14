# Tasks

## 1. Skip thesis extraction on holdout

- [x] 1.1 Add `skip_extraction: bool = False` parameter to `BaselineRunner.__init__` in `runner.py` and store it as `self._skip_extraction`. Verify the constructor accepts the flag and defaults to False.
- [x] 1.2 In `BaselineRunner._process_example`, when `self._skip_extraction` is True, skip the `extract_theses_detailed` call and set `theses_raw=[]`, `theses_norm=[]`, `extract_status="skipped"`, `extract_cr.latency_ms=0.0`, `raw_extract=None`. Only run the classification-scoring block. Verify with a unit test that no extraction LLM call is made and the result has `extract_status="skipped"`.
- [x] 1.3 In `steps.py:run_holdout`, pass `skip_extraction=True` to the `BaselineRunner` constructor. Verify with a unit test that `run_holdout` passes the flag.
- [x] 1.4 Update `test_steps.py:test_run_holdout_threads_scorer_policy_candidates` to assert `captured["skip_extraction"] is True`. Verify the test passes.

## 2. Existing-rules carry-over in composer

- [x] 2.1 Add `existing_rules: list[str] | None = None` parameter to `compose()` in `composer.py`. Prepend `existing_rules` to `accepted_rules` before calling `assemble_prompt`. Verify with a unit test that existing rules appear in the assembled prompt alongside the new rule.
- [x] 2.2 In `steps.py:compose_new_version`, extract existing rules from `active_prompt.layers.rules` (default to `[]` when `layers` is None) and pass them as `existing_rules` to `compose()`. Verify with a unit test that rules from the active prompt are carried over.

## 3. Greedy incremental accumulation in `run_round`

- [x] 3.1 Restructure the while-loop in `steps.py:run_round` (lines 401-459): track `best_prompt`, `best_metrics`, `accumulated_source_candidates`, and `any_accepted`. On accept: update best, append source candidates, continue. On degradation: break. Verify the loop structure compiles.
- [x] 3.2 Add `max_rules` check before each compose call: load `PromptComposerConfig.from_config` and stop accumulation when `len(best_prompt.layers.rules) >= max_rules`. Verify with a unit test that accumulation stops at `max_rules`.
- [x] 3.3 After the loop, set `state.active_version = best_prompt` and update `in_prompt` flags for all `accumulated_source_candidates` at once. Verify with a unit test that flags are set for all accepted clusters.
- [x] 3.4 Set rollback counter: reset to 0 when `any_accepted` is True, increment by 1 when False. Verify with a unit test that a round with no acceptance increments the counter and a round with acceptance resets it.
- [x] 3.5 Update the holdout section to use `best_prompt` as the active version for holdout evaluation. Verify the holdout runs against the correct version.

## 4. Report and state updates

- [x] 4.1 Update the `write_report` call in `run_round` to reflect the final best version as `new_version` and the accumulated decision. Verify the report contains the correct version and decision.
- [x] 4.2 Update `log_event` calls in the accumulation loop to log each accept/rollback with the candidate cluster_id and accumulated rule count. Verify cycle log events are written correctly.

## 5. Test updates

- [x] 5.1 Update `test_steps.py:test_run_round_executes_steps_and_writes_report` to mock `decide_and_update` returning accept on first call, and verify the round completes with the correct report path. Verify the test passes.
- [x] 5.2 Add `test_run_round_accumulates_rules_on_accept` — mock two candidates with accept decisions, verify both rules are accumulated and the active version contains both. Verify the test passes.
- [x] 5.3 Add `test_run_round_stops_on_degradation` — mock first candidate accept, second candidate rollback, verify the active version contains only the first rule. Verify the test passes.
- [x] 5.4 Add `test_run_round_no_acceptance_increments_rollback` — mock all candidates as rollback, verify rollback counter is incremented and active version is unchanged. Verify the test passes.
- [x] 5.5 Add `test_runner_skip_extraction` in `test_runner.py` — construct a runner with `skip_extraction=True`, process one example, verify `extract_status="skipped"`, `theses_raw=[]`, `theses_norm=[]`. Verify the test passes.

## 6. Lint, types, and integration

- [x] 6.1 Run `ruff check` and `ruff format` on all changed files. Verify no lint errors.
- [x] 6.2 Run `mypy src/` on changed files. Verify no type errors.
- [x] 6.3 Run `pytest tests/unit/run_cycle/ tests/unit/run_baseline/ -x`. Verify all tests pass.
- [x] 6.4 Run `lint-imports` to verify layer dependencies are intact. Verify no import-cycle errors.
