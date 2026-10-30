# Tasks

## 1. Port and data contracts

- [x] 1.1 Create `application/ports/outbound/stop_criteria.py` with a
  `StopCriteria` port: a `typing.Protocol` with `@runtime_checkable` and a
  single `async def evaluate(self, context: StopEvaluationContext) ->
  StopDecision` method. Define `StopEvaluationContext` and `StopDecision` as
  frozen dataclasses in this module (design D4, D5). Verify the module imports
  only `typing` and `dataclasses` (no `infrastructure`), and that
  `isinstance(mock, StopCriteria)` is `True` for a matching mock.
- [x] 1.2 Create `application/use_cases/evaluate_stop_criteria/config.py` with
  `StopCriteriaConfig` (design D9): a frozen dataclass with `plateau_window=3`,
  `min_improvement=0.01`, `degradation_window=3`, `stagnation_window=3`,
  `max_total_rounds=10`, `max_total_tokens=None`, `enabled_criteria=None`,
  `decision_metric="macro_f1"`. Add a `from_toml(config_path)` classmethod
  reading the `[stop_criteria]` section, mirroring `CycleConfig.from_toml`.
  Verify `StopCriteriaConfig.from_toml` parses a TOML with a `[stop_criteria]`
  section and that defaults are used when the section is absent.
- [x] 1.3 Create `application/use_cases/evaluate_stop_criteria/deps.py` with a
  frozen dataclass `EvaluateStopCriteriaDeps(run_repository: RunRepository)`
  (design D3). Verify it imports only `RunRepository` from the port and that
  constructing it with a mock repository succeeds.

## 2. Criteria implementation

- [x] 2.1 Create `application/use_cases/evaluate_stop_criteria/criteria.py`
  with seven pure functions (design D6), each taking a `StopEvaluationContext`
  and returning `str | None`: `_check_budget`, `_check_no_candidates`,
  `_check_plateau`, `_check_degradation`, `_check_stagnation`,
  `_check_rollback_streak`. Define `CHECK_ORDER` as the deterministic list
  (critical → qualitative → rollback). Verify the module imports only the
  context/config dataclasses and stdlib (no `infrastructure`, no `logging`).
- [x] 2.2 Verify `_check_plateau` against synthetic histories: (a) `macro_f1`
  flat for `plateau_window` rounds → returns `plateau_detected`; (b) an
  improvement > `min_improvement` resets the window; (c) only
  `plateau_window - 1` flat rounds → returns `None`; (d) a custom
  `decision_metric` is used when configured. Cover these as unit tests in
  `tests/unit/evaluate_stop_criteria/test_criteria_plateau.py`.
- [x] 2.3 Verify `_check_degradation` against synthetic histories: (a)
  `macro_f1` decreasing for `degradation_window` rounds → returns
  `metric_degradation`; (b) a single drop followed by an increase → returns
  `None`; (c) exactly `degradation_window` decreases → returns
  `metric_degradation`. Cover as unit tests in
  `tests/unit/evaluate_stop_criteria/test_criteria_degradation.py`.
- [x] 2.4 Verify `_check_no_candidates`: (a) queue empty and
  `new_candidates_found=0` → returns `no_candidates_available`; (b) queue empty
  but `new_candidates_found > 0` → returns `None`; (c) queue non-empty →
  returns `None`. Cover as unit tests.
- [x] 2.5 Verify `_check_budget`: (a) `round_counter > max_total_rounds` →
  returns `budget_exhausted`; (b) `total_teacher_tokens > max_total_tokens` →
  returns `budget_exhausted`; (c) `max_total_tokens=None` → token check is
  skipped; (d) neither exceeded → returns `None`. Cover as unit tests.
- [x] 2.6 Verify `_check_stagnation`: (a) identical rule-id sets for
  `stagnation_window` rounds → returns `rule_stagnation`; (b) a rule added in
  the last round → returns `None`; (c) a rule removed → returns `None`; (d)
  fewer than `stagnation_window` rounds of history → returns `None`. Cover as
  unit tests.
- [x] 2.7 Verify `_check_rollback_streak`: (a) `rollback_counter >=
  max_consecutive_rollbacks` → returns `rollback_streak`; (b) below the limit
  → returns `None`. Cover as unit tests.

## 3. Use case — combined decision and artifact

- [x] 3.1 Create `application/use_cases/evaluate_stop_criteria/evaluate.py`
  with `async def evaluate_stop_criteria(deps: EvaluateStopCriteriaDeps,
  context: StopEvaluationContext) -> StopDecision` (design D6, D5). The
  function runs each criterion in `CHECK_ORDER` that is in
  `context.config.enabled_criteria` (or all when `None`), collects all
  triggered reasons, and builds a `StopDecision` with `should_stop=True` and
  the first triggered reason, or `should_stop=False` when none trigger. When
  `context.metric_history` is empty, return `continue` immediately. Verify
  with a synthetic context where two criteria trigger: the decision's `reason`
  is the first in `CHECK_ORDER` and `triggered_criteria` lists both.
- [x] 3.2 Create `application/use_cases/evaluate_stop_criteria/artifact.py`
  with `write_stop_decision_artifact(decision: StopDecision, run_repository:
  RunRepository, output_dir: str) -> str` (design D7). Writes a single JSON
  file `stop_decision_{run_id}_{timestamp}.json` with `round`, `reason`,
  `triggered_criteria`, `metric_snapshot` (last `plateau_window` entries),
  `counters`, `run_id`, `timestamp`. Verify the file is written and
  JSON-loadable with all required fields.
- [x] 3.3 Wire artifact writing into `evaluate_stop_criteria`: when
  `should_stop` is `True`, call `write_stop_decision_artifact` via
  `deps.run_repository`. When `False`, do not write. Verify with a mock
  repository that `save_results` (or the file write) is called on `stop` and
  not called on `continue`.
- [x] 3.4 Add logging to `evaluate_stop_criteria`: use
  `logging.getLogger(__name__)` and append a `stop_evaluation` event to the
  cycle log via `log_event` from `run_cycle.report` (design D10). The event
  records `triggered_criteria`, `decision`, and `reason`. Verify the module
  does not call `logging.basicConfig` and that the log event is appended.
- [x] 3.5 Create `application/use_cases/evaluate_stop_criteria/__init__.py`
  exporting `evaluate_stop_criteria`, `StopDecision`,
  `EvaluateStopCriteriaDeps`, `StopCriteriaConfig`, `StopEvaluationContext`
  via `__all__`. Verify `from dynamic_prompt_core.application.use_cases
  .evaluate_stop_criteria import evaluate_stop_criteria` works and that
  internal modules (`criteria`, `artifact`) are not in `__all__`.

## 4. Combined-decision and use-case unit tests

- [x] 4.1 Add `tests/unit/evaluate_stop_criteria/test_combined.py` covering
  the deterministic check order: (a) `budget_exhausted` + `plateau_detected`
  both trigger → reason is `budget_exhausted`; (b) `plateau_detected` +
  `rollback_streak` → reason is `plateau_detected`; (c) no criteria →
  `continue`; (d) `enabled_criteria=["plateau_detected"]` → only plateau is
  checked, a triggered budget criterion is ignored. Verify all assertions
  pass.
- [x] 4.2 Add `tests/unit/evaluate_stop_criteria/test_use_case.py` covering:
  (a) empty `metric_history` → `continue` with no artifact written; (b) a
  `stop` decision writes the artifact via the mock repository; (c) a
  `continue` decision writes no artifact; (d) the `StopDecision` fields
  (`round_number`, `metric_snapshot`, `counters`) are populated correctly.
  Verify all assertions pass with no inference server.

## 5. Orchestrator integration

- [x] 5.1 Modify `application/use_cases/run_cycle/run_cycle.py` to accumulate
  `metric_history`, `decision_history`, and `rule_set_history` lists across
  rounds (design D8). After each `run_round`, append the round's
  `metrics_dev_new` (or active metrics on rollback) and decision from the
  report at `state.latest_report_path`, and the rule-id set from
  `state.active_version`. Verify with a mock that the lists grow by one entry
  per round and contain the expected values.
- [x] 5.2 Add `EvaluateStopCriteriaDeps` as an optional field on `RunCycleDeps`
  (`stop_criteria_deps: EvaluateStopCriteriaDeps | None = None`) and a
  `StopCriteriaConfig` on `CycleConfig` (`stop_criteria_config:
  StopCriteriaConfig | None = None`). Verify `RunCycleDeps` and `CycleConfig`
  construct with and without the new optional fields.
- [x] 5.3 In `run_cycle`, after `state.round_counter += 1` and the state dump,
  when `deps.stop_criteria_deps` is not `None`, build a
  `StopEvaluationContext` from the accumulated history and `CycleState`, call
  `evaluate_stop_criteria`, and `break` with the returned reason when
  `should_stop` (design D8). When `None`, skip the evaluation (backward
  compatible). Verify a mock `evaluate_stop_criteria` returning `stop` causes
  the cycle to break with the reason, and `continue` allows the next round.
- [x] 5.4 Extend `VALID_STOP_REASONS` in `application/use_cases/run_cycle/
  report.py` to include `plateau_detected`, `metric_degradation`,
  `no_candidates_available`, `budget_exhausted`, `rule_stagnation`,
  `rollback_streak`. Verify `write_summary` accepts the new reasons and that
  the existing reasons still work.
- [x] 5.5 Add unit tests in `tests/unit/run_cycle/test_stop_criteria.py`
  covering the orchestrator integration: (a) the cycle stops with
  `plateau_detected` when the mock evaluator returns `stop`; (b) the cycle
  continues when the mock returns `continue`; (c) when
  `stop_criteria_deps` is `None`, the cycle behaves as before (no evaluation
  call). Verify all assertions pass.

## 6. Composition root wiring

- [x] 6.1 In `interfaces/cli/main.py`, add wiring to construct a
  `StopCriteriaConfig` from the `[stop_criteria]` TOML section and an
  `EvaluateStopCriteriaDeps` with the `RunRepository`. Pass these through
  `RunCycleDeps` and `CycleConfig` (design D8). Verify `TeacherClient` /
  `StopCriteria` implementations are not instantiated in any module other than
  `main.py` (search the codebase for `EvaluateStopCriteriaDeps(` outside
  `main.py` and `deps.py`).
- [x] 6.2 Verify the CLI cycle command passes the stop-criteria config and
  deps through to `run_cycle` when the `[stop_criteria]` section is present,
  and passes `None` when it is absent. Verify `configure_logging()` is called
  in `main.py` and that no `evaluate_stop_criteria` module calls
  `logging.basicConfig`.

## 7. Static checks and integration

- [x] 7.1 Run `lint-imports` and verify it passes: the use case
  (`application.use_cases.evaluate_stop_criteria`) depends on
  `application.ports.outbound.stop_criteria` and
  `application.ports.outbound.run_repository`, not on `infrastructure`. The
  criteria module has no `infrastructure` imports. The orchestrator depends on
  the new use case package (allowed — both are in `application`).
- [x] 7.2 Run `mypy` and verify it passes with no errors on the new and
  modified modules. Pay attention to the `StopCriteria` protocol conformance,
  the `StopDecision` / `StopEvaluationContext` dataclass types, the optional
  `stop_criteria_deps` field on `RunCycleDeps`, and the extended
  `VALID_STOP_REASONS` set.
- [x] 7.3 Run the full unit test suite (`tests/unit/evaluate_stop_criteria/`
  and `tests/unit/run_cycle/test_stop_criteria.py`) and verify all tests pass
  with no inference server. Run an end-to-end check with a mocked evaluator
  returning `stop` after round 2: verify the cycle stops with the returned
  reason, the `stop_decision` artifact is written, and the summary records the
  new reason.
