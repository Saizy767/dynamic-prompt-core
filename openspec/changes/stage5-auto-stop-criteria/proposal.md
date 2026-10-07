# Proposal

## Why

The cycle orchestrator (`stage3-cycle-orchestrator`) currently stops only on
hard limits: `max_rounds`, `max_consecutive_rollbacks`, and
`candidate_queue_exhausted` (`run_cycle.py:48` `check_stop`). It cannot detect
that the optimization has stopped making progress — a plateau in `macro_f1`,
sustained metric degradation, rule stagnation, or budget exhaustion — so the
cycle keeps burning rounds and tokens after it is already converged. This change
introduces a content-based stop-criteria component that evaluates the cycle's
metric and decision history at the end of each round and signals the orchestrator
to stop when continuing is pointless.

## What Changes

- **Add a `stop_criteria` port** in `application/ports/outbound/stop_criteria/`
  defining the contract for evaluating cycle state and returning a continue/stop
  decision. The port SHALL use `typing.Protocol` and SHALL NOT depend on
  `infrastructure`.
- **Add an `evaluate_stop_criteria` use case** in
  `application/use_cases/evaluate_stop_criteria/` that receives cycle history
  (per-round metrics, decision history, round/rollback counters, candidate-queue
  size, teacher-token total, criteria config) through a typed dependency object
  and returns a `StopDecision` (continue/stop + reason). The use case creates no
  concrete dependencies.
- **Implement seven criteria**: plateau detection (`plateau_detected`), metric
  degradation (`metric_degradation`), candidate exhaustion
  (`no_candidates_available`), budget exhaustion (`budget_exhausted`), rule
  stagnation (`rule_stagnation`), rollback streak (`rollback_streak`), and a
  deterministic combined decision that checks critical criteria first, then
  qualitative, then `rollback_streak`.
- **Persist a `stop_decision` artifact** via `run_repository` when the decision
  is `stop`, containing the round number, reason, recent metric values, counters,
  and the list of triggered criteria.
- **Wire the orchestrator to call `evaluate_stop_criteria`** at the end of each
  round, after `check_stop` and before starting the next round. When the
  evaluation returns `stop`, the orchestrator terminates the cycle with the
  returned reason. This is a **BREAKING** change to the orchestrator's enumerated
  stop reasons: the set grows from
  `{max_rounds_reached, max_rollbacks_reached, candidate_queue_exhausted, unrecoverable_error}`
  to also include `{plateau_detected, metric_degradation, no_candidates_available,
  budget_exhausted, rule_stagnation, rollback_streak}`.
- **Add a `StopCriteriaConfig`** (plateau_window, min_improvement,
  degradation_window, stagnation_window, max_total_rounds, max_total_tokens,
  enabled_criteria) loaded by the composition root and passed into the use case.
- **Restrict logging**: the component SHALL use `logging.getLogger` and SHALL NOT
  call `logging.basicConfig`. Evaluation events are appended to the cycle log.
- **Add unit tests** with synthetic metric histories covering each criterion,
  combined scenarios, the deterministic check order, and window resets. No
  inference server is required.

## Capabilities

### New Capabilities
- `stage5-auto-stop-criteria`: Content-based stop-criteria evaluation for the
  optimization cycle — port, use case, seven criteria, combined decision,
  artifact persistence, configuration, and logging.

### Modified Capabilities
- `stage3-cycle-orchestrator`: The orchestrator SHALL call
  `evaluate_stop_criteria` at the end of each round and terminate the cycle when
  the evaluation returns `stop`. The enumerated stop-reason set is extended with
  the new content-based reasons.

## Impact

- **New code**: `application/ports/outbound/stop_criteria.py` (port),
  `application/use_cases/evaluate_stop_criteria/` (use case, dependency object,
  result types, criteria implementation), tests under
  `tests/unit/evaluate_stop_criteria/`.
- **Modified code**: `application/use_cases/run_cycle/run_cycle.py` (calls
  `evaluate_stop_criteria` after each round), `application/use_cases/run_cycle/config.py`
  (carries a `StopCriteriaConfig`), `application/use_cases/run_cycle/report.py`
  (`VALID_STOP_REASONS` extended), `interfaces/cli/main.py` (composition root
  wires the `StopCriteriaEvaluator` and config).
- **Config**: new `[stop_criteria]` TOML section with `plateau_window`,
  `min_improvement`, `degradation_window`, `stagnation_window`,
  `max_total_rounds`, `max_total_tokens`, `enabled_criteria`.
- **Dependencies**: no new third-party dependencies. Reuses `RunRepository` (port)
  and stdlib `dataclasses` / `typing`.
- **Artifacts**: new `stop_decision_{run_id}_{timestamp}.json` written only on
  `stop`. Existing per-round reports, summaries, and state dumps are unchanged.
- **Upstream**: unchanged — consumes metric and decision history already produced
  by `stage1-metrics` and `stage2-version-comparator` and tracked by the
  orchestrator.
- **Downstream**: unchanged — `stage3-cycle-observer` continues to visualize
  cycle dynamics; it is not the consumer of stop decisions.
- **Static checks**: `lint-imports` passes (use case depends on the port and
  `run_repository`, not `infrastructure`), `mypy` passes.
