# Design

## Context

The cycle orchestrator (`application/use_cases/run_cycle/run_cycle.py`) stops
via `check_stop` (`run_cycle.py:48`), which checks three hard limits at the top
of the loop before starting a round:

```python
def check_stop(state, config):
    if state.round_counter >= config.max_rounds:
        return True, "max_rounds_reached"
    if state.rollback_counter >= config.max_consecutive_rollbacks:
        return True, "max_rollbacks_reached"
    if not state.candidate_queue and state.rollback_counter > 0:
        return True, "candidate_queue_exhausted"
    return False, None
```

There is no content-based evaluation: the cycle cannot detect that `macro_f1`
has plateaued, that metrics are degrading, that the rule set has stopped
changing, or that the token budget is exhausted. The cycle keeps running until a
hard limit fires, wasting rounds and tokens after convergence.

The orchestrator already tracks the data a content-based evaluator needs:
`CycleState` (`state.py:26`) holds `round_counter`, `rollback_counter`,
`candidate_queue`, `accepted_history`, `rollback_history`, and
`active_version`. Each round produces a per-round report (`report.py:40`
`write_report`) with `metrics_dev_new` (containing `accuracy`, `macro_f1`,
`minority_f1`) and `decision`. The reports are written to disk and
`state.latest_report_path` points to the most recent one. However, the history
of metrics and decisions across rounds is not currently accumulated in memory —
it lives in the report files on disk.

Established patterns to follow:
- **Ports**: `application/ports/outbound/*.py`, each a `typing.Protocol` with
  `@runtime_checkable` (e.g. `run_repository.py:6`, `llm_client.py`).
- **Dependency object**: frozen dataclass of ports, e.g. `RunCycleDeps`
  (`run_cycle_deps.py:27`).
- **Use case package**: `application/use_cases/<name>/` with `__init__.py`
  exporting via `__all__`, internal modules not in `__all__`
  (e.g. `run_cycle/`).
- **Config**: frozen dataclass with `from_toml` classmethod, e.g.
  `CycleConfig` (`config.py:23`).
- **Logging**: `logging.getLogger(__name__)`, never `logging.basicConfig`
  (architecture spec: "Logging").
- **Artifact I/O**: `run_repository.save_results` for JSONL, or direct `json.dump`
  for single JSON files (e.g. `write_report` at `report.py:77`).

## Goals / Non-Goals

**Goals:**
- Add a `StopCriteria` port and `evaluate_stop_criteria` use case that evaluate
  cycle history and return a continue/stop decision with a reason.
- Implement seven criteria: plateau, degradation, candidate exhaustion, budget,
  stagnation, rollback streak, and a deterministic combined decision.
- Persist a `stop_decision` artifact on `stop`.
- Wire the orchestrator to call the evaluation at the end of each round.
- Make all parameters configurable via `StopCriteriaConfig`.
- Keep the component fully testable with synthetic histories, no inference
  server.

**Non-Goals:**
- Accepting or rejecting prompt versions — that is `stage2-version-comparator`.
- Recomputing metrics — the component consumes metrics already produced by
  `stage1-metrics` and recorded in per-round reports.
- Visualizing metric dynamics — that is `stage3-cycle-observer`.
- Removing or changing the orchestrator's hard-limit checks (`max_rounds`,
  `max_consecutive_rollbacks`) — they remain in `check_stop` and fire before a
  round starts; the new evaluation fires after a round.
- Statistical significance testing (bootstrap, confidence intervals) — deferred
  to Stage 7.
- Auto-tuning criterion windows (meta-optimization) — deferred to Stage 7.

## Decisions

### D1: Port location — `application/ports/outbound/stop_criteria.py`
The `StopCriteria` port is a single file at
`application/ports/outbound/stop_criteria.py`, matching every existing outbound
port (`run_repository.py`, `llm_client.py`, `teacher_llm_client.py`, etc.). The
architecture spec requires ports in `application/ports/outbound/` (or `inbound/`)
with `typing.Protocol` + `@runtime_checkable`.

The port defines a single method:

```python
async def evaluate(
    self, context: StopEvaluationContext,
) -> StopDecision: ...
```

`StopEvaluationContext` and `StopDecision` are typed dataclasses defined in the
port module (or `application/schemas/stop_criteria.py` if they grow). The port
imports only `typing` and the dataclasses.

**Alternative**: a package `application/ports/outbound/stop_criteria/`. Rejected
— every existing port is a single module file; a package for one protocol is
inconsistent.

### D2: Use case package — `application/use_cases/evaluate_stop_criteria/`
The package mirrors `run_cycle/`:

```
application/use_cases/evaluate_stop_criteria/
    __init__.py          # exports evaluate_stop_criteria, StopDecision, ...
    evaluate.py          # the public function
    deps.py              # EvaluateStopCriteriaDeps
    config.py            # StopCriteriaConfig
    criteria.py          # the seven criterion functions
    artifact.py          # write_stop_decision_artifact
```

`__init__.py` exports `evaluate_stop_criteria`, `StopDecision`,
`EvaluateStopCriteriaDeps`, `StopCriteriaConfig`, `StopEvaluationContext` via
`__all__`. Internal modules (`criteria.py`, `artifact.py`) are not in `__all__`.

The public function:

```python
async def evaluate_stop_criteria(
    deps: EvaluateStopCriteriaDeps,
    context: StopEvaluationContext,
) -> StopDecision: ...
```

**Alternative**: a single `evaluate_stop_criteria.py` module. Rejected — the
criteria, artifact I/O, and config are cohesive but distinct; a package keeps
each module small and matches the `run_cycle/` precedent.

### D3: Dependency object — `EvaluateStopCriteriaDeps`
A frozen dataclass `EvaluateStopCriteriaDeps(run_repository: RunRepository)`,
mirroring `RunCycleDeps`. The use case receives the repository through this
object for artifact persistence. The evaluator itself is stateless — the
criteria are pure functions of the context, so no `StopCriteria` port
implementation is needed in `deps`; the use case IS the evaluator. The port
exists for testability and for future consumers that may swap the evaluator.

**Alternative**: pass `run_repository` as a bare argument. Rejected — the
architecture spec requires a typed dependency object when there are multiple
dependencies, and this leaves room to add ports without a signature change.

**Alternative**: implement a `StopCriteriaEvaluator` class that implements the
`StopCriteria` port and inject it via `deps`. Rejected for now — the criteria
are pure functions with no external dependencies; a class wrapper adds
indirection without benefit. The port is still defined so an adapter can wrap
the use case if a future consumer needs polymorphism.

### D4: `StopEvaluationContext` — the input contract
A frozen dataclass carrying everything the criteria need:

```python
@dataclass(frozen=True)
class StopEvaluationContext:
    round_counter: int
    rollback_counter: int
    max_consecutive_rollbacks: int
    candidate_queue_size: int
    new_candidates_found: int          # candidates found in the current round
    metric_history: list[dict[str, float]]  # [{accuracy, macro_f1, minority_f1}, ...] per round
    decision_history: list[str]        # ["accept", "rollback", ...] per round
    rule_set_history: list[frozenset[str]]  # rule ids in the active version, per round
    total_teacher_tokens: int
    config: StopCriteriaConfig
    run_id: str
```

The orchestrator builds this from `CycleState` and the per-round reports. The
metric and decision history are accumulated in the orchestrator (see D8).

**Alternative**: have the use case read report files from disk to reconstruct
history. Rejected — the orchestrator already has the data in memory; reading
files couples the use case to the filesystem and complicates testing.

### D5: `StopDecision` — the output contract
A frozen dataclass:

```python
@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str | None                 # None when continue
    triggered_criteria: list[str]      # all criteria that fired, in check order
    round_number: int
    metric_snapshot: list[dict[str, float]]  # last plateau_window entries
    counters: dict[str, int]           # {rounds, rollbacks, tokens, candidates}
```

When `should_stop` is `False`, `reason` is `None` and `triggered_criteria` is
empty. When `True`, `reason` is the first triggered criterion in the check
order and `triggered_criteria` lists all that fired.

### D6: Criteria as pure functions in `criteria.py`
Each criterion is a pure function
`f(context: StopEvaluationContext) -> str | None`, returning the reason string
when triggered or `None` when not:

- `_check_budget(ctx)` → `budget_exhausted` when
  `ctx.round_counter > ctx.config.max_total_rounds` or
  `ctx.total_teacher_tokens > ctx.config.max_total_tokens` (when set).
- `_check_no_candidates(ctx)` → `no_candidates_available` when
  `ctx.candidate_queue_size == 0` and `ctx.new_candidates_found == 0`.
- `_check_plateau(ctx)` → `plateau_detected` when the decision metric has not
  improved by more than `min_improvement` for `plateau_window` consecutive
  rounds.
- `_check_degradation(ctx)` → `metric_degradation` when the decision metric has
  decreased for `degradation_window` consecutive rounds.
- `_check_stagnation(ctx)` → `rule_stagnation` when the rule set has not changed
  for `stagnation_window` consecutive rounds (compared as `frozenset` equality).
- `_check_rollback_streak(ctx)` → `rollback_streak` when
  `ctx.rollback_counter >= ctx.max_consecutive_rollbacks`.

Plateau and degradation use the configured `decision_metric` (default
`macro_f1`) from `StopCriteriaConfig`, falling back to `macro_f1`.

The combined evaluation runs them in a fixed order and collects all triggers:

```python
CHECK_ORDER = [
    _check_budget,              # critical
    _check_no_candidates,       # critical
    _check_plateau,             # qualitative
    _check_degradation,         # qualitative
    _check_stagnation,          # qualitative
    _check_rollback_streak,     # rollback
]
```

Only criteria in `config.enabled_criteria` are evaluated. The first triggered
criterion in `CHECK_ORDER` is the `reason`; all triggered criteria are recorded.

**Alternative**: a strategy class per criterion. Rejected — pure functions are
simpler, testable, and have no state; a class hierarchy is overkill for seven
one-liners.

### D7: Artifact — `stop_decision_{run_id}_{timestamp}.json`
Written by `artifact.py:write_stop_decision_artifact(decision, run_repository,
output_dir)` only when `should_stop` is `True`. The artifact is a single JSON
file (not JSONL) matching `write_report` / `write_summary` conventions:

```json
{
  "round": 7,
  "reason": "plateau_detected",
  "triggered_criteria": ["plateau_detected", "rollback_streak"],
  "metric_snapshot": [{"accuracy": 0.82, "macro_f1": 0.74, "minority_f1": 0.61}, ...],
  "counters": {"rounds": 7, "rollbacks": 2, "tokens": 12500, "candidates": 0},
  "run_id": "cycle-20260101T000000Z",
  "timestamp": "20260101T000005Z"
}
```

The `metric_snapshot` contains the last `plateau_window` entries from
`metric_history`. The artifact is self-contained — no recomputation needed.

### D8: Orchestrator integration — accumulate history, call after each round
The orchestrator (`run_cycle.py`) is modified to:

1. Accumulate `metric_history` and `decision_history` in local lists across
   rounds. After each `run_round`, the orchestrator reads the metrics from the
   round's report (`state.latest_report_path`) and appends
   `metrics_dev_new` (or the active metrics on rollback) and the round's
   decision. The rule-set snapshot is derived from `state.active_version` (the
   set of rule ids in the version's layers) and appended to `rule_set_history`.
2. After `state.round_counter += 1` and the state dump, build a
   `StopEvaluationContext` from the accumulated history and `CycleState`.
3. Call `evaluate_stop_criteria(deps, context)`. If `should_stop`, set
   `stop_reason` and `break`.

The hard-limit `check_stop` remains at the top of the loop and fires first; the
content-based evaluation fires after the round. This preserves the existing
behavior and stop reasons while adding the new ones.

`EvaluateStopCriteriaDeps` is added to `RunCycleDeps` as an optional field
(`stop_criteria_deps: EvaluateStopCriteriaDeps | None = None`). When `None`,
the orchestrator skips the content-based evaluation (backward-compatible
default). The composition root constructs and passes it when the
`[stop_criteria]` config section is present.

**Alternative**: replace `check_stop` with the new component entirely. Rejected
— the user's non-goals explicitly keep `max_rounds` and
`max_consecutive_rollbacks` in the orchestrator config, checked before the
round. The component is additive.

**Alternative**: have the component read reports from disk. Rejected — the
orchestrator has the data in memory; disk reads couple the component to the
filesystem and complicate testing.

### D9: `StopCriteriaConfig` — frozen dataclass with `from_toml`
```python
@dataclass(frozen=True)
class StopCriteriaConfig:
    plateau_window: int = 3
    min_improvement: float = 0.01
    degradation_window: int = 3
    stagnation_window: int = 3
    max_total_rounds: int = 10
    max_total_tokens: int | None = None
    enabled_criteria: list[str] | None = None   # None = all
    decision_metric: str = "macro_f1"
```

`from_toml(config_path)` reads the `[stop_criteria]` section, mirroring
`CycleConfig.from_toml`. `enabled_criteria=None` means all criteria are enabled;
a list restricts evaluation to the named criteria. The composition root
constructs it and passes it through `RunCycleInput` or `RunCycleDeps`.

### D10: Logging — `logging.getLogger`, append to cycle log
The use case uses `log = logging.getLogger(__name__)` and never calls
`logging.basicConfig`. Each evaluation appends an event to the cycle log via
the existing `log_event` helper (`report.py:131`):

```python
log_event("stop_evaluation", round_number, {
    "triggered_criteria": [...],
    "decision": "stop" / "continue",
    "reason": ...,
}, log_path)
```

This reuses the orchestrator's append-only log rather than introducing a
separate one.

## Risks / Trade-offs

- **[History accumulation in orchestrator]** → The orchestrator gains local
  lists for metric, decision, and rule-set history. Mitigation: these are
  plain lists of small dicts/frozensets; memory is negligible for
  `max_rounds ≤ 10`. The lists are not persisted to `CycleState` (they can be
  rebuilt from report files on resume).
- **[Resume does not reconstruct history]** → On resume, the accumulated
  history lists start empty, so the content-based evaluation only sees rounds
  after the resume point. Mitigation: documented as a known limitation; a
  follow-up can rebuild history from report files on resume. The hard-limit
  checks still work on resume because `round_counter` is persisted.
- **[Rule-set comparison]** → Stagnation compares the set of rule ids in the
  active version across rounds. If the composer changes a rule's text without
  changing its id, stagnation will not detect it. Mitigation: acceptable for
  now — rule ids are stable identities; text changes without id changes are
  not expected in the current composer. A hash-based comparison is a future
  option (Open Question).
- **[Duplicate rollback check]** → The orchestrator's `check_stop` already
  stops on `max_consecutive_rollbacks` with reason `max_rollbacks_reached`
  before the round starts. The component's `rollback_streak` check fires after
  the round and would produce `rollback_streak`. In practice the orchestrator's
  check fires first, so `rollback_streak` is a fallback for the case where the
  rollback counter reaches the limit mid-round. Mitigation: documented; the two
  reasons are distinct and both valid.
- **[Extended stop-reason set is a breaking change]** → `VALID_STOP_REASONS`
  in `report.py:23` grows. Any downstream consumer that validates against the
  old set will reject the new reasons. Mitigation: the only consumer is
  `write_summary` itself; the cycle observer reads the summary but does not
  validate the reason against a fixed set.

## Open Questions

- **Stagnation comparison — rule ids vs. prompt hash?** Rule-id sets are
  simpler and align with the composer's identity model; a full-text hash is
  more sensitive. Deferred to implementation — the spec allows either as long
  as "the rule set does not change" is detectable.
- **Multiple reasons in the artifact — all triggered or only the first?** The
  spec requires all triggered criteria in `triggered_criteria` and the first as
  `reason`. This is settled in the spec; the open question is whether the
  summary should also carry all triggered criteria or just the reason. Deferred
  — the summary already carries `stop_reason`; adding the full list is a
  follow-up.
- **Token budget when teacher refinement is disabled.** When
  `use_teacher_refinement=false`, `total_teacher_tokens` is always 0, so the
  token-budget check never fires. This is correct but means the round-budget
  (`max_total_rounds`) is the only budget check. Acceptable — the round budget
  is independent of teacher usage.
- **Should `max_total_rounds` default differ from `CycleConfig.max_rounds`?**
  `StopCriteriaConfig.max_total_rounds` defaults to 10 while
  `CycleConfig.max_rounds` defaults to 5. The orchestrator's hard limit fires
  first in the default config. The budget check is a safety net for configs
  with a high `max_rounds`. No change needed — the defaults are independent.
