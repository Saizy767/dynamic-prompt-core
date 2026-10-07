# Proposal

## Why

The `run_cycle` use case scaffold exists at `application/use_cases/run_cycle/`
with a typed signature (`RunCycleDeps`, `RunCycleInput`, `RunCycleResult`,
`run_cycle`) but the body raises `NotImplementedError`. The dependency object
is incomplete (only `llm_client` + `task_store`), the composition root
(`interfaces/cli/main.py`) catches the `NotImplementedError` and prints a
stub message, and the synced spec (`openspec/specs/stage3-cycle-orchestrator/`)
describes the old single-module `cycle_orchestrator.py` design rather than the
migrated layered architecture. All Stage 0–4 components are implemented and
testable individually, but nothing drives them as a repeatable end-to-end
optimization loop. This change implements `run_cycle` as the central use case
that orchestrates the full cycle and brings the spec into alignment with the
architectural contracts.

## What Changes

- **Implement `run_cycle`** in `application/use_cases/run_cycle/` as the
  end-to-end optimization loop. The use case receives all dependencies via a
  typed `RunCycleDeps` object and a typed `RunCycleInput`/`CycleConfig`; it
  does not read `config.toml` or instantiate infrastructure.
- **Expand `RunCycleDeps`** with the full port set: `llm_client`,
  `prompt_repository`, `run_repository`, `dataset_repository`,
  `embedding_client`, `normalizer`, and optional `teacher_llm_client`. Remove
  the unused `task_store` field. The deps object is a frozen dataclass of
  ports — no concrete infrastructure types.
- **Add a typed `CycleConfig`** dataclass carrying `max_rounds`,
  `max_consecutive_rollbacks`, `decision_metric`, `tie_breaker_metric`,
  artifact paths, `run_id`, and the behavioral toggles
  (`stop_on_first_error`, `dump_state_after_each_round`,
  `use_teacher_refinement`). Configuration is loaded in the composition root
  and passed in; the use case never opens `config.toml`.
- **Split the implementation** into cohesive internal modules:
  `run_cycle.py` (public entry point + loop), `steps.py` (the nine-step round
  sequence), `state.py` (`CycleState`, dump/load), `report.py` (per-round
  report + final summary). Only `run_cycle` and the result types are exported
  via `__init__.py` / `__all__`.
- **Execute each round** as a fixed nine-step sequence calling the existing
  use cases / services via their public APIs: run active on dev, compute
  metrics, update thesis bank + clusters, select candidates, compose new
  version, run new on dev, decide accept/rollback, update active version or
  consume next candidate, write per-round report. No logic is duplicated.
- **Thread the round counter, candidate queue, and rollback counter** across
  rounds. Stop reasons: `max_rounds_reached`, `max_rollbacks_reached`,
  `candidate_queue_exhausted`, `unrecoverable_error`.
- **On acceptance**, activate the new version via `prompt_repository`,
  archive the previous, and update `in_prompt` flags in the thesis
  collection.
- **Produce per-round reports and a final summary** via `run_repository`.
  Dump reloadable `CycleState` after each round; support resuming from a
  dumped state (skip completed rounds).
- **Log cycle events** via `logging.getLogger(__name__)` (append-only JSONL
  artifact). The module SHALL NOT call `logging.basicConfig`.
- **Wire all dependencies in the composition root**
  (`interfaces/cli/main.py`): expand `build_cli_deps` to assemble every port
  implementation and pass it through `RunCycleDeps`. Remove the
  `NotImplementedError` catch.
- **Support optional teacher refinement** (`use_teacher_refinement`): when
  enabled, `refine_theses` runs between the baseline run and thesis analysis;
  refined theses feed the analyzer. Disabled by default.
- **Isolate per-example errors** (parse failure, timeout, HTTP error): the
  example is recorded as failed and the cycle continues. Only unrecoverable
  use-case-level exceptions stop the cycle.
- **Add unit tests** with mock port implementations covering step sequence,
  counters, rollback behavior, report/summary formation, and resumability.
  No inference server required.

## Capabilities

### New Capabilities
<!-- None — this change implements an existing capability whose spec is synced
     at openspec/specs/stage3-cycle-orchestrator/. -->

### Modified Capabilities
- `stage3-cycle-orchestrator`: Replaces the old single-module spec with
  architecture-aligned requirements: typed dependency object with the full
  port set, typed config input (no direct `config.toml` reads), internal
  module split (`steps.py`, `state.py`, `report.py`), composition root
  wiring in `interfaces/cli/main.py`, optional teacher refinement, per-example
  error isolation, resumability from dumped state, and mock-based
  testability. The functional behavior (nine-step round, counters, candidate
  queue, rollback state, reports, summary, state dump, logging) is retained
  and restated against the layered architecture.

## Impact

- **New code**: `application/use_cases/run_cycle/steps.py`,
  `application/use_cases/run_cycle/state.py`,
  `application/use_cases/run_cycle/report.py`,
  `application/use_cases/run_cycle/config.py` (`CycleConfig`),
  `tests/unit/run_cycle/` (mock ports + scenarios).
- **Modified code**:
  `application/use_cases/run_cycle/run_cycle.py` (implement body),
  `application/use_cases/run_cycle/run_cycle_deps.py` (expand port set),
  `application/use_cases/run_cycle/__init__.py` (update `__all__`),
  `application/ports/inbound/run_cycle_input.py` (add `CycleConfig` field),
  `interfaces/cli/main.py` (expand `build_cli_deps`, remove stub catch).
- **Config**: the existing `[cycle_orchestrator]` section in `config.toml`
  gains `decision_metric`, `tie_breaker_metric`, `use_teacher_refinement`.
  No sections removed.
- **Dependencies**: no new third-party dependencies. Reuses existing ports,
  infrastructure implementations, and stage use cases.
- **Artifacts**: `report_{run_id}_round{N}_{ts}.json`,
  `summary_{run_id}_{ts}.json`, `state_{run_id}_{ts}.json`,
  `cycle_log_{run_id}_{ts}.jsonl` written via `run_repository`.
- **Static checks**: `lint-imports` passes (use case depends on ports + sibling
  use cases, not infrastructure), `mypy` passes.
