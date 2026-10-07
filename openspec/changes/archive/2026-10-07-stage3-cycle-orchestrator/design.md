# Design

## Context

The migration to the four-layer architecture is complete at the scaffold
level: `run_cycle/` has a typed signature (`RunCycleDeps`, `RunCycleInput`,
`RunCycleResult`, `run_cycle`) but the body is `NotImplementedError`, and
`RunCycleDeps` carries only `llm_client` + `task_store`. The composition root
(`interfaces/cli/main.py:60`) builds this partial deps object and catches the
`NotImplementedError`. The `refine_theses` use case is the exemplar of the
port-migrated pattern: a frozen `RefineThesesDeps` dataclass of ports, a typed
input/result, no infrastructure imports, wired in the composition root.

The sub-step use cases (`run_baseline`, `analyze_theses`, `select_candidates`,
`compose_prompt`, `compare_versions`) are implemented with full logic but are
not yet port-migrated — each has its own `XxxConfig.from_config()` and imports
infrastructure directly. They expose public functions used by the standalone
CLIs. The orchestrator calls these public functions; it does not re-implement
their logic. The `application/services/metrics/` package is port-free and
exported via `__init__.py` (`compute_metrics`, `compare_versions`).

Ports already defined in `application/ports/outbound/`: `LLMClient`,
`PromptRepository`, `RunRepository`, `DatasetRepository`, `EmbeddingClient`,
`Normalizer`, `TeacherLLMClient`. Concrete implementations: `AsyncTask`
(llm), `PromptStore` (prompt repo), `FileRunRepository` (run repo),
`normalize_theses` (normalizer). `EmbeddingClient` and `DatasetRepository`
have no concrete implementation yet — the orchestrator calls the existing
use-case modules directly for those concerns (see D6).

## Goals / Non-Goals

**Goals:**
- Implement `run_cycle` as the end-to-end loop, receiving all dependencies via
  `RunCycleDeps` and all configuration via `CycleConfig`.
- Call existing stage use cases / services through their public APIs without
  duplicating logic.
- Thread round counter, candidate queue, and rollback counter across rounds.
- Produce per-round reports, final summary, and reloadable state dumps.
- Support resuming from a dumped state.
- Wire all ports in the composition root.
- Be unit-testable with mock ports and no inference server.

**Non-Goals:**
- Port-migrating the sub-step use cases (`run_baseline`, `analyze_theses`,
  etc.) — they are called via their public functions as-is.
- Implementing concrete `EmbeddingClient` / `DatasetRepository` adapters —
  the orchestrator delegates to the existing modules for those concerns.
- Plateau detection, hyperparameter optimization, rule strengthening (Stage 5).
- MCP server or external interface (Stage 6).
- Multi-class / multi-label / distributed execution (Stage 6).
- Changing the public APIs of existing components.

## Decisions

### D1: Internal module split — `steps.py`, `state.py`, `report.py`, `config.py`
The orchestrator is cohesive (one loop) but internally complex (nine steps,
state threading, report/summary formation, serialization). Splitting into
internal modules improves readability without breaking the single public API:
`run_cycle.py` (entry + loop), `steps.py` (the nine-step round sequence),
`state.py` (`CycleState` + dump/load), `report.py` (per-round report + final
summary), `config.py` (`CycleConfig`). Only `run_cycle` and result types are
exported via `__all__`.

**Alternative**: a single `run_cycle.py` monolith. Rejected — at ~500+ lines
the state threading and report formation obscure the loop logic.

### D2: `RunCycleDeps` port set — full expansion
Expand `RunCycleDeps` to: `llm_client: LLMClient`,
`prompt_repository: PromptRepository`, `run_repository: RunRepository`,
`dataset_repository: DatasetRepository`, `embedding_client: EmbeddingClient`,
`normalizer: Normalizer`, `teacher_llm_client: TeacherLLMClient | None`. Drop
the unused `task_store` field. `teacher_llm_client` is optional (None when
refinement is disabled) so the deps object can be constructed without a teacher
endpoint. The deps object remains a frozen dataclass.

**Alternative**: keep `task_store` for backward compatibility. Rejected — no
caller uses it; keeping a dead port violates the "deps are the contract"
principle.

### D3: `CycleConfig` — typed config, loaded in composition root
Add `application/use_cases/run_cycle/config.py` with a frozen `CycleConfig`
dataclass: `max_rounds`, `max_consecutive_rollbacks`, `decision_metric`,
`tie_breaker_metric`, `dev_split`, `holdout_split`, `run_id`, `output_dir`,
`prompt_store_path`, `thesis_bank_path`, `candidate_queue_path`,
`stop_on_first_error`, `dump_state_after_each_round`, `use_teacher_refinement`.
The composition root builds `CycleConfig` from the `[cycle_orchestrator]` TOML
section and passes it through `RunCycleInput`. The use case never opens
`config.toml`.

**Alternative**: have the use case load its own config section. Rejected —
violates the architecture spec (composition root owns config; application
layer is infrastructure-free).

### D4: Round execution — call sub-step public functions directly
`steps.py` calls the existing use-case modules' public functions:
`BaselineRunner` (`run_baseline/runner.py`), `compute_metrics`
(`services/metrics/`), `analyze` (`analyze_theses/thesis_analyzer.py`),
`select_candidates` (`select_candidates/selector.py`), `compose`
(`compose_prompt/composer.py`), `decide` / `compare_metrics`
(`compare_versions/comparator.py`). These modules read their own config
sections from `config_path` (carried in `RunCycleInput`); the orchestrator
does not re-parse their sections. This is pragmatic: the sub-steps are not
port-migrated, but their public functions are stable and tested.

**Alternative**: port-migrate all sub-steps first, then call through ports.
Rejected — out of scope for this change and would block the orchestrator on a
much larger refactor. The orchestrator's own dependencies (LLM, repositories)
go through ports; the sub-steps are called as library functions.

### D5: Round counter — increment on completed rounds only
The counter increments after step 9 (report written + state dumped). A round
aborted mid-way by an unrecoverable error does not increment the counter; the
cycle stops with `unrecoverable_error`. This makes "total rounds" in the
summary equal to the counter and keeps resume simple (start at `counter + 1`).

### D6: Candidate queue — refilled at the start of each round
The selector runs in step 4 of every round and refills the queue from the
current thesis collection. This ensures candidates reflect the latest clusters.
On rollback within a round, the queue is consumed in order without re-running
the selector; the queue is refilled only when the next round starts.

### D7: Rollback counter — part of the dumped state
The rollback counter is stored in `CycleState` and restored on resume. It is
not recomputed from history (the history records decisions but not the
consecutive run length). Reset to zero on acceptance.

### D8: State dump — single JSON per round via `run_repository`
`state.py` serializes `CycleState` (round counter, active version, thesis
collection path, clusters path, candidate queue, rollback counter, latest
report path, accepted history, rollback history, run_id) to a single JSON
file. One file per round makes resume a single load. The path is managed by
`run_repository`.

### D9: Resume semantics — skip completed rounds, rerun the interrupted round
State is dumped after step 9 of each round, so a dumped state always represents
a completed round. On resume, the orchestrator loads the state, sets the round
counter to the dumped value, and starts the next round. A crash mid-round (no
dump for that round) causes resume to rerun the incomplete round from its
start. This is safe because each step produces idempotent artifacts (a fresh
`run_id` for the resumed round avoids collisions).

### D10: Error handling — per-example isolation, unrecoverable stops
Per-example errors (parse failure, timeout, HTTP error) are handled inside
`BaselineRunner` — the record is saved with a failed status and the run
continues. The orchestrator does not add a second retry layer. An
unrecoverable error (missing artifact, schema validation failure) aborts the
round and stops the cycle with `unrecoverable_error`, naming the failing step.
When `stop_on_first_error=true`, any step failure stops immediately.

### D11: Holdout evaluation — every round, logged only
Both versions are run on holdout each round and metrics are computed, but the
results are logged in the per-round report and never feed the decision. This
tracks holdout drift across versions. The holdout results are stored as
artifacts for offline analysis.

### D12: Active version management — via `prompt_repository`
On acceptance, the orchestrator calls `prompt_repository.activate(new_version)`
which marks the new version active and archives the previous. The active
version is tracked in `CycleState` (in-memory `PromptArtifact` + path). There
is no separate store index — the repository is the source of truth.

### D13: Thesis `in_prompt` flags — update on acceptance
On acceptance, `state.py` sets `in_prompt=true` for theses whose `cluster_id`
is in the accepted version's `source_candidates` and `false` for all others.
The updated thesis collection is persisted as part of the state dump.

### D14: Cycle log — append-only JSONL via `logging.getLogger`
`report.py` appends one JSON object per line to
`cycle_log_{run_id}_{ts}.jsonl` with `timestamp`, `round`, `event`, `details`.
The module uses `logging.getLogger(__name__)` for runtime logging and never
calls `logging.basicConfig` (that lives in the composition root). A new log
file is created per run so runs do not interleave.

### D15: Optional teacher refinement — between baseline run and thesis analysis
When `use_teacher_refinement=true`, `steps.py` calls `refine_theses` after
step 1 (baseline run) and before step 3 (thesis analysis). The refined theses
artifact path is passed to the analyzer instead of the raw theses. When
disabled, the analyzer uses `theses_raw_tiny` as before. The
`teacher_llm_client` port in `RunCycleDeps` is None when disabled; the
composition root omits it.

### D16: Composition root — expand `build_cli_deps`
`interfaces/cli/main.py:build_cli_deps` is expanded to assemble all ports:
`AsyncTask` (llm), `PromptStore` (prompt repo), `FileRunRepository` (run
repo), `Normalizer` functions, and `TeacherClient` (when enabled). It builds
`CycleConfig` from the `[cycle_orchestrator]` section and passes it through
`RunCycleInput`. The `NotImplementedError` catch at `main.py:177` is removed.
If the deps object grows complex, extract a `build_cycle_deps()` helper — but
keep it in the composition root.

## Risks / Trade-offs

- **[Sub-steps not port-migrated]** → The orchestrator calls sub-step modules
  that import infrastructure directly, so `run_cycle` transitively touches
  infrastructure. Mitigation: `run_cycle` itself imports only ports + sibling
  use-case modules (no direct infrastructure imports); the boundary is enforced
  at the orchestrator level. A future change can port-migrate the sub-steps.
- **[Long-running cycle]** → A 5-round cycle with dev + holdout runs can take
  tens of minutes. Mitigation: state is dumped after each round and the cycle
  is resumable; the CLI prints per-round progress.
- **[Mid-round crash loses the round]** → No mid-round checkpoint (D9).
  Mitigation: resume reruns the incomplete round from its start; the runner's
  own checkpointing limits lost work within a single run.
- **[Stale candidates after acceptance]** → The queue is refilled at the start
  of each round (D6) but consumed as-is within a round. Mitigation: the queue
  is short (≤ `top_n`); refilling mid-round would discard useful candidates.
- **[Holdout run cost]** → Running holdout every round (D11) doubles the run
  count. Mitigation: holdout is logged-only and can be disabled in a future
  change; the default favors information.
- **[AsyncTask / LLMClient signature mismatch]** → `AsyncTask` methods take a
  `session` first argument not in the `LLMClient` port. Mitigation: the
  composition root applies the existing `# type: ignore[arg-type]` workaround;
  a proper adapter is a future concern.
- **[cross_task_transfer expects `run_cycle.orchestrator`]** →
  `cross_task_transfer/transfer.py` references a non-existent
  `run_cycle.orchestrator` module. Mitigation: this change does not create
  `orchestrator.py`; the dangling imports in `transfer.py` are pre-existing and
  out of scope. File a follow-up if `cross_task_transfer` needs to call
  `run_cycle`.

## Open Questions

- **Refinement placement**: Should teacher refinement run once per cycle or
  every round? Default: every round when enabled (refined theses reflect the
  latest active version). Deferrable — a config toggle can change this later
  without changing the spec.
- **Holdout evaluation**: Every round or once at the end? Default: every round
  (D11). Deferrable — a `holdout_each_round` toggle can be added later.
- **Resume semantics**: Should an interrupted round rerun from scratch or from
  the last successful step? Default: rerun from scratch (D9). Deferrable —
  mid-round checkpointing is a future optimization.
- **Error classification**: Which errors are recoverable vs unrecoverable?
  Default: per-example errors are recoverable (handled by the runner); missing
  artifacts and schema failures are unrecoverable. Network errors after retry
  are recoverable. Deferrable — the classification lives in the runner, not
  the orchestrator.
