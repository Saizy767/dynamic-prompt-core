# Proposal

## Why

The optimization cycle (`stage3-cycle-orchestrator`) produces an optimized
prompt, thesis collection, and cluster structure for a single task. Starting a
new task from scratch (cold start) re-derives all of this from zero, costing
full cycle time even when the two tasks share the same output structure. There
is no way to reuse the base layers, theses, and cluster centroids of a completed
cycle as a starting point for a different but structurally compatible task. This
change introduces a cross-task transfer component that carries over the optimized
structure from a source task to a target task, re-runs the cycle on the target
dataset with a reworked rule set, and reports whether the transfer accelerates
convergence.

## What Changes

- Add a new component (`cross_task_transfer.py`) that accepts descriptions of two
  tasks — source (already optimized) and target (to optimize) — each carrying
  `task_id`, `dataset_path`, `num_classes`, `class_labels`, and `metric`
  (accuracy, macro-F1, or minority-class F1). Fail with an error naming the
  missing field when a task description is incomplete.
- Require source and target to have the same structure: same number of classes,
  same output contract (`decision`/`confidence` shape), same thesis-extraction
  schema. Reject the transfer with an error naming the mismatch when structures
  differ.
- Load source artifacts read-only: the active prompt version, version history,
  thesis collection with clusters, and cycle metrics. Never modify source
  artifacts; fail with an error naming the missing artifact when one is absent.
- Transfer base prompt layers (role, output contract, fallback) from source to
  target unchanged. Reformulate the "task" layer for the target domain via the
  LLM. Clear the "rules" layer entirely so the target cycle re-selects rules
  from the seeded thesis collection.
- Seed the target thesis collection with source theses, tagged
  `source=transfer`. Transferred theses do not count toward precision until the
  first target run records occurrences. Honor a configurable
  `max_transfer_theses` budget (default unlimited), selecting theses by
  descending source frequency.
- Seed target clusters with source centroids, tagged `source=transfer`, as
  initial clustering points. Update transferred centroids as new target theses
  are added. Skippable via `transfer_clusters=false`.
- Run the optimization cycle on the target task via
  `stage3-cycle-orchestrator`, starting from the transferred prompt version
  (base layers + empty rules) and the seeded thesis collection. All cycle
  components are used in their standard configuration. Target artifacts are
  stored under a separate `run_id` and path; source artifacts are never
  overwritten.
- Persist a transfer artifact named
  `transfer_{source_task_id}_to_{target_task_id}_{timestamp}.json` containing
  `source_version`, `target_final_version`, `transferred_theses_count`,
  `transferred_clusters_count`, `target_metrics_start`,
  `target_metrics_final`, `improvement` (absolute and relative), and
  `stop_reason`.
- Produce an effectiveness report with three metric snapshots: target baseline
  (empty prompt), after the first round (with transferred artifacts), and after
  the final round. The report records whether transfer accelerated convergence
  (fewer rounds to stop vs. cold start).
- Support an optional `compare_with_cold_start` mode (default false) that runs
  the target cycle twice — with transfer and without — and compares final
  metrics and rounds-to-stop.
- Log every transfer operation (load source, transfer layers, transfer theses,
  transfer centroids, run target cycle, final metrics) to an append-only log
  with timestamp and task id.
- Add a `[cross_task_transfer]` section to `config.toml` for `source_task_id`
  (required), `target_task_id` (required), `source_artifacts_path` (required),
  `target_artifacts_path` (required), `max_transfer_theses` (default unlimited),
  `transfer_clusters` (default true), `compare_with_cold_start` (default false),
  and `max_rounds_target` (default 5).

## Capabilities

### New Capabilities
- `stage4-cross-task-transfer`: Transfers an optimized prompt's base layers,
  thesis collection, and cluster centroids from a source task to a structurally
  compatible target task, reformulates the task layer, clears the rules, and
  re-runs the optimization cycle on the target dataset. Produces a transfer
  artifact, an effectiveness report, and an optional cold-start comparison,
  while keeping source and target artifacts fully isolated.

### Modified Capabilities
<!-- None — this change introduces a new standalone transfer component. It calls
     existing components (prompt store, cycle orchestrator, thesis analyzer,
     metrics) via their public APIs without altering their requirements. -->

## Impact

- **New code**: `cross_task_transfer.py` (component + CLI), mirroring the
  structure of `cycle_orchestrator.py` and `prompt_store.py` (dataclass config
  from TOML, artifact load/validate, argparse CLI). Reuses `PromptStore` from
  `prompt_store.py`, `run_cycle` / `load_state` from `cycle_orchestrator.py`,
  `ThesisBank` / `load_dump` / `write_dump` from `thesis_analyzer.py`,
  `compute_metrics` from `metrics.py`, and `PromptArtifact` / `PromptLayer` /
  `render` from `prompts.py`.
- **Config**: new `[cross_task_transfer]` section in `config.toml`.
- **Dependencies**: no new third-party dependencies. Reuses `aiohttp` (via the
  cycle orchestrator), `AsyncTask`, and the existing stage modules.
- **Artifacts**: new `transfer_*_to_*_*.json` (transfer artifact),
  `transfer_report_*_to_*_*.json` (effectiveness report), and
  `transfer_log_*_to_*_*.jsonl` (append-only operation log) written to
  `data/results/`.
- **Upstream**: consumes source artifacts from a completed
  `stage3-cycle-orchestrator` run (active prompt version, thesis bank dump,
  cycle metrics) and the target dataset artifact from `dataset-and-prompt`.
- **Downstream**: none — this is a top-level entry point that produces a
  self-contained transfer artifact and report.
