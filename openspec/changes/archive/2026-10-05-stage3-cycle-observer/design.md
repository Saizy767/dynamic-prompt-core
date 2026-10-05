# Design

## Context

The cycle orchestrator (`stage3-cycle-orchestrator`, `cycle_orchestrator.py`)
runs the optimization loop and writes a set of JSON artifacts per run, all in
`data/results/`:

- Per-round reports: `report_{run_id}_round{N}_{timestamp}.json` — round number,
  active version at start, new version, decision, dev/holdout metrics for both
  versions, rollback counter, changed decisions, next action. Loadable via
  `cycle_orchestrator.load_report`.
- Final summary: `summary_{run_id}_{timestamp}.json` — total rounds, stop reason,
  final active version, final dev/holdout metrics, `accepted_history`,
  `rollback_history` (list of `{round, version, reason}`),
  `all_changed_decisions`.
- State dumps: `state_{run_id}_{timestamp}.json` — round counter, active
  version, thesis dump path, clusters path, candidate queue, rollback counter.
- Cycle log: `cycle_log_{run_id}.jsonl` — append-only event log.

The version comparator (`stage2-version-comparator`, `version_comparator.py`)
writes decision artifacts `decision_{run_id}_v{new}_vs_v{active}_{ts}.json`
with `decision`, `new_version`, `active_version`, `metrics_new`,
`metrics_active`, `diff_accuracy`, `diff_macro_f1`, `diff_minority_f1`,
`rollback_count`, `reason`.

The prompt store (`stage3-prompt-version-store`, `prompt_store.py`) persists
version records with `version`, `text`, `hash`, `rules` (list of
`{cluster_id, text}` — accepted rules only), `source_candidates`,
`base_version`, `active`, `archived`. The composer's
`prompt_composer.write_prompt_version` writes the per-version artifact
`prompt_v{version}_{run_id}_{timestamp}.json` carrying the same fields plus
`metadata.counters` (`candidates_in`, `rules_formulated`, `rejected_distortion`,
`rejected_limits`, `final_rules_count`) and `metadata.config.distortion_threshold`.

The thesis analyzer (`stage1-thesis-analyzer`, `thesis_analyzer.py`) writes
thesis dumps `thesis_{run_id}_{prompt_version}_{timestamp}.json` with `metadata`,
`theses` (each with `frequency`, `in_prompt`, `cluster_id`, …), and `clusters`
(each with `cluster_id`, `count`, `frequency`, `precision`, …). Loadable via
`thesis_analyzer.load_dump`.

Metrics artifacts come from `stage1-metrics` (`metrics.py`). The observer does
not recompute metrics — it reads them from the reports and decision artifacts.

**Gap:** The composer persists aggregate rejection *counts*
(`rejected_distortion`, `rejected_limits`) and the `distortion_threshold`, but
not the individual rejected rules' text and cosine values. The observer's
distortion-rejection scenario needs those details. See D5 for the resolution.

## Goals / Non-Goals

**Goals:**
- Render a single, coherent review of a completed cycle from its artifacts.
- Cover all seven sections: per-round summaries, prompt evolution, rejected
  attempts, metric trends, changed decisions, thesis-store summary, final
  observation.
- Support markdown and json output from one intermediate report model.
- Stay strictly read-only: read cycle artifacts, write only the report + log.

**Non-Goals:**
- Accepting or rolling back versions (handled by the comparator/orchestrator).
- Mutating the active version, thesis store, prompt store, or any cycle
  artifact.
- Recomputing metrics (handled by `stage1-metrics`; observer reads artifacts).
- Visualization (graphs, dashboards).
- Sending reports to external systems (Slack, email).
- Automatically reacting to trends.
- Running concurrently with the cycle (the observer runs post-cycle).

## Decisions

### D1: Single-module component (`cycle_observer.py`)
Mirrors `cycle_orchestrator.py` / `version_comparator.py`: dataclass
`CycleObserverConfig.from_config`, `load_cycle_artifacts(run_id, config)`,
section builders (`build_round_summaries`, `build_prompt_evolution`,
`build_rejected_attempts`, `build_metric_trends`, `build_changed_decisions`,
`build_thesis_summary`, `build_final_observation`), a `render_*` pair
(markdown, json), `write_report`, `log_event`, argparse CLI.

**Alternative**: split each section into its own module. Rejected — the
sections share a single loaded-artifacts bundle and a single report model;
splitting adds import overhead without decoupling independent state.

### D2: Artifact discovery — glob by `run_id`, sort by round then timestamp
`load_cycle_artifacts(run_id, config)` globs `data/results/` for
`report_{run_id}_round*_*_*.json`, `decision_{run_id}_*_*_*.json`,
`prompt_v*_{run_id}_*.json`, `thesis_{run_id}_*_*.json`, and
`summary_{run_id}_*.json`, sorts reports by round number, decisions by
timestamp, and prompt versions by version number. The `run_id` is the only
required input; the observer reconstructs the full artifact set from it. A
missing expected artifact (e.g. a decision for a round that has a report) raises
`CycleObserverError` naming the file. Corrupted JSON raises naming the file.

**Alternative**: take an explicit list of artifact paths. Rejected — the
orchestrator names files by `run_id` + round/version/timestamp convention, so
globbing is deterministic and less error-prone than a hand-built list.

### D3: Read-only guarantee — load via existing `load_*` functions, write only report
The observer uses `cycle_orchestrator.load_report` / `load_state`,
`version_comparator.load_decision`, `prompt_store.PromptStore.load` /
`prompt_composer.load_prompt_version`, `thesis_analyzer.load_dump`, and
`json.load` for metrics artifacts. It never calls a `write_*` / `activate` /
`save` / `dump_state` on any upstream component. The only writes are
`write_report` (the observer's own report) and `log_event` (the observer's own
log). This makes the read-only guarantee structural, not conventional.

**Alternative**: open artifacts with raw `json.load`. Rejected — the existing
loaders validate schema and give typed errors; raw loads would duplicate
validation and lose the named-error requirement.

### D4: Prompt evolution — set diff of accepted rules between consecutive active versions
For each pair of consecutive active versions (v_i, v_{i+1}), the observer loads
both version records from the prompt store, reads their `rules` lists (each
entry is `{cluster_id, text}`), and computes:
- **added**: rules in v_{i+1} whose text is absent in v_i.
- **removed**: rules in v_i whose text is absent in v_{i+1}.
- **preserved**: rules whose text is in both.

Each rule is annotated with its `cluster_id` (source cluster) and the version in
which it first appeared (the earliest version in the lineage whose rules contain
that text). Comparison is by rule text (normalized whitespace), not by
`cluster_id`, because a cluster may produce a reworded rule across versions.

**Alternative**: compare by `cluster_id`. Rejected — the same cluster may
produce different rule text across versions (reformulation); text comparison
tracks actual prompt content.

### D5: Rejected attempts — from rollback history + decision artifacts + version metadata
The observer reads `rollback_history` from the final summary (list of
`{round, version, reason}`) and cross-references each rolled-back version's
decision artifact (for `metrics_new`, `metrics_active`, diffs) and prompt-version
artifact (for the formulated `rules` and `source_candidates` linking to clusters).

For distortion rejections specifically, the observer needs the rejected rule
text, the cosine value, and the threshold. The threshold is already in
`metadata.config.distortion_threshold`. The individual rejected rule text +
cosine are **not** persisted today. This change adds a `rejected_rules` list to
the prompt-version artifact metadata via a small, additive extension to
`prompt_composer.write_prompt_version`: each entry is
`{cluster_id, text, cosine, threshold, reason}` where `reason` is one of
`distortion`, `limits`, `empty_formulation`. This is a backward-compatible
metadata addition (a new optional field in an existing artifact), not a
spec-level behavior change of the composer. The observer reads `rejected_rules`;
when absent (artifacts from older runs), it falls back to the aggregate
`metadata.counters.rejected_distortion` count + threshold and notes that
individual details were not persisted.

**Alternative**: have the observer recompute the distortion check. Rejected —
the observer is read-only and must not recompute; recomputation would also need
the embedding model and centroids, duplicating the composer's work.

### D6: Metric trends — one row per round from per-round reports
The observer builds a table with one row per round, columns: round, accuracy,
macro-F1, minority-class F1, parse-failure count. Values come from the
per-round report's `metrics_dev_new` (the new version's dev metrics for that
round). When a metric is absent, the cell is `N/A` and the skip is logged.
Holdout columns are included only when `include_holdout=true` (from the report's
`metrics_holdout_new`). Trends are displayed only — the observer never labels a
trend "improving" or "degrading"; that interpretation is left to the reader.

**Alternative**: aggregate trends across the whole cycle (deltas only). Rejected
— the per-round table is the requested view; cycle-level deltas are in the final
observation (D9).

### D7: Changed decisions — from per-round reports, text truncated to config
Each per-round report carries `changed_decisions` (list of
`{id, direction, ...}`). The observer collects these per round and also reads
`all_changed_decisions` from the final summary for the cycle-wide view. Each
entry's example text is truncated to `max_example_text_length` (default 100)
characters with an ellipsis. When the list is empty, the observer emits an
explicit "no changes" message. The direction (`correct→incorrect`,
`incorrect→correct`) is read from the artifact; the observer does not recompute
it.

**Alternative**: truncate by words. Rejected as the default — character
truncation is predictable for table layout; word truncation is an open question
(OQ1).

### D8: Thesis-store summary — from the latest thesis dump
The observer loads the thesis dump with the latest timestamp for the `run_id`
(via `thesis_analyzer.load_dump`), reads `theses` and `clusters`, and computes:
total theses, cluster count, top-N clusters by `frequency` (default 10,
configurable via `top_clusters_count`) with `precision` and representative
theses (the first few `text_raw` values in the cluster), count of clusters with
any thesis where `in_prompt=true`, and count of theses with `cluster_id` unset
(unassigned).

**Alternative**: summarize every thesis dump per round. Rejected — the
end-of-cycle snapshot is the requested summary; per-round thesis evolution is
visible through the prompt-evolution section.

### D9: Final observation — start vs final, absolute and relative deltas
The observer reads the final summary for `final_active_version`,
`final_dev_metrics`, `final_holdout_metrics`, `stop_reason`, `total_rounds`,
`accepted_history`, `rollback_history`. The start version is the active version
at round 1's start (from the first per-round report's
`active_version_at_start`); the start metrics are that version's dev metrics
(from the first report's `metrics_dev_active`). For each metric, the observer
computes `absolute = final - start` and `relative = (final - start) / start`
(guarded against divide-by-zero). Counts: accepted = `len(accepted_history)`,
rolled back = `len(rollback_history)`. When no metric improved, the observation
explicitly states "no improvement".

**Alternative**: read start metrics from the state dump. Rejected — the first
per-round report is the canonical record of round-1 start conditions and is
always present for a completed cycle.

### D10: Output formats — shared intermediate model, two renderers
The section builders produce one intermediate `ObservationReport` dataclass
(round summaries, evolution, rejected attempts, metric trends, changed
decisions, thesis summary, final observation). Two renderers consume it:
`render_markdown(report) -> str` and `render_json(report) -> str`. Markdown uses
headers per section, a pipe-table for metric trends, and bullet lists elsewhere;
it is structured for pasting into a README or issue. JSON is
`json.dumps(dataclasses.asdict(report), indent=2)`. `output_format` selects the
renderer; `output_path` selects the destination. Both renderers are pure
functions of the model, so adding a third format later touches nothing else.

**Alternative**: build markdown directly without an intermediate model.
Rejected — the json format would then duplicate the section logic, and the model
also makes the report testable independently of rendering.

### D11: Config — `[cycle_observer]` section
`CycleObserverConfig.from_config` reads `[cycle_observer]` from `config.toml`:
`output_format` (default `markdown`), `output_path` (default
`observer_report.md`), `max_example_text_length` (default 100),
`top_clusters_count` (default 10), `include_holdout` (default true), plus
`output_dir` (default `data/results`) for artifact discovery and the log. Falls
back to defaults for missing keys. Mirrors `CycleOrchestratorConfig`.

### D12: Logging — append-only JSONL
`log_event(event, details, run_id, config)` appends one JSON object
(`{timestamp, event, details}`) to `observer_log_{run_id}.jsonl` in `output_dir`.
Events: `artifacts_loaded`, `section_built` (with section name), `report_written`,
`error`. Append-only via `open(..., "a")`. A new log per `run_id` so runs do not
interleave. Mirrors the orchestrator's `cycle_log` convention.

## Risks / Trade-offs

- **[Distortion details need a composer extension]** → The observer's
  distortion-rejection scenario needs individual rejected rule text + cosine,
  which the composer does not persist today (D5). Mitigation: this change adds a
  backward-compatible `rejected_rules` metadata field to the prompt-version
  artifact; older artifacts fall back to aggregate counts + threshold.
- **[Artifact discovery depends on filename conventions]** → Globbing by
  `run_id` assumes the orchestrator's naming convention (D2). Mitigation: the
  convention is fixed by the orchestrator spec and tested; a missing artifact is
  a named error, not a silent skip.
- **[Read-only guarantee is convention-structural]** → Nothing forces the
  observer to stay read-only except its own code. Mitigation: the observer uses
  only `load_*` functions from upstream components (D3); a code review checks
  that no `write_*` / `activate` / `save` is called.
- **[Large cycles produce large reports]** → A 5-round cycle with many changed
  decisions and a big thesis store can make the markdown report long.
  Mitigation: changed-decision text is truncated (D7); the thesis summary is
  top-N (D8); the json format is available for machine consumption.
- **[Relative improvement undefined at zero baseline]** → `relative = (final -
  start) / start` divides by zero when a start metric is 0. Mitigation: the
  observer renders `relative` as `N/A` when the start value is 0 and logs it.

## Open Questions

- **OQ1: Truncate example text by characters or by words?** Default is characters
  (100). Word truncation avoids cutting mid-word but is less predictable for
  table layout. Deferred — `max_example_text_length` is configurable; switch to
  word-based only if character truncation reads poorly in practice.
- **OQ2: Include holdout always, or only when holdout was run?** Default is
  `include_holdout=true`. When the cycle did not run holdout, the columns are
  `N/A`. Deferred — the config flag already lets users exclude holdout; auto-
  detection can be added if the `N/A` columns are noisy.
- **OQ3: Report filename fixed or `run_id`-stamped?** Default `output_path` is
  fixed (`observer_report.md`). Stamping with `run_id` avoids overwriting across
  runs. Deferred — `output_path` is configurable; a future change can default it
  to `observer_report_{run_id}.md` if overwrites become a problem.
- **OQ4: Incremental mode — report after each round or only post-cycle?** This
  design is post-cycle only. Incremental would require the observer to run
  mid-cycle and re-load artifacts each round. Deferred — post-cycle is the
  requested behavior; incremental can be added behind a `--watch` flag later.
