# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[cycle_observer]` section to `config.toml` with keys
  `output_format`, `output_path`, `max_example_text_length`,
  `top_clusters_count`, `include_holdout`, and `output_dir`, with inline
  comments documenting each default. Verify the file parses with `tomllib.load`
  and the section round-trips.
- [x] 1.2 Create `cycle_observer.py` with module docstring,
  `from __future__ import annotations`, a `CycleObserverConfig` dataclass with
  `from_config()` reading `[cycle_observer]` from `config.toml` (mirroring
  `CycleOrchestratorConfig`), and default constants (`DEFAULT_OUTPUT_FORMAT="markdown"`,
  `DEFAULT_OUTPUT_PATH="observer_report.md"`,
  `DEFAULT_MAX_EXAMPLE_TEXT_LENGTH=100`,
  `DEFAULT_TOP_CLUSTERS_COUNT=10`,
  `DEFAULT_INCLUDE_HOLDOUT=True`,
  `DEFAULT_OUTPUT_DIR="data/results"`). Verify `CycleObserverConfig.from_config()`
  returns the configured values and falls back to defaults for missing keys.
- [x] 1.3 Add a `CycleObserverError(ValueError)` exception class and an
  `ObservationReport` dataclass holding `round_summaries`, `prompt_evolution`,
  `rejected_attempts`, `metric_trends`, `changed_decisions`, `thesis_summary`,
  and `final_observation` (each a typed structure: dataclass or list of
  dataclasses). Verify `ObservationReport` constructs with defaults and
  serializes to a plain dict via `dataclasses.asdict`.

## 2. Composer artifact extension — persist rejected-rule details

- [x] 2.1 Extend `prompt_composer.compose` to collect a `rejected_rules` list
  alongside `rules_with_lineage`: for each rejected candidate, append
  `{cluster_id, text, cosine, threshold, reason}` where `reason` is one of
  `distortion`, `limits`, `empty_formulation` (design D5). Return it as a fourth
  element of the compose result tuple. Verify with a mocked formulation that a
  candidate failing the distortion check produces a `rejected_rules` entry with
  the cosine value and threshold.
- [x] 2.2 Extend `prompt_composer.write_prompt_version` to accept the
  `rejected_rules` list and persist it under `metadata.rejected_rules`. Update
  `prompt_composer.load_prompt_version` to surface the field (defaulting to `[]`
  when absent for backward compatibility). Verify a round-trip: a written
  artifact reloads with `rejected_rules` intact, and an older artifact without
  the field loads with `rejected_rules == []`.
- [x] 2.3 Update `cycle_orchestrator.compose_new_version` to thread the new
  `rejected_rules` from `prompt_composer.compose` into
  `prompt_composer.write_prompt_version`. Verify the orchestrator's existing
  round-end test still passes and the written prompt-version artifact now
  contains `metadata.rejected_rules`.

## 3. Artifact loading

- [x] 3.1 Implement `discover_artifacts(run_id, config)` that globs
  `output_dir` for `report_{run_id}_round*_*.json`,
  `decision_{run_id}_*_*_*.json`, `prompt_v*_{run_id}_*.json`,
  `thesis_{run_id}_*_*.json`, and `summary_{run_id}_*.json`, and returns the
  sorted path lists (reports by round, decisions by timestamp, prompt versions
  by version number, thesis dumps by timestamp) (design D2). Verify with a
  fixture `output_dir` that the glob returns the expected files in order.
- [x] 3.2 Implement `load_cycle_artifacts(run_id, config)` that calls
  `discover_artifacts`, then loads each artifact via the existing `load_*`
  functions (`cycle_orchestrator.load_report`, `version_comparator.load_decision`,
  `prompt_composer.load_prompt_version` / `prompt_store.PromptStore.load`,
  `thesis_analyzer.load_dump`, `json.load` for the summary) (design D3). Raise
  `CycleObserverError` naming the missing file when an expected artifact is
  absent, and naming the corrupted file when JSON fails to parse. Verify: all
  artifacts present loads without error; a missing decision raises naming the
  file; a corrupted report raises naming the file.
- [x] 3.3 Verify `load_cycle_artifacts` returns a typed bundle (dataclass) with
  `reports` (sorted list), `decisions` (sorted list), `prompt_versions` (sorted
  list), `thesis_dumps` (sorted list), and `summary` (dict). Verify the bundle
  fields round-trip against the on-disk artifacts.

## 4. Section builders — round summaries, evolution, rejected attempts

- [x] 4.1 Implement `build_round_summaries(bundle, config)` that produces one
  `RoundSummary` per report with: round number, active version at start, new
  version, decision, dev metrics before (`metrics_dev_active`) and after
  (`metrics_dev_new`), minority-class F1 change
  (`metrics_dev_new.minority_f1 - metrics_dev_active.minority_f1`), count of
  changed predictions (`len(changed_decisions)`), and rollback counter. Skip
  rounds absent from the reports and log the skip. Verify a 3-report bundle
  yields 3 summaries with all fields, and a bundle missing round 2 yields 2
  summaries and a logged skip.
- [x] 4.2 Implement `build_prompt_evolution(bundle, config)` that, for each pair
  of consecutive active versions, computes added/removed/preserved rules by text
  (design D4), annotates each rule with its `cluster_id` and the version in
  which it first appeared (walk the version lineage). Verify: a rule in v2 not
  in v1 is "added in v2"; a rule in v1 not in v2 is "removed in v2"; a rule in
  both is "preserved"; the first-appearance version is correct across a 3-version
  chain.
- [x] 4.3 Implement `build_rejected_attempts(bundle, config)` that reads
  `rollback_history` from the summary, cross-references each rolled-back
  version's decision artifact (metrics, diffs, reason) and prompt-version
  artifact (formulated `rules`, `source_candidates`, `metadata.rejected_rules`)
  (design D5). For distortion rejections, include the rule text, cosine, and
  threshold from `rejected_rules`; when `rejected_rules` is absent, fall back to
  the aggregate `metadata.counters.rejected_distortion` count + threshold and
  note that individual details were not persisted. Verify: a rolled-back version
  with `rejected_rules` shows the rule text, cosine, and threshold; a rolled-back
  version without `rejected_rules` shows the aggregate count + threshold with a
  fallback note.

## 5. Section builders — metric trends, changed decisions, thesis summary, final observation

- [x] 5.1 Implement `build_metric_trends(bundle, config)` that produces one row
  per round with accuracy, macro-F1, minority-class F1, and parse-failure count
  from `metrics_dev_new` (design D6). Mark a cell `N/A` and log when a metric is
  absent. Include holdout columns from `metrics_holdout_new` only when
  `include_holdout=true`. Verify a 3-round bundle yields 3 rows; a missing
  metric yields `N/A`; `include_holdout=false` omits holdout columns.
- [x] 5.2 Implement `build_changed_decisions(bundle, config)` that collects
  `changed_decisions` per round and the cycle-wide `all_changed_decisions` from
  the summary, truncating each example text to `max_example_text_length` with an
  ellipsis (design D7). Emit an explicit "no changes" message when the list is
  empty. Verify: 12 changed predictions yield 12 entries with truncated text;
  an empty list yields the "no changes" message; text longer than 100 chars is
  truncated to 100 + ellipsis.
- [x] 5.3 Implement `build_thesis_summary(bundle, config)` that loads the latest
  thesis dump and computes: total theses, cluster count, top-N clusters by
  `frequency` (N = `top_clusters_count`) with `precision` and representative
  theses (first few `text_raw` in the cluster), count of clusters with any
  thesis where `in_prompt=true`, and count of unassigned theses (design D8).
  Verify with a 20-cluster dump and `top_clusters_count=10`: the summary has 10
  top clusters sorted by frequency, the correct `in_prompt` cluster count, and
  the correct unassigned count.
- [x] 5.4 Implement `build_final_observation(bundle, config)` that reads the
  summary for final version/metrics/stop reason/total rounds/accepted and
  rollback histories, reads the first report for start version and start
  metrics, and computes absolute and relative improvement per metric (design D9).
  Render `relative` as `N/A` when the start value is 0 and log it. Emit an
  explicit "no improvement" message when no metric improved. Verify: a cycle
  with improvement shows positive deltas; a cycle with no improvement shows the
  "no improvement" message; a zero start metric yields `N/A` relative.

## 6. Rendering, report writing, and logging

- [x] 6.1 Implement `render_markdown(report)` that renders the
  `ObservationReport` to a markdown string with a header per section, a
  pipe-table for metric trends, and bullet lists elsewhere (design D10). Verify
  the output contains all seven section headers, the metric-trends table has one
  row per round, and the string is valid markdown (no unbalanced pipes in the
  table).
- [x] 6.2 Implement `render_json(report)` that serializes the report via
  `json.dumps(dataclasses.asdict(report), indent=2)`. Verify the output is valid
  JSON and round-trips: `json.loads` reproduces the report dict with all seven
  sections.
- [x] 6.3 Implement `write_report(report, config)` that selects the renderer by
  `output_format` and writes to `output_path` (design D10). Verify:
  `output_format=markdown` writes a `.md` file whose content equals
  `render_markdown`; `output_format=json` writes a `.json` file whose content
  equals `render_json`; `output_path=/tmp/report.md` writes to that path.
- [x] 6.4 Implement `log_event(event, details, run_id, config)` that appends one
  JSON object (`{timestamp, event, details}`) to
  `observer_log_{run_id}.jsonl` in `output_dir` (design D12). Verify the file is
  append-only (two calls produce two lines), each line is valid JSON, and the
  timestamp is present.

## 7. CLI and end-to-end integration

- [x] 7.1 Implement `run_observer(run_id, config)` that loads artifacts, builds
  all seven sections, assembles the `ObservationReport`, writes the report, and
  logs `artifacts_loaded`, `section_built` (per section), and `report_written`.
  Verify with a fixture cycle output that the report file is written and the log
  contains one `artifacts_loaded`, seven `section_built`, and one
  `report_written` event.
- [x] 7.2 Implement the argparse CLI (`python cycle_observer.py <run_id>
  [--config config.toml] [--format markdown|json] [--output <path>]`) that loads
  config, overrides `output_format` / `output_path` from flags when given, and
  runs `run_observer`. Verify the CLI `--help` lists all flags and that
  `--format json --output /tmp/r.json` writes a json report to `/tmp/r.json`.
- [x] 7.3 Run the observer end-to-end against the artifacts of a real
  `max_rounds=3` cycle run. Verify: the markdown report contains all seven
  sections, the metric-trends table has 3 rows, the final observation lists the
  stop reason and accepted/rollback counts, the changed-decisions section lists
  the ids with direction, the thesis summary lists the top-10 clusters, and no
  cycle artifact (reports, decisions, prompt versions, thesis dumps, summary,
  state) was modified (compare mtimes before and after).
- [x] 7.4 Run the observer with `output_format=json` against the same cycle run.
  Verify the json report is valid JSON, contains all seven sections, and
  round-trips through `json.loads`. Then run with `include_holdout=false` and
  verify the metric-trends section omits holdout columns.
