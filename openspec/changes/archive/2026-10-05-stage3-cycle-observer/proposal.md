# Proposal

## Why

The cycle orchestrator produces a rich set of artifacts per run — per-round
reports, decision artifacts, prompt-version records, metrics, thesis-bank dumps,
and changed-decisions lists — but there is no way for a developer to see, in one
human-readable view, how the prompt evolved across rounds, which rules were
added and rolled back, where the model errored persistently, and whether the
cycle actually improved anything. Today, understanding a run means opening a
handful of JSON files and mentally diffing them. This change introduces a
read-only observer that loads the cycle artifacts and renders a coherent
review: per-round summaries, prompt-rule evolution, rejected attempts, metric
trends, changed predictions, a thesis-bank summary, and a final observation with
net improvement and stop reason. The observer takes no decisions and mutates no
state — it only aggregates and explains.

## What Changes

- Add a new component (`cycle_observer.py`) that loads the artifacts produced by
  a completed cycle: per-round reports (`report_*_round*_*_*.json`), decision
  artifacts (`decision_*_v*_vs_v*_*_*.json`), prompt-version records from
  `stage3-prompt-version-store` (`prompt_store.py`), metrics artifacts from
  `stage1-metrics`, thesis dumps (`thesis_*_*_*.json`), and
  changed-decisions lists. Fail with an error naming the missing or corrupted
  artifact when an expected file is absent or unparseable.
- Produce a per-round summary for each round containing: round number, active
  version at round start, new composed version, decision (accept or rollback),
  key dev metrics before and after, minority-class F1 change, count of changed
  predictions, and rollback counter value. Skip rounds absent from the artifacts
  and log the skip.
- Produce a prompt-evolution overview: rules added, removed, and preserved
  between consecutive versions, with each rule annotated by its source cluster
  and the version in which it first appeared.
- Produce a rejected-attempts overview: rolled-back versions with the formulated
  rules, the rejection reason (meaning distortion, limit violation, metric
  degradation), and a link to the originating cluster. Distortion rejections
  include the cosine value and threshold.
- Produce a metric-trends table: one row per round with accuracy, macro-F1,
  minority-class F1, and parse-failure count. Missing metrics are rendered as
  `N/A` and logged. Trends are displayed only, never interpreted.
- Produce a changed-decisions overview: list of example ids whose
  `predicted_decision` changed between the active and new version, with the
  direction (correct→incorrect, incorrect→correct) and a truncated example text.
  Explicitly report when there are no changes.
- Produce a thesis-store summary at cycle end: total theses, cluster count,
  top-N clusters by frequency with precision and representative theses, count of
  clusters with `in_prompt=true`, and count of unassigned theses.
- Produce a final observation: start version and metrics, final version and
  metrics, absolute and relative improvement per metric, stop reason, total
  rounds, and counts of accepted and rolled-back versions. Explicitly report
  when there is no improvement.
- Support two output formats — `markdown` (human-readable, README/issue-ready)
  and `json` (machine-readable) — selected via config.
- Remain strictly read-only: the observer reads cycle artifacts and writes only
  its own report. It never mutates the active version, thesis bank, prompt
  store, or any cycle artifact.
- Log all observer operations (artifact load, section formation, report write)
  to an append-only log.
- Add a `[cycle_observer]` section to `config.toml` for `output_format`
  (default `markdown`), `output_path` (default `observer_report.md`),
  `max_example_text_length` (default 100), `top_clusters_count` (default 10),
  and `include_holdout` (default true).

## Capabilities

### New Capabilities
- `stage3-cycle-observer`: A read-only component that loads the artifacts
  produced by a completed optimization cycle and renders a human- and
  machine-readable review of what changed between rounds: per-round summaries,
  prompt-rule evolution (added/removed/preserved), rejected attempts with
  reasons, metric-trends table, changed-prediction list, thesis-bank summary,
  and a final observation with net improvement and stop reason. Supports
  markdown and json output. Mutates no cycle state.

### Modified Capabilities
<!-- None — this change introduces a new read-only component. It consumes
     existing artifacts via their public load functions and writes only its own
     report; no per-component requirement changes. -->

## Impact

- **New code**: `cycle_observer.py` (component + CLI), mirroring the structure
  of `cycle_orchestrator.py` and `version_comparator.py` (dataclass config from
  TOML, artifact load/validate, argparse CLI). Reuses `cycle_orchestrator`
  `load_report` / `load_state`, `version_comparator` decision-artifact loading,
  `prompt_store.PromptStore` for version records, `thesis_analyzer.load_dump`
  for thesis dumps, and `metrics` artifact loading.
- **Config**: new `[cycle_observer]` section in `config.toml`.
- **Dependencies**: no new third-party dependencies. Uses only the standard
  library (`json`, `os`, `dataclasses`, `tomllib`, `datetime`) plus the existing
  stage modules for artifact loading.
- **Artifacts**: new `observer_report.md` (or `.json`, path configurable) and
  `observer_log_*.jsonl` (append-only operation log) written to `data/results/`.
- **Upstream**: consumes per-round reports, the final summary, state dumps, and
  the cycle log from `stage3-cycle-orchestrator`; decision artifacts and
  changed-decisions lists from `stage2-version-comparator`; version records from
  `stage3-prompt-version-store`; metrics artifacts from `stage1-metrics`;
  thesis dumps from `stage1-thesis-analyzer`; and `[cycle_observer]`
  config.
- **Downstream**: none — this is a terminal reporting component. Its output is
  consumed by humans or by downstream tooling that reads the json format.
