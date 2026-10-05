# Design

## Context

The project has a linear optimization pipeline whose per-step components are all
implemented and tested:

`dataset-and-prompt` → `stage1-baseline-runner` (`runner.py`) →
`stage1-metrics` (`metrics.py`) → `stage1-thesis-analyzer`
(`thesis_analyzer.py`) → `stage2-rule-candidate-selector`
(`rule_candidate_selector.py`) → `stage3-prompt-composer`
(`prompt_composer.py`) → `stage2-version-comparator`
(`version_comparator.py`).

Each component is a standalone Python module with a dataclass config loaded from
a TOML section, artifact load/validate helpers, and an argparse CLI. The
orchestrator follows the same pattern but, unlike the per-step components, it
calls the other components' public functions rather than doing a single
transform.

Key existing interfaces the orchestrator reuses:

- `BaselineRunner(task, config, split, run_id, dataset_artifact,
  classify_prompt)` — runs a split through the model. `classify_prompt` is an
  optional `PromptArtifact` override (added by the comparator change); when
  `None`, the runner uses `CLASSIFICATION_PROMPT_V0`.
- `metrics.compute_metrics(results_path, config)` — computes metrics from a
  results artifact. `metrics.compare_versions(path_a, path_b, config)` returns
  deltas and a `changed` list with `{id, direction}`.
- `thesis_analyzer` — produces a thesis-bank dump with clusters and centroids.
- `rule_candidate_selector` — emits a `rule_candidates_*.json` artifact.
- `prompt_composer.compose(...)` / `prompt_composer.write_prompt_version(...)`
  / `prompt_composer.load_prompt_version(...)` — turns candidates into a new
  prompt-version artifact.
- `version_comparator.decide(...)`, `.update_rollback_count(...)`,
  `.check_stop(...)`, `.next_candidate(...)`, `.changed_decisions(...)` — the
  decision primitives. The comparator is stateless across calls; the orchestrator
  threads the rollback counter and candidate queue.
- `prompts.PromptArtifact` / `prompts.CLASSIFICATION_PROMPT_V0` — the prompt
  store. Today there is only v0; the orchestrator treats the in-memory active
  version as the store and persists composed versions via the composer's
  `write_prompt_version`.

## Goals / Non-Goals

**Goals:**
- Drive the full optimization loop for a configurable number of rounds without
  manual intervention.
- Thread the round counter, candidate queue, and rollback counter across rounds.
- Produce per-round reports and a final summary with full lineage.
- Dump reloadable state after each round and support resuming an interrupted
  cycle.

**Non-Goals:**
- Composition of individual prompt versions (handled by the composer).
- Metrics computation internals (handled by `stage1-metrics`).
- Thesis analysis and clustering (handled by `stage1-thesis-analyzer`).
- Rule candidate selection (handled by `stage2-rule-candidate-selector`).
- Version comparison logic (handled by `stage2-version-comparator`).
- UI or visualization.
- Concurrent execution of multiple cycles.
- Multi-model or multi-task execution.

## Decisions

### D1: Single-module component (`cycle_orchestrator.py`)
Mirrors `version_comparator.py` / `prompt_composer.py`: dataclass
`CycleOrchestratorConfig.from_config`, `load_initial_state(config)`,
`run_round(state, config, ...)`, `run_cycle(config)` orchestration,
`write_report(...)`, `write_summary(...)`, `dump_state(...)` /
`load_state(...)`, `log_event(...)`, argparse CLI.

**Alternative**: split the round execution from the cycle loop into two modules.
Rejected — the round and the loop are tightly coupled (the loop threads the
state the round mutates), and a single module keeps the state threading
explicit.

### D2: Round execution — fixed nine-step sequence calling existing components
`run_round(state, config, async_task)` executes, in order:

1. `BaselineRunner` with the active prompt on dev → active results artifact.
2. `metrics.compute_metrics` on the active results.
3. `thesis_analyzer` on the active results → updated thesis bank + clusters.
4. `rule_candidate_selector` on the thesis bank → candidate artifact; the
   candidate queue is populated from its candidates.
5. `prompt_composer.compose` on the first candidate + active base layers → new
   prompt-version artifact.
6. `BaselineRunner` with the new prompt on dev → new results artifact.
7. `version_comparator.decide` on the new vs. active dev metrics →
   accept/rollback.
8. On accept: mark the new version active, archive the old, update `in_prompt`
   flags. On rollback: pop the next candidate from the queue and re-compose +
   re-run + re-decide (steps 5–7) until a candidate is accepted or the queue is
   exhausted.
9. `write_report(...)` with all round fields.

Each step that produces an artifact returns its path so the report and state
dump carry full lineage.

**Alternative**: model each step as a pluggable strategy object. Rejected — the
sequence is fixed by the spec and there is exactly one implementation per step;
a plugin layer adds indirection without benefit.

### D3: Round counter — increment on completed rounds only
The counter increments after step 9 (report written). A round aborted mid-way by
an unrecoverable error does not increment the counter; the cycle stops with
reason `unrecoverable_error` and the partial round is not counted. This makes
"total rounds executed" in the final summary equal to the counter value and
keeps resume semantics simple (resume starts at `counter + 1`).

**Alternative**: increment on round start. Rejected — an aborted round would
then count as executed, and resume would skip it, losing the work-in-progress
without a record of the failure.

### D4: Candidate queue — refilled at the start of each round
The selector runs in step 4 of every round and refills the queue from the
current thesis bank. This guarantees the candidates reflect the latest clusters
(which may have changed because the active version changed). On rollback within
a round, the queue is consumed in order without re-running the selector; the
queue is refilled only when the next round starts.

**Alternative**: refill only when exhausted. Rejected — stale candidates from a
previous thesis bank would be tried against a new active version, wasting
formulation calls on candidates that may no longer be relevant.

### D5: Rollback counter — part of the dumped state
The rollback counter is stored in the state dump and restored on resume. It is
not recomputed from the decision history because the history records decisions
but not the "consecutive" run length at each point; recomputing would require
replaying the entire history. The counter is reset to zero on acceptance
(via `version_comparator.update_rollback_count`).

**Alternative**: recompute from history on resume. Rejected — more complex and
fragile for no benefit; the counter is a single integer.

### D6: State dump — single JSON file per round
`state_{run_id}_{timestamp}.json` containing `round_counter`, `active_version`,
`active_version_path`, `thesis_bank_path`, `clusters_path`,
`candidate_queue` (list of artifact paths), `rollback_counter`,
`latest_report_path`, `accepted_history` (list of version ids),
`rollback_history` (list of `{round, version, reason}`), and `run_id`. One file
per round is simpler to reason about than per-component files and makes resume a
single load.

**Alternative**: separate files per component. Rejected — the orchestrator needs
all components together to resume; splitting them adds bookkeeping without
benefit.

### D7: Report and summary format — JSON
Per-round reports are `report_{run_id}_round{N}_{timestamp}.json`; the final
summary is `summary_{run_id}_{timestamp}.json`. JSON is machine-readable and
matches the artifact convention of every other component. `report_format` is
configurable (default `json`); a future change can add markdown without changing
the JSON path.

**Alternative**: markdown reports. Rejected as the default — downstream tooling
consumes JSON; a markdown renderer can be added later behind `report_format`.

### D8: Resume semantics — skip completed rounds, rerun the interrupted round
The state is dumped after step 9 of each round, so a dumped state always
represents a completed round. On resume, the orchestrator loads the state,
sets the round counter to the dumped value, and starts the next round. If a
crash occurs mid-round (no dump was written for that round), resume reruns the
incomplete round from its start — there is no partial-round checkpoint. This is
safe because every step is idempotent at the artifact level (the runner
checkpoints within a run, but a fresh run_id for the resumed round avoids
collisions).

**Alternative**: checkpoint mid-round. Rejected — the round is short relative to
the full cycle, and mid-round checkpointing would require each step to be
resumable, adding complexity for a marginal speedup.

### D9: Error handling — unrecoverable stops, recoverable is per-step
An unrecoverable error (missing artifact, schema validation failure, limit
violation) aborts the round and stops the cycle with reason
`unrecoverable_error`, naming the failing step. Recoverable errors (network
timeouts, transient LLM failures) are retried by the runner and `AsyncTask`
themselves; the orchestrator does not add a second retry layer. When
`stop_on_first_error=true`, any step failure — recoverable or not — stops the
cycle immediately.

**Alternative**: orchestrator-level retry with backoff for network errors.
Rejected — the runner already retries via `AsyncTask`; a second layer masks the
root cause and complicates the log.

### D10: Holdout evaluation — every round, logged only
Both versions are run on holdout each round and metrics are computed, but the
results are logged in the per-round report and never feed the decision (matching
the comparator spec). Every-round evaluation is more informative (it tracks
holdout drift across versions) and the holdout run is cheap relative to the dev
run + composition. The holdout results are stored as artifacts for offline
analysis.

**Alternative**: holdout only at the end. Rejected — the final summary would
have holdout for only the final version, losing the drift history that is the
main reason to log holdout separately.

### D11: Active version management — in-memory + prompt-version artifacts
The "prompt store" is the set of `prompt_v*_*_*.json` artifacts written by the
composer plus the built-in `CLASSIFICATION_PROMPT_V0`. The orchestrator keeps
the active version as an in-memory `PromptArtifact` (built from the composer's
artifact via `version_comparator.build_prompt_artifact` for composed versions,
or `CLASSIFICATION_PROMPT_V0` for the baseline). On acceptance, the new version
becomes active; the previous active version's artifact remains on disk
(archived = no longer active). There is no separate store index file — the
active version is tracked in the state dump and the cycle log.

**Alternative**: a `prompt_store.json` index. Rejected — the state dump already
records the active version, and a separate index would need to be kept in sync
with the state.

### D12: Thesis bank `in_prompt` flags — update on acceptance
On acceptance, the orchestrator sets `in_prompt=true` for theses whose
`cluster_id` is in the accepted version's `source_candidates`, and `false` for
all others. This is a direct mutation of the in-memory thesis bank, which is
then persisted as part of the next thesis-analyzer dump (or immediately if the
analyzer does not dump until the next round). The flag tells the selector to
skip clusters already represented in the active prompt.

**Alternative**: recompute `in_prompt` from the active prompt's rules on every
selector call. Rejected — the selector already reads the flag; recomputing would
duplicate the mapping and risk divergence.

### D13: Cycle log — append-only JSONL
`cycle_log_{run_id}_{timestamp}.jsonl` — one JSON object per line, each with
`timestamp`, `round`, `event` (round_start, round_end, decision, accept,
rollback, stop, error), and an `details` payload. Append-only guarantees the
history is never lost; per-round filtering is a linear scan over the lines
(cheap for ≤ `max_rounds` rounds). A new log file is created per run (not
appended to a global file) so runs do not interleave.

**Alternative**: a SQLite log. Rejected — overkill for ≤ 5 rounds and adds a
dependency.

## Risks / Trade-offs

- **[Long-running cycle]** → A 5-round cycle with dev + holdout runs per round
  can take tens of minutes on a CPU llama-server. Mitigation: state is dumped
  after each round and the cycle is resumable; the CLI prints progress per
  round.
- **[Mid-round crash loses the round]** → No mid-round checkpoint (D8). Mitigation:
  resume reruns the incomplete round from its start; the runner's own
  checkpointing limits lost work within a single run.
- **[Stale candidates after acceptance]** → The queue is refilled at the start of
  each round (D4), but within a round the queue is consumed as-is. Mitigation:
  the queue is short (≤ `top_n`), and refilling mid-round would discard
  candidates that may still be useful.
- **[Holdout run cost]** → Running holdout every round (D10) doubles the run
  count. Mitigation: holdout runs are logged-only and can be disabled in a
  future change by skipping the holdout step; the default favors information.
- **[No orchestrator-level retry]** → Transient failures rely on the runner /
  AsyncTask retry. Mitigation: `stop_on_first_error=false` (default) lets the
  cycle continue when a step's internal retry succeeds; unrecoverable errors
  stop with a named step.
- **[Active version is in-memory]** → A crash between rounds loses the active
  version pointer if the state dump was not written. Mitigation: the state is
  dumped after every completed round, so the active version is always
  recoverable from the latest dump.

## Open Questions

- **Should the orchestrator cap the number of rollback attempts within a round
  independently of `max_consecutive_rollbacks`?** Currently a round tries
  candidates until the queue is exhausted, and the rollback counter is global.
  Deferred — add a per-round cap only if a round is observed to spin through the
  entire queue without progress.
- **Should the final summary include the full metrics history per round or only
  the final version's metrics?** Currently it includes the final version's
  metrics plus the accepted/rollback histories. Deferred — the per-round reports
  already carry full metrics; the summary can link to them.
- **Should `report_format=markdown` render a human-readable console summary in
  addition to the JSON file?** Deferred — the CLI prints a short summary table
  per round; a markdown report can be added behind `report_format` later.
