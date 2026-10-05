# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[cycle_orchestrator]` section to `config.toml` with keys
  `max_rounds`, `max_consecutive_rollbacks`, `dev_split`, `holdout_split`,
  `prompt_store_path`, `thesis_bank_path`, `candidate_queue_path`,
  `stop_on_first_error`, `dump_state_after_each_round`, `report_format`, and
  `output_dir`, with inline comments documenting each default. Verify the file
  parses with `tomllib.load` and the section round-trips.
- [x] 1.2 Create `cycle_orchestrator.py` with module docstring,
  `from __future__ import annotations`, a `CycleOrchestratorConfig` dataclass
  with `from_config()` reading `[cycle_orchestrator]` from `config.toml`
  (mirroring `VersionComparatorConfig`), and default constants
  (`DEFAULT_MAX_ROUNDS=5`, `DEFAULT_MAX_CONSECUTIVE_ROLLBACKS=2`,
  `DEFAULT_DEV_SPLIT="dev"`, `DEFAULT_HOLDOUT_SPLIT="holdout"`,
  `DEFAULT_STOP_ON_FIRST_ERROR=False`,
  `DEFAULT_DUMP_STATE_AFTER_EACH_ROUND=True`,
  `DEFAULT_REPORT_FORMAT="json"`, `DEFAULT_OUTPUT_DIR="data/results"`). Verify
  `CycleOrchestratorConfig.from_config()` returns the configured values and
  falls back to defaults for missing keys.
- [x] 1.3 Add a `CycleOrchestratorError(ValueError)` exception class and a
  `CycleState` dataclass holding `round_counter`, `active_version`,
  `active_version_path`, `thesis_bank_path`, `clusters_path`,
  `candidate_queue` (list of paths), `rollback_counter`, `latest_report_path`,
  `accepted_history` (list of version ids), `rollback_history` (list of
  `{round, version, reason}`), and `run_id`. Verify `CycleState` constructs with
  defaults and serializes to a plain dict via `dataclasses.asdict`.

## 2. Initial state loading

- [x] 2.1 Implement `load_active_prompt(config)` that returns the active
  `PromptArtifact`. On a fresh run it returns `CLASSIFICATION_PROMPT_V0`; when
  a `prompt_store_path` points to a composed `prompt_v*_*_*.json` artifact, it
  loads it via `prompt_composer.load_prompt_version` and builds a
  `PromptArtifact` via `version_comparator.build_prompt_artifact` (design D11).
  Raise `CycleOrchestratorError` naming the missing active prompt when neither
  is available. Verify it returns v0 by default and loads a composed artifact
  when the path is set.
- [x] 2.2 Implement `load_dataset_splits(config)` that loads the prepared
  dataset artifact (from `[runner].dataset_artifact` or the orchestrator's
  `dataset_artifact` path) via `dataset.load_artifact` and returns the dev and
  holdout splits. Raise `CycleOrchestratorError` naming the missing dataset when
  the artifact is absent. Verify it loads the existing
  `data/prepared_42_*.jsonl` and returns two splits.
- [x] 2.3 Implement `load_initial_state(config)` that returns a `CycleState`
  with the active prompt, dataset splits, an empty thesis bank, round counter 0,
  rollback counter 0, and empty histories (design D11). Verify the returned
  state has v0 active, an empty thesis bank, and zero counters.

## 3. Round execution — the nine-step sequence

- [x] 3.1 Implement `run_active_on_dev(state, config, async_task)` (step 1)
  that instantiates `BaselineRunner` with the active `PromptArtifact` on the
  dev split and returns the results artifact path. Verify it produces a
  `results_*_{version}_dev_*.jsonl` artifact and that passing a composed
  `PromptArtifact` uses that version in the filename.
- [x] 3.2 Implement `compute_dev_metrics(results_path, config)` (step 2) that
  calls `metrics.compute_metrics` and returns the metrics dict. Verify it
  returns accuracy, macro_f1, minority_f1, and confusion matrix for an existing
  results artifact.
- [x] 3.3 Implement `update_thesis_bank(results_path, config)` (step 3) that
  runs `thesis_analyzer` on the active results and returns the updated thesis
  bank path and clusters path. Verify it produces a `thesis_bank_*.json` dump
  and a clusters artifact.
- [x] 3.4 Implement `select_candidates(thesis_bank_path, config)` (step 4) that
  runs `rule_candidate_selector` on the thesis bank and returns the candidate
  artifact path plus the candidate queue (list of candidate paths or candidate
  dicts) populated from the artifact (design D4). Verify it produces a
  `rule_candidates_*.json` artifact and a non-empty queue.
- [x] 3.5 Implement `compose_new_version(candidate, active_prompt, config,
  async_task)` (step 5) that calls `prompt_composer.compose` on the candidate
  and the active prompt's base layers, validates limits, writes the prompt
  version via `prompt_composer.write_prompt_version`, and returns the new
  `PromptArtifact` plus its artifact path. Verify with a mocked composer that
  it returns a `PromptArtifact` with an incremented version.
- [x] 3.6 Implement `run_new_on_dev(new_prompt, config, async_task)` (step 6)
  that runs the new prompt version on dev via `BaselineRunner` and returns the
  results artifact path. Verify it produces a results artifact carrying the new
  version in its filename.
- [x] 3.7 Implement `decide_and_update(state, new_prompt, metrics_new,
  metrics_active, config)` (step 7–8) that calls `version_comparator.decide`,
  updates the rollback counter via `update_rollback_count`, and on accept
  updates the active version and `in_prompt` flags (design D12); on rollback
  pops the next candidate via `next_candidate`. Return a
  `(decision, reason, new_state, changed)` result. Verify: accept resets the
  rollback counter and updates the active version; rollback increments the
  counter and pops a candidate.
- [x] 3.8 Implement `run_round(state, config, async_task)` that chains steps
  1–9 (design D2): run active on dev, compute metrics, update thesis bank,
  select candidates, compose, run new on dev, decide, update state, and write
  the per-round report. On rollback within the round, consume the candidate
  queue (steps 5–7) until a candidate is accepted or the queue is exhausted.
  Return the updated `CycleState` and the report path. Verify with mocked
  sub-steps that all nine steps execute in order and the round counter
  increments by one.

## 4. Round counter, candidate queue, and rollback state

- [x] 4.1 Implement `increment_round_counter(state)` that returns a new state
  with `round_counter + 1` (design D3). Verify it does not mutate the input and
  that an aborted round (exception before increment) leaves the counter
  unchanged.
- [x] 4.2 Implement `check_stop(state, config)` that returns
  `(should_stop, reason)`: `max_rounds_reached` when
  `round_counter >= max_rounds`; `max_rollbacks_reached` when
  `rollback_counter >= max_consecutive_rollbacks`; `candidate_queue_exhausted`
  when a rollback left the queue empty; `(False, None)` otherwise. Verify each
  stop reason fires at the right threshold and that a healthy state returns
  `(False, None)`.
- [x] 4.3 Implement `refill_queue(state, candidate_artifact_path)` that reloads
  the candidate artifact and replaces `state.candidate_queue` with its
  candidates (design D4). Verify the queue is replaced (not appended) and that
  the new queue length matches the artifact's candidate count.

## 5. Accept handling and thesis bank update

- [x] 5.1 Implement `update_active_version(state, new_prompt, new_path)` that
  sets `state.active_version` to the new prompt, records the old version in
  `accepted_history`, and sets `active_version_path` (design D11). Verify the
  active version changes, the old version is appended to `accepted_history`,
  and the old artifact remains on disk.
- [x] 5.2 Implement `update_in_prompt_flags(thesis_bank_path,
  accepted_source_candidates)` that loads the thesis bank, sets `in_prompt=true`
  for theses whose `cluster_id` is in `accepted_source_candidates` and `false`
  for all others, and persists the updated bank (design D12). Verify: theses in
  accepted clusters get `in_prompt=true`; theses in other clusters get
  `in_prompt=false`; the persisted bank reflects the change on reload.

## 6. Per-round report and final summary

- [x] 6.1 Implement `write_report(round_number, active_version_at_start,
  new_version, decision, metrics_dev_active, metrics_dev_new,
  metrics_holdout_active, metrics_holdout_new, rollback_counter,
  changed_decisions, next_action, run_id, config)` that serializes to
  `report_{run_id}_round{N}_{timestamp}.json` with all required fields (design
  D7). Verify the file is valid JSON, the filename matches the pattern, and all
  fields round-trip on load.
- [x] 6.2 Implement `load_report(path)` that restores a per-round report without
  recomputation. Verify loading a file written by `write_report` reproduces
  every field.
- [x] 6.3 Implement `write_summary(state, final_dev_metrics,
  final_holdout_metrics, stop_reason, run_id, config)` that serializes to
  `summary_{run_id}_{timestamp}.json` with `total_rounds`, `stop_reason`,
  `final_active_version`, `final_dev_metrics`, `final_holdout_metrics`,
  `accepted_history`, `rollback_history`, and `all_changed_decisions` (design
  D7). Verify the file is valid JSON and the stop reason is one of the four
  enumerated values.
- [x] 6.4 Verify `write_summary` rejects a `stop_reason` outside the enumerated
  set (`max_rounds_reached`, `max_rollbacks_reached`,
  `candidate_queue_exhausted`, `unrecoverable_error`) with a
  `CycleOrchestratorError`.

## 7. State dump and resumability

- [x] 7.1 Implement `dump_state(state, config)` that serializes the full
  `CycleState` to `state_{run_id}_{timestamp}.json` (design D6). Verify the file
  is valid JSON, the filename contains `run_id` and a timestamp, and the
  `round_counter`, `active_version`, `thesis_bank_path`, `candidate_queue`,
  `rollback_counter`, and `latest_report_path` round-trip on load.
- [x] 7.2 Implement `load_state(path)` that restores a `CycleState` from a
  dumped JSON file. Verify that loading a file written by `dump_state`
  reproduces the same `round_counter`, `active_version`, `candidate_queue`, and
  `rollback_counter`.
- [x] 7.3 Implement `resume_from_state(state_path, config, async_task)` that
  loads a dumped state and continues the cycle from `round_counter + 1`,
  skipping completed rounds (design D8). Verify with a mocked state at
  round 3 that resume starts at round 4 and does not re-run rounds 1–3.
- [x] 7.4 Verify resume preserves the thesis bank, clusters, active version,
  and rollback counter from the dumped state (spec scenario: Resume preserves
  state).

## 8. Cycle logging

- [x] 8.1 Implement `log_event(event, round, details, run_id, config)` that
  appends one JSON object (`{timestamp, round, event, details}`) to
  `cycle_log_{run_id}_{timestamp}.jsonl` (design D13). Verify the file is
  append-only (two calls produce two lines), each line is valid JSON, and the
  timestamp and round are present.
- [x] 8.2 Implement `read_cycle_log(path, round=None)` that reads the JSONL log
  and returns all events, or only events for a given round when `round` is
  provided. Verify: with no filter all events are returned; filtering by round
  N returns only that round's events.
- [x] 8.3 Verify the orchestrator logs `round_start`, `round_end`, `decision`,
  `accept`/`rollback`, `stop`, and `error` events during a full round. Verify
  each event carries a timestamp and round number.

## 9. CLI and end-to-end integration

- [x] 9.1 Implement the argparse CLI (`python cycle_orchestrator.py [--config
  config.toml] [--endpoint <url>] [--resume <state.json>]`) that loads config,
  loads initial state (or resumes from `--resume`), runs the cycle via
  `asyncio.run`, writes per-round reports, dumps state after each round when
  `dump_state_after_each_round` is true, writes the final summary on stop, and
  prints a per-round summary table. Verify the CLI `--help` lists all flags and
  that a dry run with `max_rounds=0` writes only a summary with stop reason
  `max_rounds_reached`.
- [x] 9.2 Run the orchestrator end-to-end for `max_rounds=1` against a running
  LLM server on the small prepared dataset. Verify: one round report is written
  with all fields, a state dump is written, a final summary is written with a
  valid stop reason, the cycle log contains `round_start` and `round_end`
  events, and the active version after the round is either v0 (rollback) or a
  composed version (accept).
- [x] 9.3 Run the orchestrator for `max_rounds=5` end-to-end on the small
  dataset. Verify the cycle completes without crashes, five round reports are
  written, the final summary's `total_rounds` is 5, and the stop reason is
  `max_rounds_reached`. Then resume from the round-3 state dump and verify the
  resumed cycle runs rounds 4–5 only and produces a summary with
  `total_rounds=5`.
