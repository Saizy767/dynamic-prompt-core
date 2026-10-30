# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[version_comparator]` section to `config.toml` with keys
  `max_consecutive_rollbacks`, `decision_metric`, `tie_breaker_metric`, and
  `output_dir`, with inline comments documenting each default. Verify the file
  parses with `tomllib.load` and the section round-trips.
- [x] 1.2 Create `version_comparator.py` with module docstring,
  `from __future__ import annotations`, a `VersionComparatorConfig` dataclass
  with `from_config()` reading `[version_comparator]` from `config.toml`
  (mirroring `PromptComposerConfig`), and default constants
  (`DEFAULT_MAX_CONSECUTIVE_ROLLBACKS=2`, `DEFAULT_DECISION_METRIC="macro_f1"`,
  `DEFAULT_TIE_BREAKER_METRIC="minority_f1"`, `DEFAULT_OUTPUT_DIR="data/results"`).
  Verify `VersionComparatorConfig.from_config()` returns the configured values and
  falls back to defaults for missing keys.
- [x] 1.3 Add a `VersionComparatorError(ValueError)` exception class to
  `version_comparator.py`. Verify it can be raised and caught as a `ValueError`
  subclass.

## 2. Runner extension for custom classification prompt

- [x] 2.1 Add an optional `classify_prompt: Optional[PromptArtifact] = None`
  parameter to `BaselineRunner.__init__` in `runner.py` (design D2). When
  provided, set `self._classify_prompt = classify_prompt` and
  `self._prompt_version = classify_prompt.version`; otherwise keep the existing
  default (`CLASSIFICATION_PROMPT_V0` / `config.prompt_version`). Verify a
  `BaselineRunner` constructed without `classify_prompt` still uses
  `CLASSIFICATION_PROMPT_V0` (backward compatibility), and one constructed with a
  custom `PromptArtifact` uses its `.text` and `.version`.

## 3. Artifact loading and validation

- [x] 3.1 Implement `load_new_version(path)` that reads a
  `prompt_v*_*_*.json` artifact and validates the required fields `version`,
  `text`, `hash`, `rules`, `source_candidates`, `base_version` (design D3,
  reusing `prompt_composer.load_prompt_version`). Raise
  `VersionComparatorError` naming the missing field on failure. Verify it loads
  a valid composer artifact and raises on an artifact missing
  `source_candidates`.
- [x] 3.2 Implement `build_prompt_artifact(version_dict)` that constructs a
  `PromptArtifact(version=..., layers=None, text=..., sha256=...)` from the
  loaded dict (design D3). Verify the returned `PromptArtifact` has the correct
  `version`, `text`, and `sha256`, and `layers` is `None`.
- [x] 3.3 Implement `load_active_version()` that returns the currently active
  `PromptArtifact` from `prompts.py` (default `CLASSIFICATION_PROMPT_V0`) and
  its results artifact path on dev. Raise `VersionComparatorError` when no active
  version is found. Verify it returns `CLASSIFICATION_PROMPT_V0` by default and
  raises when the active version is `None`.

## 4. Run new version on dev

- [x] 4.1 Implement `run_version(prompt_artifact, config, dataset_artifact,
  split, endpoint, run_id)` that constructs a `BaselineRunner` with the given
  `prompt_artifact` as `classify_prompt`, runs it on the dev split via
  `asyncio.run`, and returns the results artifact path (design D2). Verify with a
  mocked `BaselineRunner` that the correct `classify_prompt` is passed and the
  results artifact path is returned.
- [x] 4.2 Verify `run_version` uses the fixed extraction prompt
  (`EXTRACTION_PROMPT`) without modification — the runner's `self._extract_prompt`
  is always `EXTRACTION_PROMPT` regardless of the `classify_prompt` override.

## 5. Metrics comparison and decision logic

- [x] 5.1 Implement `compare_metrics(results_active, results_new, config)` that
  reuses `metrics.compare_versions` to compute deltas and changed predictions
  (design D4). Return `(metrics_active, metrics_new, deltas, changed)`. Verify
  it returns the correct deltas and changed-decisions list for two known results
  artifacts, and raises when the splits differ.
- [x] 5.2 Implement `decide(metrics_new, metrics_active, config)` that extracts
  the configured `decision_metric` and `tie_breaker_metric` from both metrics
  dicts and returns `(decision, reason)` where `decision` is `"accept"` or
  `"rollback"` (design D5). Verify: macro-F1 improvement → accept; macro-F1 tie
  with minority-F1 improvement → accept; macro-F1 tie with minority-F1 equal →
  accept (>= on primary); macro-F1 degradation → rollback; custom
  `decision_metric=accuracy` → decision based on accuracy.
- [x] 5.3 Implement `changed_decisions(results_active, results_new)` that
  returns the list of `{id, direction}` for examples whose `predicted_decision`
  changed, reusing the output of `metrics.compare_versions` (design D4). Verify
  it returns 8 entries when v1 fixes 5 and breaks 3, and an empty list when no
  predictions changed.

## 6. Rollback counter and candidate queue

- [x] 6.1 Implement `update_rollback_count(current_count, decision)` that
  returns `0` when `decision == "accept"` and `current_count + 1` when
  `decision == "rollback"` (design D6). Verify: accept with count 2 → 0;
  rollback with count 1 → 2; rollback with count 0 → 1.
- [x] 6.2 Implement `check_stop(rollback_count, config)` that returns
  `(True, "max_consecutive_rollbacks reached")` when `rollback_count >=
  config.max_consecutive_rollbacks`, otherwise `(False, None)`. Verify: count 2
  with limit 2 → stop; count 1 with limit 2 → continue; count 3 with limit 3 →
  stop.
- [x] 6.3 Implement `next_candidate(candidate_queue)` that pops and returns the
  next candidate artifact path from the queue, or returns `(None, "candidate
  queue exhausted")` when the queue is empty (design D7). Verify: a queue with
  one path returns that path and empties the queue; an empty queue returns
  `(None, "candidate queue exhausted")`. Verify the rejected version's
  `source_candidates` are not carried into the next candidate.

## 7. Decision artifact persistence

- [x] 7.1 Implement `write_decision(decision, new_version, active_version,
  metrics_new, metrics_active, diff_accuracy, diff_macro_f1, diff_minority_f1,
  rollback_count, reason, changed_decisions, run_id, config, output_dir)` that
  serializes to
  `decision_{run_id}_v{new_version}_vs_v{active_version}_{timestamp}.json`
  (design D8). Verify the file is valid JSON, the filename matches the pattern,
  and all required top-level keys are present (`decision`, `new_version`,
  `active_version`, `metrics_new`, `metrics_active`, `diff_accuracy`,
  `diff_macro_f1`, `diff_minority_f1`, `rollback_count`, `reason`,
  `changed_decisions`, `run_id`, `timestamp`, `created_at`, `metadata`).
- [x] 7.2 Implement `load_decision(path)` that restores the decision artifact
  without re-running the comparison (design D8). Verify that loading a file
  written by `write_decision` reproduces the same `decision`, `new_version`,
  `active_version`, `metrics_new`, `metrics_active`, `diff_*`, `rollback_count`,
  `reason`, and `changed_decisions`.

## 8. Logging and CLI integration

- [x] 8.1 Implement `log_decision(decision, new_version, active_version,
  metrics_new, metrics_active, diffs, rollback_count)` that logs all required
  fields at INFO level via `logging.info` (design D10). Verify the log output
  contains `new_version`, `active_version`, `metrics_new`, `metrics_active`,
  `diff_accuracy`, `diff_macro_f1`, `diff_minority_f1`, `rollback_count`, and
  `decision`.
- [x] 8.2 Implement the argparse CLI (`python version_comparator.py --artifact
  <prompt_v*.json> --active-results <results_*.jsonl> [--candidate-queue
  <path1> <path2> ...] [--rollback-count 0] [--config config.toml --endpoint
  <url>]`) that loads config, loads the new prompt version, builds a
  `PromptArtifact`, runs it on dev, computes metrics via `compare_metrics`,
  decides via `decide`, updates the rollback counter, checks for stop, writes
  the decision artifact, logs the decision, and prints a summary table (design
  D10). On rollback with candidates remaining, print the next candidate path.
  On stop, print the stop reason. Verify the CLI runs against an existing
  prompt-version artifact and prints the summary without errors.
- [x] 8.3 Run the comparator end-to-end on a real prompt-version artifact with a
  running LLM server, then reload the written decision artifact with
  `load_decision` and confirm the reloaded `decision`, `new_version`,
  `active_version`, `diff_*`, and `rollback_count` match the in-memory result.
  Verify the reload path does not re-run the comparison or call the LLM. Verify
  the `changed_decisions` list in the artifact matches the diff computed by
  `metrics.compare_versions`.
