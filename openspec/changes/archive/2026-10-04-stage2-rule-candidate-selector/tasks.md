# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[rule_candidate_selector]` section to `config.toml` with keys
  `frequency_threshold`, `top_n`, `max_representative_theses`, `output_dir`, and
  inline comments documenting each default. Verify the file parses with
  `tomllib.load` and the section round-trips.
- [x] 1.2 Create `rule_candidate_selector.py` with module docstring,
  `from __future__ import annotations`, a `RuleCandidateSelectorConfig` dataclass
  with `from_config()` reading `[rule_candidate_selector]` from `config.toml`
  (mirroring `ThesisAnalyzerConfig`), and default constants
  (`DEFAULT_FREQUENCY_THRESHOLD=3`, `DEFAULT_TOP_N=5`,
  `DEFAULT_MAX_REPRESENTATIVE_THESES=3`, `DEFAULT_OUTPUT_DIR="data/results"`).
  Verify `RuleCandidateSelectorConfig.from_config()` returns the configured values
  and falls back to defaults for missing keys.

## 2. Artifact loading and validation

- [x] 2.1 Implement metadata reading from the artifact's JSON `metadata` section
  (`run_id`, `prompt_version`) instead of parsing the filename (design D6). Verify
  `load_thesis_artifact` returns metadata with `run_id` and `prompt_version` and
  raises `RuleCandidateSelectorError` when metadata is missing a required field.
- [x] 2.2 Implement `load_thesis_artifact(path)` that reads the JSON artifact and
  validates that every cluster has `cluster_id`, `frequency`, `precision`,
  `positive_hits`, `negative_hits` (design D2). Raise
  `RuleCandidateSelectorError` naming the `cluster_id` and the first missing
  field on failure. Verify it loads a valid artifact and raises on a
  cluster missing `precision`.
- [x] 2.3 Implement `cluster_in_prompt(cluster, theses)` that derives a cluster's
  `in_prompt` flag as true when all member theses have `in_prompt=true`, false
  when the cluster has no members or any member is not in the prompt (design D3).
  Verify: a cluster whose members all have `in_prompt=true` returns true; a
  cluster with one member `in_prompt=false` returns false; an empty cluster
  returns false.

## 3. Selection logic

- [x] 3.1 Implement `filter_by_frequency(clusters, threshold)` returning clusters
  with `frequency >= threshold`. Verify a cluster with `frequency=5` passes
  threshold 3 and a cluster with `frequency=2` is excluded.
- [x] 3.2 Implement `exclude_in_prompt(clusters, theses)` splitting clusters into
  `(candidates, already_in_prompt)` using `cluster_in_prompt`. Verify a cluster
  with all members `in_prompt=true` lands in `already_in_prompt` and is absent
  from `candidates`.
- [x] 3.3 Implement `rank_by_precision(clusters)` sorting by `precision`
  descending with `frequency` descending as tie-breaker. Verify: a cluster with
  `precision=0.9` ranks before one with `precision=0.8`; two clusters with
  `precision=0.85` are ordered by `frequency` descending.
- [x] 3.4 Implement `select_top_n(ranked_clusters, top_n)` returning the first
  `top_n` entries (or all when fewer remain). Verify 12 clusters with `top_n=5`
  yields 5 and 3 clusters with `top_n=5` yields 3.
- [x] 3.5 Implement `representative_theses(cluster, theses, max_theses)` returning
  up to `max_theses` member theses sorted by descending thesis `frequency`, ties
  broken by `positive_hits` descending, each as `{text_norm, text_raw, frequency}`
  (design D4). Verify a cluster with 5 members and `max_theses=3` returns the 3
  highest-frequency theses.

## 4. Candidate artifact persistence

- [x] 4.1 Implement `select_candidates(artifact, config)` that chains the filter,
  exclude, rank, and top-N steps, attaches `representative_theses` and a `rank`
  index to each candidate, and returns `(candidates, already_in_prompt,
  counters)` where `counters` has `total_clusters`, `passed_frequency`,
  `excluded_in_prompt`, `selected_top_n`. Verify the counters are consistent with
  the input (e.g. 10 clusters, 7 pass frequency, 2 excluded as in_prompt, 5
  selected).
- [x] 4.2 Implement `write_candidates(candidates, already_in_prompt, counters,
  run_id, prompt_version, config, output_dir)` that serializes to
  `rule_candidates_{run_id}_{prompt_version}_{timestamp}.json` with top-level keys
  `metadata`, `candidates`, `already_in_prompt` (design D5). Verify the file is
  valid JSON, the filename matches the pattern, and each candidate has `cluster_id`,
  `rank`, `precision`, `frequency`, `positive_hits`, `negative_hits`,
  `representative_theses`.
- [x] 4.3 Implement `load_candidates(path)` that restores the candidate list from
  a `rule_candidates_*.json` artifact without re-running selection. Verify that
  loading a file written by `write_candidates` reproduces the same candidates and
  `already_in_prompt` list.

## 5. CLI and integration

- [x] 5.1 Implement the argparse CLI (`python rule_candidate_selector.py
  --artifact <path> [--config config.toml]`) that loads config, loads the
  thesis artifact, runs `select_candidates`, writes the candidate artifact,
  logs the four selection counters at INFO level, and prints a summary table
  (design D7). Verify the CLI runs against an existing thesis artifact
  and prints the summary without errors.
- [x] 5.2 Run the selector end-to-end on a real thesis artifact,
  then reload the written candidate artifact with `load_candidates` and confirm
  the reloaded candidates match the in-memory candidates (same count, same
  `cluster_id` order, same `rank` and `precision` values). Verify the reload path
  does not re-run selection.
