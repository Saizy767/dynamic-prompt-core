# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[cross_task_transfer]` section to `config.toml` with keys
  `source_task_id`, `target_task_id`, `source_artifacts_path`,
  `target_artifacts_path`, `source_store_path`, `target_store_path`,
  `source_thesis_bank_path`, `max_transfer_theses`, `transfer_clusters`,
  `compare_with_cold_start`, `max_rounds_target`, `output_dir`, and `log_path`,
  with inline comments documenting each default and which keys are required.
  Verify the file parses with `tomllib.load` and the section round-trips.
- [x] 1.2 Create `cross_task_transfer.py` with module docstring,
  `from __future__ import annotations`, a `CrossTaskTransferConfig` dataclass
  with `from_config()` reading `[cross_task_transfer]` from `config.toml`
  (mirroring `CycleOrchestratorConfig`), and default constants
  (`DEFAULT_MAX_TRANSFER_THESES=None`, `DEFAULT_TRANSFER_CLUSTERS=True`,
  `DEFAULT_COMPARE_WITH_COLD_START=False`, `DEFAULT_MAX_ROUNDS_TARGET=5`,
  `DEFAULT_OUTPUT_DIR="data/results"`,
  `DEFAULT_LOG_PATH="data/cross_task_transfer.jsonl"`). Verify
  `CrossTaskTransferConfig.from_config()` returns configured values, falls back
  to defaults for missing optional keys, and raises `CrossTaskTransferError`
  naming each required key when it is absent.
- [x] 1.3 Add a `CrossTaskTransferError(ValueError)` exception class, a `TaskSpec`
  dataclass holding `task_id`, `dataset_path`, `num_classes`, `class_labels`
  (list), and `metric` (str), and a `TransferResult` dataclass holding
  `source_version`, `target_final_version`, `transferred_theses_count`,
  `transferred_clusters_count`, `target_metrics_start`,
  `target_metrics_final`, `improvement` (dict with `absolute` and `relative`),
  and `stop_reason`. Verify `TaskSpec` and `TransferResult` construct with
  defaults and serialize to a plain dict via `dataclasses.asdict`.

## 2. Task definition and structure compatibility

- [x] 2.1 Implement `_validate_task_spec(spec, role)` that checks all required
  fields are present (`task_id`, `dataset_path`, `num_classes`, `class_labels`,
  `metric`) and that `metric` is one of `accuracy`, `macro_f1`,
  `minority_f1`. Raise `CrossTaskTransferError` naming the missing or invalid
  field and the role (`source` or `target`). Verify: a valid spec passes; a spec
  missing `metric` fails naming `metric`; a spec with `metric="rmse"` fails
  naming the invalid value.
- [x] 2.2 Implement `_check_compatibility(source_spec, target_spec,
  source_output_contract)` that compares `num_classes` and the output contract
  (design D3). Raise `CrossTaskTransferError` naming the mismatch when
  `num_classes` differs or the output contract strings differ. Verify: same
  structure passes; different class counts fail naming the class-count mismatch;
  different output contracts fail naming the contract mismatch.

## 3. Load source artifacts

- [x] 3.1 Implement `_load_source_prompt(config)` that opens the source
  `PromptStore` read-only via `PromptStoreConfig` pointing at
  `source_store_path` and returns `store.get_active()`. Raise
  `CrossTaskTransferError` naming the missing artifact when no active version
  exists. Verify: a source store with an active version returns the record; a
  store with no active version raises `CrossTaskTransferError` naming the
  missing active version.
- [x] 3.2 Implement `_load_source_thesis_bank(config)` that loads the source
  thesis bank dump via `thesis_analyzer.load_dump(config.source_thesis_bank_path)`
  and validates the `embedding_model` in the dump metadata matches the target
  `[thesis_analyzer].embedding_model` (design Risks). Raise
  `CrossTaskTransferError` naming the mismatch when the embedding models differ.
  Verify: a matching model loads; a mismatched model raises
  `CrossTaskTransferError` naming both models.
- [x] 3.3 Implement `_load_source_artifacts(config)` that calls 3.1 and 3.2 and
  returns a `SourceArtifacts` namedtuple (`prompt_record`, `thesis_dump`,
  `output_contract`). Verify: with a complete source cycle, all three are
  loaded; with a missing thesis bank path, the error names the missing file.

## 4. Transfer base layers

- [x] 4.1 Implement `_extract_source_layers(prompt_record)` that reconstructs a
  `PromptLayer` from the source active prompt version's `text` by parsing the
  `## Role`, `## Task`, `## Rules`, `## Output contract`, and `## Fallback`
  sections (matching `prompts.render`). Verify: the v0 prompt round-trips
  through `render` → parse → `render` to identical text.
- [x] 4.2 Implement `async _reformulate_task_layer(source_task_layer,
  target_spec, async_task)` that makes a single LLM call (temperature 0) taking
  the source task layer and the target `class_labels` / `dataset_path` and
  returns a reformulated task layer string for the target domain (design D5).
  Verify: the call returns a non-empty string shorter than a configurable max
  (default 500 chars); a failed LLM call raises `CrossTaskTransferError` naming
  the formulation failure.
- [x] 4.3 Implement `_build_transferred_prompt(source_layers, target_task_layer)`
  that assembles a `PromptLayer` with `role`, `output_contract`, and `fallback`
  copied from the source, `task` set to the reformulated target task layer, and
  `rules=[]` (empty), then builds a `PromptArtifact` via `PromptLayer` +
  `render` directly (bypassing `build_classification_prompt`'s 3–5 rule guard,
  design D6). Verify: the transferred prompt's rules list is empty and its
  role/output_contract/fallback match the source.

## 5. Seed target thesis collection

- [x] 5.1 Implement `_seed_target_theses(source_bank, target_bank,
  max_transfer_theses)` that iterates the source thesis collection and creates a
  `ThesisEntry` in the target `ThesisBank` for each, copying `text_raw`,
  `text_norm`, and `embedding`, with `frequency=0`, `positive_hits=0`,
  `negative_hits=0`, and a `source=transfer` tag stored in an extension field
  (design D7). When `max_transfer_theses` is set, sort by descending source
  `frequency` and transfer the top N. Return the count transferred. Verify: 120
  source theses produce 120 target entries all tagged `source=transfer` with
  zero frequency; `max_transfer_theses=50` transfers only the 50 highest
  frequency theses.
- [x] 5.2 Verify transferred theses do not participate in precision computation:
  after seeding, every transferred thesis has `frequency=0` and
  `precision == 0.0`, and the target bank's cluster counters are zeroed (no
  transferred thesis inflates cluster frequency). Verify a candidate selector run
  on the seeded bank does not rank transferred theses by precision.

## 6. Seed target clusters

- [x] 6.1 Implement `_seed_target_clusters(source_bank, target_bank)` that
  creates a `Cluster` in the target bank for each source cluster, copying the
  `centroid` vector and `cluster_id`, with `count=0` and zeroed counters (design
  D8). Link transferred theses to these clusters by `cluster_id` without calling
  `assign_cluster`. Return the count transferred. Verify: 20 source clusters
  produce 20 target clusters with copied centroids and zero counters.
- [x] 6.2 Verify transferred centroids update when new target theses are added:
  after seeding, call `target_bank.add_or_update(new_text, new_raw, label)` for
  a thesis that matches a transferred cluster; the cluster's centroid shifts and
  its counters increment. Verify a cluster that receives no new theses keeps its
  original centroid.
- [x] 6.3 Implement the `transfer_clusters=false` path: when the config disables
  cluster transfer, skip 6.1 and let transferred theses be assigned by the
  standard `assign_cluster` logic on the first target run. Verify: with
  `transfer_clusters=false`, the target bank starts with zero clusters and
  transferred theses have `cluster_id=None` until the first target run.

## 7. Run target cycle

- [x] 7.1 Implement `_write_transferred_seed(config, transferred_prompt,
  target_bank)` that writes the transferred prompt version to the target
  `PromptStore` (via `init_store` + `save` + `activate`) and the seeded thesis
  bank to `target_artifacts_path` via `thesis_analyzer.write_dump`. Return the
  target store path and thesis bank path. Verify: after writing, the target
  store's active version has the transferred prompt text and the target thesis
  bank dump loads with the transferred theses and clusters.
- [x] 7.2 Implement `_build_target_cycle_config(config, target_store_path,
  target_thesis_bank_path)` that builds a `CycleOrchestratorConfig` pointing at
  the target dataset, target prompt store, and seeded thesis bank path, with
  `max_rounds=config.max_rounds_target` and a distinct `run_id` of
  `transfer-{source_task_id}-to-{target_task_id}-{timestamp}` (design D9).
  Verify: the built config's `prompt_store_path` and `thesis_bank_path` point at
  the target paths and `max_rounds` matches `max_rounds_target`.
- [x] 7.3 Implement `async _run_target_cycle(config, transferred_prompt,
  target_bank, async_task, endpoint)` that calls 7.1, 7.2, then
  `cycle_orchestrator.run_cycle(target_config, endpoint, run_id)`. Return the
  final `CycleState` and the target metrics. Verify: the target cycle runs to
  completion and its `run_id` differs from any source `run_id`; the target
  store's final active version is a composed (3–5 rule) prompt, not the
  transferred empty-rules seed.
- [x] 7.4 Verify source artifacts are unchanged after the target cycle: compare
  the source `PromptStore` file and source thesis bank dump before and after the
  transfer. Verify both files are byte-identical after the transfer run.

## 8. Transfer artifact

- [x] 8.1 Implement `_compute_improvement(metrics_start, metrics_final, metric)`
  that returns `{absolute: float, relative: float}` where `absolute` is
  `final - start` and `relative` is `(final - start) / start` when `start > 0`,
  else `0.0`. Verify: start=0.60, final=0.75 → absolute=0.15,
  relative=0.25; start=0.0 → relative=0.0.
- [x] 8.2 Implement `write_transfer_artifact(config, result, timestamp)` that
  writes a JSON file named
  `transfer_{source_task_id}_to_{target_task_id}_{timestamp}.json` to
  `config.output_dir` containing all `TransferResult` fields (design D10).
  Verify: the file is named correctly and loads as JSON with all fields present
  and accessible without re-running the cycle.

## 9. Effectiveness report

- [x] 9.1 Implement `async _compute_target_baseline(config, async_task,
  endpoint)` that runs the target dataset through `BaselineRunner` with
  `CLASSIFICATION_PROMPT_V0` (empty-prompt baseline) and computes metrics via
  `metrics.compute_metrics`. Return the baseline metrics dict. Verify: the
  baseline metrics contain the configured `metric` key with a float value.
- [x] 9.2 Implement `write_effectiveness_report(config, baseline_metrics,
  transfer_result, target_state, timestamp)` that writes
  `transfer_report_{source_task_id}_to_{target_task_id}_{timestamp}.json`
  containing `baseline`, `after_first_round` (from
  `transfer_result.target_metrics_start`), `after_final_round` (from
  `transfer_result.target_metrics_final`), and `rounds_to_stop` (from
  `target_state.round_counter`) (design D11). Verify: the report contains three
  metric snapshots and the round count, and loads as JSON.

## 10. Cold-start comparison

- [x] 10.1 Implement `async _run_cold_start_cycle(config, async_task, endpoint)`
  that builds a `CycleOrchestratorConfig` for the target task with an empty
  thesis bank (`thesis_bank_path=""`) and the v0 baseline prompt
  (`prompt_store_path=""`), then runs `cycle_orchestrator.run_cycle`. Return the
  cold-start `CycleState` and final metrics. Verify: the cold-start cycle runs
  independently with a `run_id` distinct from the transfer `run_id`.
- [x] 10.2 Implement `_write_comparison_section(report, cold_start_state,
  cold_start_metrics, transfer_state, transfer_result)` that adds a `cold_start`
  section to the effectiveness report with `cold_start.metrics_final`,
  `cold_start.rounds_to_stop`, and
  `acceleration_rounds = cold_start.rounds_to_stop - transfer.rounds_to_stop`
  (design D11). Verify: when the transfer cycle stops in 3 rounds and cold start
  in 5, `acceleration_rounds` is 2.
- [x] 10.3 Verify cold-start comparison is disabled by default: when
  `compare_with_cold_start` is unset in config, only the transfer cycle runs and
  the report has no `cold_start` section. Verify when
  `compare_with_cold_start=true`, both cycles run and the report contains the
  `cold_start` section.

## 11. Isolation between tasks

- [x] 11.1 Implement `_check_target_path_clean(config)` that verifies the target
  artifacts path does not already contain a prior run's transfer artifact or
  target prompt store (design D12). Raise `CrossTaskTransferError` naming the
  conflict when prior artifacts are found. Verify: a clean target path passes; a
  path with an existing `prompt_store.json` raises `CrossTaskTransferError`
  naming the conflict.
- [x] 11.2 Verify source and target use different `run_id` values, different
  `PromptStore` files, and different thesis bank dump paths throughout the
  transfer. Verify the source `PromptStore` file is never opened for writing
  (no `save`/`activate` calls on the source store instance).

## 12. Logging of transfer

- [x] 12.1 Implement `_log_operation(config, operation, task_id, details)` that
  appends one JSON object (`{timestamp, operation, task_id, details}`) to
  `config.log_path` in append mode (design D14). A log write failure is caught
  and printed to stderr, never blocking the transfer. Verify: two operations
  produce two lines, each valid JSON with `timestamp` and `task_id`; a log path
  in a non-existent directory does not crash the transfer.
- [x] 12.2 Wire logging into `_load_source_artifacts` (`load_source`),
  `_check_compatibility` (`check_compatibility`), `_build_transferred_prompt`
  (`transfer_layers`), `_seed_target_theses` (`transfer_theses`),
  `_seed_target_clusters` (`transfer_clusters`), `_run_target_cycle`
  (`run_target_cycle`), `write_transfer_artifact` (`write_artifact`),
  `write_effectiveness_report` (`write_report`), and `_run_cold_start_cycle`
  (`cold_start_run`). Verify each operation appends a log line with the correct
  `operation` name and `task_id`.

## 13. CLI and end-to-end integration

- [x] 13.1 Implement the argparse CLI (`python cross_task_transfer.py
  [--config config.toml] [--endpoint URL]`) with a `run` subcommand that
  executes the full transfer: validate task specs, check compatibility, load
  source artifacts, transfer base layers, seed theses and clusters, run the
  target cycle, write the transfer artifact and effectiveness report, and
  optionally run the cold-start comparison. Verify `--help` lists the
  subcommand and a `run` with a valid config completes without error.
- [x] 13.2 Run an end-to-end transfer using a completed source cycle's
  artifacts: `run` produces a `transfer_*_to_*_*.json` artifact and a
  `transfer_report_*_to_*_*.json` report. Verify the artifact's
  `transferred_theses_count` and `transferred_clusters_count` match the source
  collection, `target_metrics_final` is populated, and the source artifacts are
  byte-identical before and after. Verify the log contains one line per
  operation in order.
- [x] 13.3 Run an end-to-end transfer with `compare_with_cold_start=true` and
  verify the report contains the `cold_start` section with
  `acceleration_rounds` computed correctly. Verify the cold-start cycle's
  artifacts are stored separately from the transfer cycle's artifacts (different
  `run_id`, different output paths).
- [x] 13.4 Verify the transfer rejects incompatible tasks: run with a source
  task of 2 classes and a target of 3 classes and confirm the CLI exits with a
  `CrossTaskTransferError` naming the class-count mismatch, without modifying
  any source or target artifacts.
