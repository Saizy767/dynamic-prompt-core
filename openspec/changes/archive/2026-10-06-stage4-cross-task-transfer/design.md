# Design

## Context

The project has a complete optimization pipeline whose top-level driver is
`cycle_orchestrator.run_cycle` (`cycle_orchestrator.py:1100`). A completed cycle
leaves behind: an active prompt version in the `PromptStore`
(`prompt_store.py:187`), a thesis-bank dump produced by
`thesis_analyzer.write_dump` (`thesis_analyzer.py:382`) containing `ThesisEntry`
records and `Cluster` centroids, and per-round reports + a final summary. The
cycle's `load_initial_state` (`cycle_orchestrator.py:194`) accepts a
`prompt_store_path` and `thesis_bank_path` to seed a fresh run — today these are
used only for resume, but the same hooks let a transfer component inject a
pre-built starting point.

Every component follows the same pattern: a `*Config` dataclass with
`from_config()` reading a TOML section via `tomllib`, a `*Error(ValueError)`
exception, artifact load/validate helpers, and an argparse CLI. The transfer
component mirrors this pattern but, like the orchestrator, calls other
components' public functions rather than doing a single transform.

Key existing interfaces the transfer reuses:

- `PromptStore.get_active()` / `.get(version)` (`prompt_store.py`) — read the
  source active version and version history read-only.
- `thesis_analyzer.load_dump(path)` (`thesis_analyzer.py:456`) — load the source
  thesis bank and clusters into a `ThesisBank`-shaped dict.
- `thesis_analyzer.ThesisBank` / `ThesisEntry` / `Cluster`
  (`thesis_analyzer.py:230` / `:179` / `:202`) — build the seeded target
  collection. `Cluster.add_member` (`:215`) updates centroids incrementally.
- `thesis_analyzer.write_dump(...)` — persist the seeded target collection.
- `prompts.PromptLayer` / `render` / `PromptArtifact`
  (`prompts.py:17` / `:27` / `:39`) — assemble the transferred prompt version
  (base layers from source, reformulated task, empty rules).
- `cycle_orchestrator.CycleOrchestratorConfig` / `run_cycle`
  (`cycle_orchestrator.py:60` / `:1100`) — run the target cycle from the
  transferred starting point.
- `metrics.compute_metrics` (`metrics.py`) — compute the target baseline
  (empty-prompt) metrics for the effectiveness report.

## Goals / Non-Goals

**Goals:**
- Reuse a completed cycle's base layers, theses, and cluster centroids as the
  starting point for a structurally compatible target task.
- Keep source artifacts strictly read-only and fully isolated from target
  artifacts.
- Measure whether transfer accelerates convergence vs. cold start.
- Fit the existing component pattern (config dataclass, error class, CLI) with
  no new third-party dependencies.

**Non-Goals:**
- Modifying the source task or its artifacts (source is read-only).
- Transfer between tasks with different class structure (rejected at init).
- Training or fine-tuning a model for the target task.
- Multilingual transfer (source and target in different languages) — deferred;
  this change assumes a shared language. Thesis translation is out of scope.
- Automatically choosing the source task for a given target.
- Transfer to a model of a different size — separate component.
- Simultaneous transfer to multiple target tasks — one target per run.
- Mid-transfer checkpointing — the target cycle's own state dump covers resume.

## Decisions

### D1: Single-module component (`cross_task_transfer.py`)
Mirrors `cycle_orchestrator.py`: dataclass `CrossTaskTransferConfig.from_config`,
`CrossTaskTransferError(ValueError)`, `TaskSpec` dataclass for task descriptions,
`TransferResult` dataclass for the artifact, and an argparse CLI. The component
is the top-level entry point for a transfer run; it calls the orchestrator,
prompt store, and thesis analyzer via their public APIs.

**Alternative**: split transfer logic from the target-cycle driver into two
modules. Rejected — the transfer and the target cycle are tightly coupled (the
transfer builds the seed the cycle consumes), and a single module keeps the
seeding-to-cycle handoff explicit.

### D2: Task description — `TaskSpec` dataclass from config
Each task is a `TaskSpec` with `task_id`, `dataset_path`, `num_classes`,
`class_labels` (list of labels), and `metric` (one of `accuracy`, `macro_f1`,
`minority_f1`). `CrossTaskTransferConfig.from_config` reads `[cross_task_transfer]`
and builds two `TaskSpec` instances (`source` and `target`). A missing field
raises `CrossTaskTransferError` naming the field. `task_id` comes from config
(not derived from the directory name), making the transfer artifact filename
deterministic and independent of the on-disk layout.

**Alternative**: derive `task_id` from the artifacts directory name. Rejected —
the directory name is an operator convention, not a contract; config is explicit
and validated.

### D3: Structure compatibility check — class count + output contract
`_check_compatibility(source_spec, target_spec)` compares `num_classes` and the
output contract. The output contract is the `output_contract` layer text from the
source active prompt version; the target contract is declared in the target task
spec. When they differ (e.g., `decision` as string vs. integer), the transfer is
rejected with `CrossTaskTransferError` naming the mismatch. The thesis-extraction
schema is shared (both use the fixed `EXTRACTION_PROMPT` from `prompts.py`), so
no separate check is needed.

**Alternative**: infer the target contract from the target dataset. Rejected —
the dataset has labels, not an output contract; the contract is a prompt-layer
property declared in the task spec.

### D4: Load source artifacts — read-only via existing loaders
`_load_source_artifacts(config)` loads the source active prompt version via
`PromptStore(config.source_store_config).get_active()`, the source thesis bank
via `thesis_analyzer.load_dump(config.source_thesis_bank_path)`, and the source
cycle metrics from the source summary artifact. The `PromptStore` is opened
read-only in practice (the transfer never calls `save`/`activate` on it). A
missing artifact raises `CrossTaskTransferError` naming it. Source artifacts are
never held open; they are loaded into memory and the store is discarded.

**Alternative**: copy source artifacts to a staging area first. Rejected —
copying adds I/O and a failure mode (partial copy) with no benefit; the source is
read once and never written.

### D5: Transfer base layers — reuse `PromptLayer`, reformulate task via LLM
The transferred prompt version is built from the source active version's
`PromptLayer`: `role`, `output_contract`, and `fallback` are copied unchanged;
`task` is reformulated by a single LLM call that takes the source task layer and
the target task description (`class_labels`, `dataset_path`) and produces a
target-domain task layer; `rules` is set to `[]` (empty). The result is assembled
via `prompts.build_classification_prompt` (which enforces 3–5 rules — see D6 for
the empty-rules tension). The reformulation uses the same `AsyncTask` LLM client
as the cycle, at temperature 0 for determinism.

**Alternative**: copy the task layer unchanged. Rejected — the task layer
encodes the source domain; using it verbatim would misdescribe the target domain
and mislead the classifier.

### D6: Empty rules vs. `build_classification_prompt`'s 3–5 rule guard
`prompts.build_classification_prompt` (`prompts.py:58`) rejects prompts with
fewer than 3 rules. The transferred version needs zero rules (the spec requires
the rules layer to be cleared). The transfer bypasses the guard by constructing
the `PromptArtifact` directly via `PromptLayer` + `render` rather than
`build_classification_prompt`, and writes it to the target prompt store as the
seed active version. The target cycle's first round then selects and composes
rules via the standard pipeline, producing a compliant 3–5-rule version. The
seed version is an intermediate artifact, not a candidate for the classifier
until the cycle composes the first real version.

**Alternative**: seed with 3 placeholder rules. Rejected — placeholder rules
would be semantically meaningless and could bias the first round's thesis
extraction; an empty rules layer is honest about the state.

### D7: Seed target theses — copy `ThesisEntry` records with `source=transfer`
`_seed_target_theses(source_bank, target_bank, config)` iterates the source
thesis collection and, for each entry, creates a `ThesisEntry` in the target
`ThesisBank` with `text_raw`, `text_norm`, and `embedding` copied from the
source, `frequency=0`, `positive_hits=0`, `negative_hits=0` (so precision is
undefined/zero until the first target run), and a new `source=transfer` tag
stored in an extension field on the entry. When `max_transfer_theses` is set,
entries are sorted by descending source `frequency` and the top N are
transferred. The target bank's `assign_cluster` is not called for seeded theses
(they are pre-assigned to transferred clusters, D8); it is called only for new
theses the target cycle produces.

**Alternative**: recompute embeddings on the target. Rejected — the embedding
model is the same (shared config); recomputing wastes a model call per thesis
and could introduce float drift.

### D8: Seed target clusters — copy centroids as initial points
`_seed_target_clusters(source_bank, target_bank, config)` creates a `Cluster` in
the target bank for each source cluster, copying the `centroid` vector and
`cluster_id`, with `count=0` and zeroed counters (the transferred theses are
linked to these clusters by `cluster_id` but do not inflate cluster frequency
until target runs occur). When `transfer_clusters=false`, no clusters are
transferred and the target bank starts with empty clustering; transferred theses
are assigned to clusters by the standard `assign_cluster` logic on the first
target run. Transferred centroids are updated by the standard
`Cluster.add_member` path when new target theses join their clusters — no
special update logic is needed.

**Alternative**: recompute centroids from transferred theses. Rejected — the
source centroid is already the running mean of source members; recomputing from
the transferred subset (which may be budget-limited) would shift the centroid
away from the source's learned position.

### D9: Run target cycle — configure orchestrator with transferred seed
The transfer writes the seeded target thesis bank via
`thesis_analyzer.write_dump` to `target_artifacts_path` and the transferred
prompt version to the target `PromptStore`. It then builds a
`CycleOrchestratorConfig` pointing at the target dataset, target prompt store,
and seeded thesis bank path, with `max_rounds=config.max_rounds_target`, and
calls `cycle_orchestrator.run_cycle(target_config, endpoint, run_id)`. The
`run_id` is `transfer-{source_task_id}-to-{target_task_id}-{timestamp}`,
distinct from any source `run_id`. The cycle runs in its standard configuration;
no cycle component is modified. The cycle's `load_initial_state` loads the
transferred prompt as the active version and the seeded thesis bank as the
starting collection.

**Alternative**: drive the cycle steps directly instead of via `run_cycle`.
Rejected — the orchestrator already threads the round counter, candidate queue,
and rollback state; reimplementing that logic duplicates the orchestrator and
risks divergence.

### D10: Transfer artifact — single JSON file
`transfer_{source_task_id}_to_{target_task_id}_{timestamp}.json` containing
`source_version` (the source active version number), `target_final_version`
(the target cycle's final active version), `transferred_theses_count`,
`transferred_clusters_count`, `target_metrics_start` (metrics after the first
round with transfer), `target_metrics_final` (metrics after the final round),
`improvement` (`{absolute: float, relative: float}` on the configured metric),
and `stop_reason` (the target cycle's stop reason). Written to
`config.target_output_dir`.

**Alternative**: split the artifact from the effectiveness report. Rejected —
the report (D11) references the artifact's fields; keeping them in one file
makes the report a pure read of the artifact plus the baseline snapshot.

### D11: Effectiveness report — three metric snapshots + optional comparison
`transfer_report_{source_task_id}_to_{target_task_id}_{timestamp}.json`
containing `baseline` (target metrics on an empty prompt, computed via a single
`BaselineRunner` pass with `CLASSIFICATION_PROMPT_V0`), `after_first_round`
(the `target_metrics_start` from the transfer artifact), `after_final_round`
(the `target_metrics_final`), and `rounds_to_stop` (the target cycle's round
count). When `compare_with_cold_start=true`, the report additionally contains a
`cold_start` section: a second target cycle is run from a cold start (empty
thesis bank, v0 prompt) with the same `max_rounds_target`, and the report
records `cold_start.metrics_final`, `cold_start.rounds_to_stop`, and
`acceleration_rounds` (`cold_start.rounds_to_stop - transfer.rounds_to_stop`).
The comparison metric is the target task's configured `metric` (the same metric
the target cycle uses for decisions) — no separate comparison metric is
introduced.

**Alternative**: use a separate comparison metric. Rejected — the effectiveness
question is "did transfer help on the metric we optimize for?", which is the
target cycle's own metric.

### D12: Isolation — separate paths, run ids, and stores
Source and target use different `run_id` (D9), different `PromptStore` files
(`source_artifacts_path` vs. `target_artifacts_path`), different thesis bank
dumps, and different output directories. The transfer never opens the source
store for writing. If the target artifacts path already contains a prior run's
artifacts, the transfer rejects with `CrossTaskTransferError` naming the conflict
rather than overwriting or archiving — the operator resolves the conflict
explicitly (archive or delete the prior target run).

**Alternative**: auto-archive a prior target run. Rejected — silent archiving
could mask a mistake (e.g., pointing at the wrong target path); an explicit
error is safer.

### D13: Config dataclass — `[cross_task_transfer]` section
`CrossTaskTransferConfig.from_config(config_path)` reads `[cross_task_transfer]`
from `config.toml` via `tomllib`, mirroring `CycleOrchestratorConfig`. Keys:
`source_task_id` (required), `target_task_id` (required),
`source_artifacts_path` (required — path to the source cycle's output dir),
`target_artifacts_path` (required — path for target cycle artifacts),
`source_store_path` (required — source `prompt_store.json`),
`target_store_path` (required — target `prompt_store.json`),
`source_thesis_bank_path` (required — source thesis bank dump),
`max_transfer_theses` (default `None` = unlimited), `transfer_clusters`
(default `True`), `compare_with_cold_start` (default `False`),
`max_rounds_target` (default `5`), `output_dir` (default `data/results`),
`log_path` (default `data/cross_task_transfer.jsonl`). Defaults are module-level
constants matching the existing convention (`DEFAULT_*`).

**Alternative**: derive source paths from `source_artifacts_path`. Rejected —
the source output dir may contain multiple runs' artifacts; explicit paths are
unambiguous and validated at load time.

### D14: Operation log — append-only JSONL
`log_path` receives one JSON object per line:
`{timestamp, operation, task_id, details}`. Operations: `load_source`,
`check_compatibility`, `transfer_layers`, `transfer_theses`,
`transfer_clusters`, `run_target_cycle`, `write_artifact`, `write_report`,
`cold_start_run`. The log is opened in append mode (`"a"`) per write. A log
write failure is caught and printed to stderr, never blocking the transfer.

**Alternative**: reuse the cycle orchestrator's log. Rejected — the transfer is
a distinct operation with its own task ids; interleaving transfer and cycle
events in one log makes per-transfer filtering harder.

## Risks / Trade-offs

- **[Transferred theses may not fit the target domain]** → Source theses encode
  source-domain patterns; some may be noise on the target. Mitigation: the
  `source=transfer` tag and zeroed frequency mean transferred theses do not
  influence precision or candidate selection until target runs validate them;
  the target cycle's first round naturally down-weights theses that do not
  recur.
- **[Cold-start comparison doubles target cycle cost]** → Running the target
  cycle twice (D11) when `compare_with_cold_start=true` doubles the target
  inference cost. Mitigation: the mode is opt-in (default false); operators
  enable it when measuring transfer effectiveness, not on every transfer.
- **[Reformulated task layer quality]** → The LLM-reformulated task layer (D5)
  depends on a single formulation call. Mitigation: temperature 0 for
  determinism; the task layer is a short domain description, not a complex
  generation, so the risk of a bad formulation is low and the operator can
  inspect the transferred version before running the cycle.
- **[Source and target must share an embedding model]** → Transferred embeddings
  (D7) are only valid in the same embedding space. Mitigation: the transfer
  reads the source thesis bank's `embedding_model` from its metadata and
  validates it matches the target `[thesis_analyzer].embedding_model`; a
  mismatch raises `CrossTaskTransferError` at load time.
- **[Prior target artifacts conflict]** → The target path must be clean (D12).
  Mitigation: an explicit error names the conflict; the operator archives or
  deletes the prior run.
- **[Long-running transfer]** → A transfer with cold-start comparison runs two
  full cycles. Mitigation: the CLI prints progress per phase (load, transfer,
  target cycle, cold-start cycle); each cycle is independently resumable via the
  orchestrator's state dump.

## Open Questions

- **Should the transfer support partial-structure compatibility (e.g., same
  class count but different label names)?** Currently the check requires an
  exact output-contract match. Deferred — relax only if a concrete target task
  needs label-name remapping.
- **Should transferred theses be pruned after the first target round if they do
  not recur?** Currently they persist with zero frequency until target runs
  validate them. Deferred — add a post-first-round pruning pass only if
  transferred theses are observed to bloat the target collection without
  benefit.
- **Should the effectiveness report include a per-round metric trend, or only
  the three snapshots?** Currently three snapshots (baseline, first, final).
  Deferred — the target cycle's per-round reports already carry the full trend;
  the report can link to them.
