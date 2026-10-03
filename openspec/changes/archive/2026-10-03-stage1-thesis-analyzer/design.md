# Design

## Context

The project has a linear Stage 1 pipeline: `dataset-and-prompt` →
`stage1-baseline-runner` → `stage1-metrics`. The runner emits a
`results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl` artifact whose
records carry `theses_raw`, `theses_norm`, `true_label`, `extract_status`, and
`classify_status`. Normalization (lemmatize + lower-case + whitespace collapse)
is already applied by the runner via `normalize.py`; the analyzer must not
re-normalize.

Each component is a standalone Python module with a dataclass config loaded from
a TOML section (`RunnerConfig`, `MetricsConfig`), artifact load/validate helpers,
and an argparse CLI. The analyzer follows the same pattern.

## Goals / Non-Goals

**Goals:**
- Build a thesis bank from a results artifact with dedup, embeddings, and clusters.
- Produce a reloadable dump so later runs can extend the bank without recomputation.
- Resolve the open questions with sensible defaults that are all config-overridable.

**Non-Goals:**
- Candidate selection for prompt rules (next component).
- Rule formulation, prompt optimization, version comparison.
- Cluster merging / aggregation (deferred).
- Cluster visualization.

## Decisions

### D1: Single-module component (`thesis_analyzer.py`)
Mirrors `metrics.py`: dataclass `ThesisAnalyzerConfig.from_config`, `load_results`
(reuse the runner artifact format), `ThesisBank` + `Cluster` dataclasses,
`analyze()`, `write_dump()` / `load_dump()`, argparse CLI.

**Alternative**: split into `thesis_bank.py` + `clustering.py` + `thesis_analyzer.py`.
Rejected for now — the bank and clustering are tightly coupled and the module
will be comparable in size to `metrics.py`. Split later if it grows.

### D2: Embedding model — `paraphrase-multilingual-MiniLM-L12-v2`
Default via `sentence-transformers`. Chosen because the dataset is Russian and
this model is multilingual, lightweight (384-dim), and fast on CPU. Configurable
through `[thesis_analyzer].embedding_model`.

**Alternative**: `intfloat/multilingual-e5-small` — higher quality but heavier and
requires a `query:` / `passage:` prefix convention. Start with MiniLM; switch via
config if quality is insufficient.

### D3: Cosine thresholds — configurable, defaults 0.60 / 0.75
`cosine_lower=0.60`, `cosine_upper=0.75`. Below 0.60 → not attached. Above 0.75 →
attached directly. Between → grey zone, resolved by Manhattan distance.

**Rationale**: 0.60 is a common weak-similarity floor for short normalized phrases;
0.75 leaves a meaningful grey band. Both are in `[thesis_analyzer]`.

### D4: Manhattan threshold — configurable, default 8.0
`manhattan_threshold=8.0` on L1 distance of the (384-dim) embedding vectors.
This is a rough heuristic; the value is config-tunable and can be calibrated after
the first run by inspecting the dump.

### D5: Cluster limit — configurable, default 20
`max_clusters=20` for a dataset with <200 examples. When the limit is reached and
a thesis matches no cluster, attach it to the nearest cluster by cosine and log
the forced assignment. This keeps every thesis in a cluster (no orphans), which
simplifies downstream candidate selection.

**Alternative**: log as `unassigned`. Rejected — orphans complicate the next stage
and the nearest cluster is still the best available grouping.

### D6: Single-file dump
One JSON file: `thesis_bank_{run_id}_{prompt_version}_{timestamp}.json` with
top-level keys `theses` (dict keyed by `text_norm`) and `clusters` (dict keyed by
`cluster_id`), plus metadata (`run_id`, `prompt_version`, `embedding_model`,
config snapshot). Simpler to load, atomically write, and version.

**Alternative**: separate `bank_*.json` + `clusters_*.json`. Rejected — they are
always needed together and a single file avoids partial-load inconsistency.

### D7: Centroid update — incremental mean
On cluster creation, centroid = first thesis embedding. On each new member,
centroid = running mean of all member embeddings (incrementally updated without
recomputing from scratch). Stored as a numpy array in the dump (serialized as a
list).

### D8: Reuse `metrics.py` artifact-name parsing
The results artifact filename pattern `results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl`
is already parsed by `metrics.parse_artifact_name`. Import and reuse it rather
than duplicating the regex.

## Risks / Trade-offs

- **[Embedding model download on first run]** → `sentence-transformers` downloads
  the model from Hugging Face on first use. Mitigation: document in config
  comments; the model path can be pre-cached.
- **[Manhattan threshold is a guess]** → The default 8.0 is uncalibrated for
  384-dim MiniLM embeddings. Mitigation: configurable; inspect the first dump to
  tune.
- **[Centroid drift with incremental mean]** → Adding many theses to one cluster
  shifts its centroid, potentially misassigning later theses. Mitigation: stable
  enough for <200 examples; revisit if cluster quality degrades. Full recomputation
  is the deferred cluster-merge non-goal.
- **[sentence-transformers adds a heavy dependency]** → It pulls in torch.
  Mitigation: it is the standard embedding library; lazy-import so the module
  loads without it (matching `normalize.py`'s lazy-import pattern for pymorphy3/spaCy).

## Open Questions

None remaining — all questions from the request are resolved by decisions D2–D6
above, each with a config-overridable default.
