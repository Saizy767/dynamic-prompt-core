# Proposal

## Why

The baseline runner produces a results artifact with per-example theses, but
those theses are not yet aggregated, deduplicated, or clustered. To select
candidate theses for prompt rules in later stages, we need a thesis bank with
normalized forms, embeddings, semantic clusters, and counters (precision,
frequency, demand, selected, in_prompt). This component turns the raw run
artifact into that structured bank, giving the next stage its input.

## What Changes

- Add a new component (`thesis_analyzer.py`) that loads a
  `stage1-baseline-runner` results artifact and builds an in-memory thesis bank
  keyed by normalized form (`theses_norm` from the artifact — no re-normalization).
- Deduplicate theses by exact `text_norm` match; merge morphological variants
  into a single bank entry with accumulated counters.
- Compute and store embeddings for each unique normalized thesis using a
  configurable embedding model.
- Cluster theses by cosine similarity with a grey-zone fallback to Manhattan
  distance, a configurable cluster limit, and stable centroids.
- Aggregate per-cluster counters (frequency-weighted precision, summed
  frequency/positive_hits/negative_hits/demand/selected) updated incrementally.
- Persist a `thesis_bank_{run_id}_{prompt_version}_{timestamp}.json` dump after
  each update that is reloadable without recomputing embeddings or clusters.
- Add a `[thesis_analyzer]` section to `config.toml` for embedding model,
  cosine thresholds, Manhattan threshold, and cluster limit.

## Capabilities

### New Capabilities
- `stage1-thesis-analyzer`: Loads a baseline-runner results artifact, builds a
  deduplicated thesis bank with embeddings and semantic clusters, tracks
  precision/frequency/demand/selected/in_prompt counters, and persists a
  reloadable thesis-bank dump.

### Modified Capabilities
<!-- None — this change introduces a new component without altering existing specs. -->

## Impact

- **New code**: `thesis_analyzer.py` (component + CLI), mirroring the structure
  of `metrics.py` and `runner.py` (dataclass config from TOML, artifact
  load/validate, CLI via argparse).
- **Config**: new `[thesis_analyzer]` section in `config.toml`.
- **Dependencies**: `numpy` (already a transitive dep via scikit-learn in
  metrics), `sentence-transformers` (new) for embeddings.
- **Artifacts**: new `thesis_bank_*.json` files written to `data/results/`
  alongside existing `results_*.jsonl` and `metrics_*.json` artifacts.
- **Upstream**: consumes `results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl`
  from `stage1-baseline-runner`; uses only records with
  `classify_status=ok` and `extract_status=ok`.
- **Downstream**: feeds the future candidate-selection component (out of scope).
