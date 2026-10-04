# Proposal

## Why

The `stage1-thesis-analyzer` produces a thesis artifact with semantic clusters
and per-cluster counters, but nothing yet turns those clusters into actionable
prompt-rule candidates. To build a new prompt version, we need a repeatable
selection step that ranks clusters by precision, drops the noisy and
already-in-prompt ones, and emits a compact candidate list that the next
component (rule formulation) can consume without re-running analysis.

## What Changes

- Add a new component (`rule_candidate_selector.py`) that loads a thesis artifact
  (JSON with `metadata`, `theses`, and `clusters`) and schema-validates its
  clusters. The component is not tied to any specific upstream producer — it
  accepts any artifact with the right structure.
- Filter clusters by a configurable `frequency_threshold` (default 3), excluding
  clusters whose `frequency` is below the threshold.
- Exclude clusters with `in_prompt=true` from the candidate list; retain them in
  a separate `already_in_prompt` list for logging.
- Rank surviving clusters by `precision` descending, with `frequency` descending
  as the tie-breaker.
- Select the top-N candidates (configurable `top_n`, default 5); take fewer when
  fewer survive.
- For each candidate, attach up to `max_representative_theses` (default 3)
  representative theses chosen by descending thesis `frequency`.
- Persist a `rule_candidates_{run_id}_{prompt_version}_{timestamp}.json` artifact
  that is reloadable without re-running selection.
- Log four selection counters: `total_clusters`, `passed_frequency`,
  `excluded_in_prompt`, `selected_top_n`.
- Add a `[rule_candidate_selector]` section to `config.toml` for
  `frequency_threshold`, `top_n`, `max_representative_theses`, and `output_dir`.

## Capabilities

### New Capabilities
- `stage2-rule-candidate-selector`: Loads a thesis artifact, filters and
  ranks clusters by frequency and precision, excludes clusters already in the
  prompt, selects top-N candidates with representative theses, and persists a
  reloadable candidate-list artifact for downstream rule formulation.

### Modified Capabilities
<!-- None — this change introduces a new component without altering existing specs. -->

## Impact

- **New code**: `rule_candidate_selector.py` (component + CLI), mirroring the
  structure of `thesis_analyzer.py` and `metrics.py` (dataclass config from TOML,
  artifact load/validate, CLI via argparse).
- **Config**: new `[rule_candidate_selector]` section in `config.toml`.
- **Dependencies**: none new — reuses `numpy` (already present) and stdlib
  `tomllib`/`json`. No embedding model is loaded; selection is purely numerical.
- **Artifacts**: new `rule_candidates_*.json` files written to `data/results/`.
- **Upstream**: consumes any JSON thesis artifact with `metadata`, `theses`,
  and `clusters` (e.g. from `stage1-thesis-analyzer`).
- **Downstream**: feeds the future rule-formulation component (out of scope).
