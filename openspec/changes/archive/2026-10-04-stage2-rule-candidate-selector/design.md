# Design

## Context

The project has a linear Stage 1 pipeline: `dataset-and-prompt` →
`stage1-baseline-runner` → `stage1-metrics`, with `stage1-thesis-analyzer`
extending it to produce a thesis artifact (JSON with `metadata`,
`theses`, and `clusters`). That artifact contains top-level keys `theses`
(dict keyed by `text_norm`) and `clusters` (dict keyed by `cluster_id`), where
each cluster carries `cluster_id`, `frequency`, `precision`, `positive_hits`,
`negative_hits`, `demand`, `selected`, `member_norms`, and each thesis carries
`in_prompt` and `frequency`. The selector is not tied to any specific upstream
producer — it accepts any JSON artifact with this structure.

Each existing component is a standalone Python module with a dataclass config
loaded from a TOML section, artifact load/validate helpers, and an argparse CLI
(`runner.py`, `metrics.py`, `thesis_analyzer.py`). The selector follows the same
pattern. No embedding model is needed — selection is purely numerical over
already-computed cluster counters.

## Goals / Non-Goals

**Goals:**
- Turn a thesis artifact into a ranked, filtered candidate list for rule
  formulation.
- Produce a reloadable artifact so downstream components do not re-run selection.
- Resolve the open questions with config-overridable defaults.

**Non-Goals:**
- Rule formulation from candidates (next component).
- Prompt version assembly or optimization.
- Candidate quality evaluation on holdout.
- Cluster merging or thesis aggregation.

## Decisions

### D1: Single-module component (`rule_candidate_selector.py`)
Mirrors `thesis_analyzer.py` / `metrics.py`: dataclass
`RuleCandidateSelectorConfig.from_config`, `load_thesis_artifact(path)` with
schema validation, `select_candidates(...)` core logic, `write_candidates(...)` /
`load_candidates(...)`, argparse CLI.

**Alternative**: split selection logic from artifact I/O. Rejected — the module
is small and the I/O and selection are tightly coupled.

### D2: Cluster schema validation — required fields
On load, validate that every cluster has `cluster_id`, `frequency`, `precision`,
`positive_hits`, `negative_hits`, and `in_prompt`. Raise an error naming the
`cluster_id` and the first missing field, matching the error style of
`thesis_analyzer._validate_row`.

**Alternative**: validate via JSON Schema. Rejected — adds a dependency for a
flat field check that is clearer inline.

### D3: `in_prompt` at cluster level
The thesis artifact stores `in_prompt` per thesis, not per cluster. A cluster is
treated as `in_prompt=true` when all of its member theses have `in_prompt=true`.
This is computed at load time from `member_norms` and the thesis entries. When a
cluster has no members, it is treated as not in prompt.

**Alternative**: store `in_prompt` explicitly on the cluster in the artifact.
Rejected — the artifact format is owned by the upstream producer and changing it
is out of scope; deriving the flag keeps this component self-contained.

### D4: Representative theses — top-K by thesis frequency
For each candidate cluster, select up to `max_representative_theses` (default 3)
member theses sorted by descending thesis `frequency`, breaking ties by
`positive_hits` descending. Store `text_norm` and `text_raw` for each.

**Alternative**: a single most-frequent thesis, or all members. Rejected — a
single thesis loses coverage of the cluster's semantic range; all members bloats
the artifact and the downstream rule-formulation context.

### D5: Artifact format — single JSON file
`rule_candidates_{run_id}_{prompt_version}_{timestamp}.json` with top-level keys
`metadata` (`run_id`, `prompt_version`, `timestamp`, config snapshot),
`candidates` (list of candidate objects in rank order), and `already_in_prompt`
(list of `cluster_id` values excluded for logging). Each candidate object carries
`cluster_id`, `rank`, `precision`, `frequency`, `positive_hits`,
`negative_hits`, and `representative_theses` (list of `{text_norm, text_raw,
frequency}`).

**Alternative**: omit `already_in_prompt`. Rejected — keeping it in the artifact
makes the exclusion auditable without re-running selection, and it is small.

### D6: Metadata read from JSON content, not filename
The component reads `run_id` and `prompt_version` from the artifact's `metadata`
section rather than parsing them from the filename. This makes the component
independent of any specific filename convention — it accepts any JSON artifact
with the right structure.

**Alternative**: parse `run_id` / `prompt_version` from the filename via a regex.
Rejected — couples the component to a specific naming convention and breaks if
the upstream producer changes its filename format.

### D7: Selection counters logged via `logging.info`
Log `total_clusters`, `passed_frequency`, `excluded_in_prompt`,
`selected_top_n` at INFO level, matching the logging style of
`thesis_analyzer.py`. Also print a short CLI summary table.

## Risks / Trade-offs

- **[Cluster `in_prompt` derivation is all-members]** → A cluster with one
  thesis not in the prompt is treated as a candidate. Mitigation: this is the
  conservative direction (prefer re-proposing a cluster over silently skipping
  it); the rule-formulation stage can still deduplicate against the live prompt.
- **[Defaults tuned for <200 examples]** → `frequency_threshold=3` and `top_n=5`
  may be too strict or too loose for larger datasets. Mitigation: all three
  parameters are config-overridable.
- **[No holdout evaluation]** → Candidates are selected purely by precision and
  frequency without measuring incremental F1. Mitigation: that is an explicit
  non-goal; the holdout evaluation belongs to a later optimization stage.

## Open Questions

- **Default `frequency_threshold`**: 3 suits a dataset with <200 examples; should
  it be 2 to avoid dropping borderline clusters? Deferred — config-overridable,
  tune after the first run.
- **`top_n` vs final rule count**: should `top_n` be larger than the final number
  of rules (e.g. 10) so rule formulation can pick 5 by incremental F1? Deferred —
  raise via config if the formulation stage needs a wider pool.
- **Tie-breaker beyond frequency**: is `frequency` sufficient, or should
  `demand` / `selected` be considered? Deferred — add only if ranking proves
  unstable in practice.
- **`already_in_prompt` artifact**: kept inside the candidate artifact for now;
  split into a separate file only if downstream consumers need it independently.
