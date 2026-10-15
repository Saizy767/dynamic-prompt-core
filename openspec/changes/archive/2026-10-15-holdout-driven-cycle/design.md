# Design

## Context

The current cycle (`steps.py:run_round`) decides accept/rollback using
dev-split metrics. Candidates are selected from dev-side thesis analysis,
creating an explore/evaluate coupling on the same split. Holdout metrics
are computed after the accumulation loop but are explicitly marked
`logged only` and never influence the decision (`steps.py:519`).

The accumulation loop (`steps.py:421`) already implements greedy rule
accumulation: accept → `best = new` → continue; rollback → break. The
structural shape is correct; the metric source and decision rule are what
change.

Versioning is linear (`classify-v0 → classify-v1 → …`) via
`composer.py:increment_version`. State tracks `round_counter` and
`rollback_counter`. Stop conditions are `max_rounds_reached`,
`max_rollbacks_reached`, `candidate_queue_exhausted`.

## Goals / Non-Goals

**Goals:**
- Drive accept/rollback by holdout metrics, not dev.
- Require strict improvement (`new > best`) for acceptance.
- Introduce `vN_k` versioning that exposes cycle/cluster structure.
- Replace round/rollback stop logic with cycle/no-improvement stop logic.
- Eliminate per-candidate dev runs (holdout only in accumulation loop).

**Non-Goals:**
- Changing the thesis collection method (still dev-side, step 1).
- Changing the candidate selection logic (`stage2-rule-candidate-selector`).
- Changing the composition/formulation logic (`stage3-prompt-composer`).
- Adding a nested outer-loop ("macro-cycle"). One cycle = one finalized
  version; the process is a flat sequence of cycles.
- Cross-validation or multi-split evaluation.

## Decisions

### D1: Holdout as sole decision metric in accumulation loop

**Decision**: Each `vN_k` candidate runs only on holdout
(`skip_extraction=True`). The active version's holdout metrics are computed
once per cycle (step 5) and cached as the baseline. Dev runs only for the
active version in step 1 (thesis collection + dev metrics for logging).

**Rationale**: Decouples exploration (dev theses) from evaluation (holdout
metrics), giving a valid generalization estimate. Avoids the cost of
per-candidate dev runs.

**Alternative considered**: Run candidates on both dev and holdout, use dev
for logging. Rejected by the user — dev runs during accumulation are
unnecessary since the decision is holdout-only.

### D2: Strict improvement (`new > best`)

**Decision**: `decide()` returns `accept` only when the holdout decision
metric is strictly greater. Equal → rollback. Tie-breaker is logged but
not used in the decision.

**Rationale**: Conservative — a cluster is added only if it genuinely
improves generalization. Prevents accumulating marginal rules that tie on
holdout but add prompt complexity.

**Alternative considered**: Keep `>=` with tie-breaker. Rejected by the
user — strict improvement is more conservative and avoids over-fitting
the prompt with non-improving rules.

### D3: `vN_k` versioning scheme

**Decision**: `v0` = baseline (`CLASSIFICATION_PROMPT_V0`). Cycle `N`
starts with base `v(N-1)`, produces intermediates `vN_1, vN_2, …`, and
finalizes the best as `vN`. The composer receives `cycle_number` and
`cluster_index` to generate the version string.

**Rationale**: Makes the cycle/cluster lineage visible in the version name.
`v2_3` immediately tells you: cycle 2, third cluster attempt.

**Alternative considered**: Keep linear `classify-vN` and track cycle/cluster
in metadata only. Rejected — the user wants the structure in the version
name itself.

**Implementation**: `increment_version()` in `composer.py` is replaced by
two functions: `intermediate_version(cycle, cluster) → "v{cycle}_{cluster}"`
and `finalized_version(cycle) → "v{cycle}"`. The `assemble_prompt` function
receives the version string explicitly instead of computing it internally.

### D4: Stop conditions — `max_cycles_reached` + `no_improvement`

**Decision**: Remove `max_consecutive_rollbacks` and
`candidate_queue_exhausted`. The process stops when:
- `cycle_counter >= max_cycles` → `max_cycles_reached`
- A cycle ends with no candidate accepted → `no_improvement`
- External `evaluate_stop_criteria` (plateau, budget) → as before
- Unrecoverable error → `unrecoverable_error`

**Rationale**: A rollback is the normal termination of a cycle's accumulation
(degradation detected). It is not a failure to count. The process should
stop only when a cycle produces zero improvement — meaning no cluster can
help the current base.

**Alternative considered**: Keep `max_consecutive_rollbacks` as a safety
net. Rejected — it conflates normal cycle termination with failure. The
`no_improvement` condition is cleaner and directly captures "stuck".

### D5: State changes

**Decision**: `CycleState` fields:
- `round_counter` → `cycle_counter` (rename)
- Add `cycle_number: int = 0` (current cycle number for versioning)
- Add `cluster_index: int = 0` (current cluster index, resets each cycle)
- Remove `rollback_counter` (no longer used)
- Remove `last_round_new_candidates` → `last_cycle_new_candidates` (rename)

**Migration**: Old state dumps are not resumable after this change (field
renames). The `load_state` function will handle missing fields with defaults
for best-effort, but a clean start is recommended.

### D6: Report changes

**Decision**: Per-cycle report includes:
- `cycle_number`, `base_version` (active at start), `finalized_version`
- `metrics_dev_active` (from step 1, logged)
- `metrics_holdout_active` (cached baseline), `metrics_holdout_finalized`
- `attempts`: list of `{version, cluster_id, holdout_metric, decision}`
- `accepted_cluster_ids`, `next_action`

**Stop reasons**: `VALID_STOP_REASONS` updated to
`{max_cycles_reached, no_improvement, unrecoverable_error, plateau_detected}`.
Removed: `max_rounds_reached`, `max_rollbacks_reached`,
`candidate_queue_exhausted`.

## Risks / Trade-offs

- **[Holdout noise]** Holdout is 20% of the dataset (config
  `holdout_ratio = 0.2`), so holdout metrics are noisier than dev. Strict
  improvement on a noisy metric may cause premature stopping.
  → **Mitigation**: The `no_improvement` stop is per-cycle, not per-candidate.
  A cycle can still accept multiple clusters before stopping. The noise
  affects only the boundary decision (last accepted vs first rejected).

- **[Cost shift]** Each candidate now requires a holdout run instead of a
  dev run. Holdout is smaller (20% vs 80%), so per-candidate cost decreases.
  But the active version needs both a dev run (step 1) and a holdout run
  (step 5), whereas before it only had a dev run. Net: fewer per-candidate
  costs, one extra holdout run per cycle.
  → **Mitigation**: `skip_extraction=True` on holdout makes it fast
  (classification only, no thesis extraction LLM calls).

- **[Breaking config]** `max_rounds` → `max_cycles`,
  `max_consecutive_rollbacks` removed, `--rounds` → `--cycles`.
  → **Mitigation**: Document in proposal. Old config files fail with a
  clear error on the renamed key.

- **[State incompatibility]** Old state dumps use `round_counter` and
  `rollback_counter`.
  → **Mitigation**: `load_state` uses `.get()` with defaults. A clean
  start is recommended; resume from old dumps is best-effort only.

## Migration Plan

1. Update `config.toml`: rename `max_rounds` → `max_cycles`, remove
   `max_consecutive_rollbacks` from `[cycle_orchestrator]` and
   `[version_comparator]`.
2. Update CLI: `--rounds` → `--cycles`.
3. Existing prompt artifacts (`classify-vN`) remain loadable as base
   versions; new artifacts use `vN_k` / `vN` naming.
4. Existing state dumps are not resumable — start a fresh cycle.
5. Existing reports/summaries remain readable (JSON); new ones use the
   updated schema.

## Open Questions

(none)
