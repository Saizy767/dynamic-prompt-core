# Proposal

## Why

The `stage3-prompt-composer` produces a new classification prompt version, but
nothing yet evaluates whether that version is actually better than the active one.
Without a comparison-and-decision step, the optimization loop cannot accept or
reject a composed version, cannot track consecutive failures, and cannot feed the
next candidate back into composition on rollback. This change closes the loop by
running both versions on the same dev split, comparing metrics, deciding accept
or rollback, and persisting a reloadable decision artifact with full lineage.

## What Changes

- Add a new component (`version_comparator.py`) that loads a composed
  prompt-version artifact (`prompt_v*_*_*.json` from the composer) and
  schema-validates its fields (`version`, `text`, `hash`, `rules`,
  `source_candidates`, `base_version`).
- Load the currently active prompt version from `prompts.py` as the baseline for
  comparison. Fail with an error when no active version is found.
- Run the new prompt version on the same dev split used for the active version,
  using the same `BaselineRunner` and the same fixed extraction prompt, producing
  a results artifact for the new version.
- Compute the same metrics for the new version as for the active version (via
  `stage1-metrics`): accuracy, macro-F1, minority-class F1, confusion matrix,
  prediction distribution, and parse-failure count.
- Compare the new version's metrics against the active version's on dev. Report
  the difference for accuracy, macro-F1, and minority-class F1. Holdout metrics
  are logged but do not participate in the decision.
- Decide accept when the new version's macro-F1 on dev is greater than or equal
  to the active version's; use minority-class F1 as the tie-breaker when macro-F1
  is equal; roll back when both are lower.
- Maintain a consecutive-rollback counter: reset to zero on acceptance, increment
  on rollback, stop the cycle when the counter reaches `max_consecutive_rollbacks`
  (default 2).
- Return the next candidate from the candidate queue on rollback, without reusing
  the rejected version's rules. Stop the cycle when the queue is exhausted.
- Persist a `decision_{run_id}_v{new_version}_vs_v{active_version}_{timestamp}.json`
  artifact containing `decision`, `new_version`, `active_version`, `metrics_new`,
  `metrics_active`, `diff_accuracy`, `diff_macro_f1`, `diff_minority_f1`,
  `rollback_count`, and `reason`.
- Report the list of example ids whose `predicted_decision` changed between the
  active and new versions, with the direction of change
  (correct→incorrect, incorrect→correct).
- Log the decision process: new version, active version, metrics for both,
  differences, rollback count, and final decision.
- Add a `[version_comparator]` section to `config.toml` for
  `max_consecutive_rollbacks` (default 2), `decision_metric` (default
  `macro_f1`), and `tie_breaker_metric` (default `minority_f1`).

## Capabilities

### New Capabilities
- `stage2-version-comparator`: Loads a composed prompt version and the active
  version, runs both on the same dev split, computes comparable metrics, compares
  by macro-F1 with minority-class F1 as tie-breaker, decides accept or rollback,
  maintains a consecutive-rollback counter, returns the next candidate on
  rollback, and persists a reloadable decision artifact with lineage and changed
  predictions.

### Modified Capabilities
<!-- None — this change introduces a new component without altering existing specs. -->

## Impact

- **New code**: `version_comparator.py` (component + CLI), mirroring the structure
  of `prompt_composer.py` and `rule_candidate_selector.py` (dataclass config from
  TOML, artifact load/validate, CLI via argparse). Reuses `BaselineRunner` from
  `runner.py`, `compute_metrics` / `compare_versions` from `metrics.py`, and
  `PromptArtifact` from `prompts.py`.
- **Config**: new `[version_comparator]` section in `config.toml`.
- **Dependencies**: no new dependencies. Reuses `aiohttp` (via `BaselineRunner`),
  `AsyncTask`, `metrics`, `runner`, `prompts`, and `dataset`.
- **Artifacts**: new `decision_*_*_*.json` files written to `data/results/`.
- **Upstream**: consumes a `prompt_v*_*_*.json` artifact from
  `stage3-prompt-composer`, the active `PromptArtifact` from `prompts.py`, and a
  candidate queue (list of candidate artifacts) for rollback.
- **Downstream**: feeds the orchestrator that drives the full optimization cycle
  across rounds (out of scope).
