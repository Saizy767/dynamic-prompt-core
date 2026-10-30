# Design

## Context

The project has a linear optimization pipeline: `dataset-and-prompt` →
`stage1-baseline-runner` → `stage1-metrics` → `stage1-thesis-analyzer` →
`stage2-rule-candidate-selector` → `stage3-prompt-composer`. The composer emits a
`prompt_v{version}_{run_id}_{timestamp}.json` artifact with `version`, `text`,
`hash`, `rules` (list of `{cluster_id, text}`), `source_candidates` (list of
`cluster_id`), `base_version`, `created_at`, and `metadata`.

The runner (`runner.py`) runs a dataset split through the model with a fixed
extraction prompt and a classification prompt. It hardcodes
`CLASSIFICATION_PROMPT_V0` as `self._classify_prompt` in `BaselineRunner.__init__`
(line 184) and uses `self._prompt_version` (from `RunnerConfig`, default
`classify-v0`) in artifact filenames. After the run, the CLI auto-computes
metrics via `metrics.compute_metrics` and writes a metrics artifact.

`metrics.py` already provides `compare_versions(results_path_v0,
results_path_v1, config)` which loads two results artifacts, computes metrics
for both, returns deltas for accuracy / macro-F1 / minority-F1, and lists ids
whose `predicted_decision` changed with direction (`fixed` / `broke` / `flip`).

`prompts.py` exposes `PromptArtifact(version, layers, text, sha256)` and
`build_classification_prompt(version, layers)` which enforces 3–5 rules.

Each existing component is a standalone Python module with a dataclass config
from a TOML section, artifact load/validate helpers, and an argparse CLI. The
comparator follows the same pattern.

## Goals / Non-Goals

**Goals:**
- Determine whether a composed prompt version improves on the active version on
  dev, using macro-F1 with minority-class F1 as tie-breaker.
- Maintain a consecutive-rollback counter and stop the cycle after
  `max_consecutive_rollbacks`.
- Return the next candidate from the queue on rollback for re-composition.
- Persist a reloadable decision artifact with full lineage and changed
  predictions.

**Non-Goals:**
- Composition of the new prompt version (handled by the composer).
- Rule candidate selection (handled by `stage2-rule-candidate-selector`).
- Metrics computation internals (handled by `stage1-metrics`).
- Baseline run internals (handled by `stage1-baseline-runner`).
- Statistical significance testing (bootstrap, confidence intervals) — deferred.
- Multi-version comparison beyond the active vs. new pair.
- Cycle orchestration across all five rounds — handled by a separate orchestrator.

## Decisions

### D1: Single-module component (`version_comparator.py`)
Mirrors `prompt_composer.py` / `rule_candidate_selector.py`: dataclass
`VersionComparatorConfig.from_config`, `load_prompt_version(path)` with schema
validation (reusing the composer's loader), `build_prompt_artifact(version_dict)`
to construct a `PromptArtifact` from the loaded dict, `run_version(prompt_artifact,
config, ...)` via `BaselineRunner`, `decide(metrics_new, metrics_active, config)`
decision logic, `changed_decisions(results_active, results_new)` diff,
`write_decision(...)`, argparse CLI.

**Alternative**: split the runner invocation from the decision logic into two
modules. Rejected — the module is small and the run, compare, and decide steps
are tightly coupled in a single pass.

### D2: Runner extension — optional `classify_prompt` parameter
`BaselineRunner.__init__` currently hardcodes
`self._classify_prompt = CLASSIFICATION_PROMPT_V0`. The comparator adds an
optional `classify_prompt: Optional[PromptArtifact] = None` parameter to
`BaselineRunner.__init__`; when provided, it overrides the default. This is a
minimal, backward-compatible change: existing callers are unaffected. The
runner's `self._prompt_version` is set from `classify_prompt.version` when
provided, so artifact filenames carry the new version.

**Alternative**: subclass `BaselineRunner`. Rejected — a two-line constructor
change is simpler and avoids an inheritance layer.

**Alternative**: monkey-patch the prompt after construction. Rejected — fragile
and invisible.

### D3: Building a `PromptArtifact` from the composer's dict
`load_prompt_version` returns a plain dict (`version`, `text`, `hash`, `rules`,
...). The runner needs a `PromptArtifact` to access `.text` and `.version`. The
comparator builds a `PromptArtifact(version=..., layers=None, text=..., sha256=...)`
directly. `layers` is `None` because the composer's artifact stores rendered
`text`, not the layer structure — the runner only uses `.text` and `.version`,
so `layers=None` is safe (matching how `EXTRACTION_PROMPT` is constructed).

**Alternative**: reconstruct `PromptLayer` from the text. Rejected — fragile
parsing of rendered text; the runner does not need layers.

### D4: Reuse `metrics.compare_versions` for comparison and changed decisions
`compare_versions` already loads two results artifacts, computes metrics for
both, returns deltas, and lists changed ids with direction. The comparator reuses
it directly, then extracts the specific fields needed for the decision artifact
(`metrics_new`, `metrics_active`, `diff_accuracy`, `diff_macro_f1`,
`diff_minority_f1`, changed list). This avoids duplicating the comparison logic.

**Alternative**: re-implement the comparison in the comparator. Rejected —
`compare_versions` is tested and already produces the exact output needed.

### D5: Decision logic — macro-F1 primary, minority-F1 tie-breaker
`decide(metrics_new, metrics_active, config)` extracts the configured
`decision_metric` (default `macro_f1`) and `tie_breaker_metric` (default
`minority_f1`) from both metrics dicts. Accept when `new >= active` on the
primary metric. When equal, accept only if `new > active` on the tie-breaker.
Otherwise roll back. The reason string records which metric decided
(e.g. "macro_f1 improved", "macro_f1 tied, minority_f1 improved",
"macro_f1 degraded").

**Alternative**: require strict improvement on the primary metric. Rejected —
the spec requires `>=` so that a tie on macro-F1 with minority-F1 improvement is
accepted.

### D6: Rollback counter — persisted in the decision artifact
The counter is an integer that persists across calls within a cycle. The
comparator accepts the current `rollback_count` as input (from the orchestrator
or CLI state), increments it on rollback, resets to zero on acceptance. The
counter is stored in the decision artifact so the orchestrator can read it
without separate state. When the counter reaches `max_consecutive_rollbacks`,
the comparator returns a stop signal with the reason
"max_consecutive_rollbacks reached". The comparator itself does not maintain
long-term state — the orchestrator (out of scope) threads the counter across
calls.

**Alternative**: persist the counter in a separate state file. Rejected — the
decision artifact already carries it; a separate file adds coupling without
benefit.

### D7: Candidate queue — list of artifact paths
The comparator accepts a `candidate_queue` (list of candidate artifact paths,
e.g. `rule_candidates_*.json`). On rollback, it pops the next path and returns
it for composition. When the queue is empty, it returns a stop signal with
reason "candidate queue exhausted". The comparator does not own the queue — the
orchestrator or CLI passes it in. The rejected version's `source_candidates`
(rule cluster ids) are NOT carried forward; the next candidate is a fresh
artifact from the queue.

**Alternative**: the comparator generates new candidates by calling the
selector. Rejected — candidate selection is out of scope (non-goal).

### D8: Decision artifact format — single JSON file
`decision_{run_id}_v{new_version}_vs_v{active_version}_{timestamp}.json` with
top-level keys: `decision` (`"accept"` or `"rollback"`), `new_version`,
`active_version`, `metrics_new` (full metrics dict for the new version on dev),
`metrics_active` (full metrics dict for the active version on dev),
`diff_accuracy`, `diff_macro_f1`, `diff_minority_f1`, `rollback_count`,
`reason` (human-readable string), `changed_decisions` (list of
`{id, direction}`), `run_id`, `timestamp`, `created_at`, and `metadata`
(config snapshot, artifact paths).

**Alternative**: store only the deltas, not the full metrics. Rejected — the
full metrics make the artifact self-contained for offline analysis without
reloading the metrics artifacts.

### D9: Holdout metrics — logged, not in the decision artifact
When holdout results artifacts are available for both versions, the comparator
logs the holdout deltas at INFO level but does not include them in the decision
artifact or the decision logic. This matches the spec requirement and the
existing `stage1-metrics` convention (holdout metrics are stored separately).

**Alternative**: include holdout metrics in the decision artifact. Deferred —
listed as an open question in the spec; the default is to log separately.

### D10: Logging — `logging.info` matching existing components
Log `new_version`, `active_version`, `metrics_new` (accuracy, macro-F1,
minority-F1), `metrics_active` (same), `diff_accuracy`, `diff_macro_f1`,
`diff_minority_f1`, `rollback_count`, and `decision` at INFO level, matching
the logging style of `prompt_composer.py` and `rule_candidate_selector.py`.
Also print a short CLI summary table.

## Risks / Trade-offs

- **[Runner change required]** → D2 adds an optional parameter to
  `BaselineRunner.__init__`. Mitigation: the parameter defaults to `None`,
  preserving existing behavior; the change is backward-compatible.
- **[Decision metric sensitivity]** → Macro-F1 may not capture class-imbalance
  nuances perfectly. Mitigation: minority-class F1 is the tie-breaker, and both
  metrics are configurable via `decision_metric` / `tie_breaker_metric`.
- **[No statistical significance testing]** → A small improvement could be noise.
  Mitigation: deferred to a future change (non-goal); the decision artifact
  records full metrics for offline analysis.
- **[Rollback counter is stateless across calls]** → The comparator does not
  persist the counter between invocations; the orchestrator must thread it.
  Mitigation: the counter is recorded in every decision artifact, so the
  orchestrator can recover it by reading the last artifact.
- **[Candidate queue is external]** → The comparator does not own or persist the
  queue. Mitigation: the orchestrator manages the queue; the comparator's
  contract is to return the next path or a stop signal.

## Open Questions

- **Decision metric**: macro-F1 as the primary metric — is this right for a binary
  classification task with potential class imbalance, or should weighted-F1 be
  used? Deferred — configurable via `decision_metric`; default stays macro-F1.
- **Tie-breaker**: when macro-F1 is equal, is minority-class F1 the right
  tie-breaker, or should accuracy be used? Deferred — configurable via
  `tie_breaker_metric`; default stays minority-F1.
- **Rollback counter reset**: should the counter reset on acceptance only, or
  also when the queue is refreshed? Deferred — current design resets on
  acceptance only; revisit if queue refresh becomes common.
- **Changed decisions list scope**: should the list include all examples where
  the prediction changed, or only those where at least one version was incorrect?
  Deferred — current design includes all changed predictions (matching
  `metrics.compare_versions`); filtering to incorrect-only can be added later.
- **Queue exhaustion behavior**: when the queue is exhausted and a rollback
  occurs, should the cycle stop or should the composer generate new candidates?
  Deferred — current design stops; revisited when the orchestrator is built.
- **Holdout metrics in the artifact**: should holdout metrics be included in the
  decision artifact or logged separately? Deferred — current design logs
  separately; the artifact can be extended later.
