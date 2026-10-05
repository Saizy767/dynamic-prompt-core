# Proposal

## Why

The pipeline has all the per-step components — baseline runner, metrics, thesis
analyzer, rule-candidate selector, prompt composer, and version comparator — but
nothing yet drives them as a repeatable optimization loop. Each stage is run
manually and the accept/rollback decision is not threaded across rounds, so the
system cannot improve a prompt version unattended, cannot recover from a
rejected candidate by trying the next one, and cannot produce a coherent
end-of-run summary. This change introduces a top-level orchestrator that runs
the full cycle for a configured number of rounds, maintains the round counter,
candidate queue, and rollback state, and produces per-round reports plus a final
summary.

## What Changes

- Add a new component (`cycle_orchestrator.py`) that loads all cycle
  configuration: `max_rounds` (default 5), `max_consecutive_rollbacks`
  (default 2), `dev_split`, `holdout_split`, `prompt_store_path`,
  `thesis_bank_path`, `candidate_queue_path`, and paths for intermediate
  artifacts. Fail with an error naming the missing key when required
  configuration is absent.
- Load the initial state: the active prompt version (v0 baseline), the prepared
  dataset artifact (dev and holdout splits), the fixed extraction prompt, and
  the initial (empty) thesis bank. Fail with an error naming the missing
  artifact when the active prompt or dataset is absent.
- Execute each round as a fixed nine-step sequence: run the active prompt on dev
  via `stage1-baseline-runner`, compute metrics via `stage1-metrics`, update the
  thesis bank and clusters via `stage1-thesis-analyzer`, select rule candidates
  via `stage2-rule-candidate-selector`, compose a new prompt version via
  `stage3-prompt-composer`, run the new version on dev, delegate the
  accept-or-rollback decision to `stage2-version-comparator`, update the active
  version or return the next candidate on rollback, and write the per-round
  report.
- Maintain a round counter starting at 0, incremented on each successfully
  completed round; stop at `max_rounds` with reason `max_rounds_reached`.
- Maintain a candidate queue populated by the selector at the start of each
  round, consumed in order; on rollback, return the next candidate to the
  composer; stop with reason `candidate_queue_exhausted` when the queue is empty.
- Track consecutive rollbacks via the comparator; stop with reason
  `max_rollbacks_reached` when the counter reaches `max_consecutive_rollbacks`;
  reset the counter on acceptance.
- On acceptance, mark the new version active in the prompt store (archiving the
  previous active version) and update the thesis bank's `in_prompt` flags to
  reflect the new rule set.
- Produce a per-round report containing round number, active version at round
  start, new version composed, decision, dev metrics for both versions, holdout
  metrics for both versions (logged only), rollback counter, changed decisions,
  and next action.
- Produce a final summary on stop containing total rounds, stop reason, final
  active version, final dev/holdout metrics, accepted-version history,
  rollback history, and all changed decisions across rounds. Stop reasons are
  one of: `max_rounds_reached`, `max_rollbacks_reached`,
  `candidate_queue_exhausted`, `unrecoverable_error`.
- Dump the full cycle state (round counter, active version, thesis bank,
  clusters, candidate queue, rollback counter, latest report path) to a JSON
  file named with `run_id` and timestamp after each round, and support resuming
  from a dumped state by skipping completed rounds.
- Log all cycle events (round start/end, decision, accept/rollback, stop
  reason, errors) to an append-only, per-round-queryable log.
- Add a `[cycle_orchestrator]` section to `config.toml` for `max_rounds`,
  `max_consecutive_rollbacks`, `dev_split`, `holdout_split`,
  `stop_on_first_error` (default false), `dump_state_after_each_round`
  (default true), and `report_format` (default json).

## Capabilities

### New Capabilities
- `stage3-cycle-orchestrator`: Drives the full optimization loop across a
  configured number of rounds, coordinating the baseline runner, metrics,
  thesis analyzer, rule-candidate selector, prompt composer, and version
  comparator. Maintains round counter, candidate queue, and rollback state,
  produces per-round reports and a final summary, dumps reloadable state after
  each round, and supports resuming an interrupted cycle.

### Modified Capabilities
<!-- None — this change introduces a new top-level component without altering
     existing specs. The orchestrator calls existing components via their
     public APIs; no per-component requirement changes. -->

## Impact

- **New code**: `cycle_orchestrator.py` (component + CLI), mirroring the
  structure of `version_comparator.py` and `prompt_composer.py` (dataclass
  config from TOML, artifact load/validate, CLI via argparse). Reuses
  `BaselineRunner` from `runner.py`, `compute_metrics` / `compare_versions` from
  `metrics.py`, `PromptArtifact` from `prompts.py`, and the public functions of
  `thesis_analyzer.py`, `rule_candidate_selector.py`, `prompt_composer.py`, and
  `version_comparator.py`.
- **Config**: new `[cycle_orchestrator]` section in `config.toml`.
- **Dependencies**: no new dependencies. Reuses `aiohttp` (via `BaselineRunner`),
  `AsyncTask`, and the existing stage modules.
- **Artifacts**: new `report_*_*_*.json` (per-round), `summary_*_*_*.json`
  (final), `state_*_*_*.json` (cycle state dump), and `cycle_log_*_*.jsonl`
  (append-only event log) files written to `data/results/`.
- **Upstream**: consumes the active `PromptArtifact` from `prompts.py`, the
  prepared dataset artifact, and the `[runner]` / `[version_comparator]` /
  `[prompt_composer]` config sections.
- **Downstream**: none — this is the top-level entry point of the system.
