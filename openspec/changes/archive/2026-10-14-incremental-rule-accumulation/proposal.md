# Proposal

## Why

The optimization cycle composes and tests one rule at a time, replacing the
entire rule set on each attempt. A candidate that improves metrics is accepted
and the round ends immediately — no further candidates are tried, so the cycle
needs many rounds to accumulate multiple rules. Additionally, the holdout
evaluation runs full thesis extraction (an LLM call per example) even though
holdout results are only used for metrics, not for rule discovery. This wastes
inference budget and slows every round.

## What Changes

- **Incremental rule accumulation within a round**: instead of composing one
  rule and stopping on accept, the cycle greedily accumulates rules from
  successive candidates. Each candidate's rule is added on top of the
  previously accepted rules; the new prompt is run on dev and compared against
  the current best. When the metric does not degrade, the rule is kept and the
  next candidate is tried. When the metric degrades, accumulation stops and
  the best version is preserved.
- **Carry-over of existing rules**: `compose()` accepts an `existing_rules`
  parameter so the newly formulated rule is appended to the rules already in
  the active prompt, not replacing them.
- **max_rules enforcement in the cycle**: accumulation stops when the number
  of rules reaches `prompt_composer.max_rules`, preventing limit violations.
- **Skip thesis extraction on holdout**: `BaselineRunner` gains a
  `skip_extraction` flag. When set, `_process_example` skips
  `extract_theses_detailed` and only runs candidate-scoring classification.
  `run_holdout` in the cycle passes `skip_extraction=True`, so holdout runs no
  longer make thesis-extraction LLM calls.
- **Rollback semantics**: a round in which no candidate was accepted increments
  the rollback counter. A round with at least one acceptance resets it.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `stage3-cycle-orchestrator`: the per-round step sequence changes from
  "compose one rule → run → decide → break on accept / next candidate on
  rollback" to greedy incremental accumulation. The candidate queue is
  consumed until degradation or exhaustion, not until the first accept.
- `stage1-baseline-runner`: `BaselineRunner` gains a `skip_extraction` flag
  that skips thesis extraction and only runs classification scoring.

## Impact

- `application/use_cases/run_cycle/steps.py` — `run_round` while-loop
  restructured; `compose_new_version` passes existing rules; `run_holdout`
  passes `skip_extraction=True`.
- `application/use_cases/compose_prompt/composer.py` — `compose()` gains
  `existing_rules` parameter; rules are prepended before `assemble_prompt`.
- `application/use_cases/run_baseline/runner.py` — `BaselineRunner.__init__`
  gains `skip_extraction` param; `_process_example` conditionally skips
  extraction.
- `tests/unit/run_cycle/test_steps.py` — existing round test updated; new
  tests for accumulation, degradation stop, and no-acceptance rollback.
- `tests/unit/run_baseline/test_runner.py` — new test for `skip_extraction`.
- `openspec/specs/stage3-cycle-orchestrator/spec.md` — requirements for
  "Run a single round", "Candidate queue", and "Rollback state" updated;
  new requirement for holdout without thesis extraction.
- `openspec/specs/stage1-baseline-runner/spec.md` — requirement for
  `skip_extraction` mode.
