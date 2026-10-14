# Design

## Context

The cycle orchestrator (`run_cycle/steps.py:run_round`) currently processes
candidates one at a time: compose a single rule, run the new prompt on dev,
decide accept/rollback, and break on the first accept. Rules do not accumulate
within a round — each composition replaces the entire rule set. The composer
(`compose_prompt/composer.py:compose`) receives a list of candidates but is
called with a single-element list from `compose_new_version`. The baseline
runner (`run_baseline/runner.py:BaselineRunner._process_example`) always runs
both thesis extraction and classification scoring, even on holdout where only
metrics are needed.

## Goals / Non-Goals

**Goals:**
- Accumulate multiple rules within a single round via greedy incremental
  addition, stopping on degradation.
- Carry over previously accepted rules when composing a new candidate's rule.
- Enforce `max_rules` during accumulation to prevent limit violations.
- Skip thesis extraction on holdout runs to avoid wasted LLM calls.

**Non-Goals:**
- Changing the candidate selection logic or ranking.
- Changing the thesis analyzer or clustering.
- Changing the decision metric or tie-breaker logic.
- Batch composition of all candidates at once (the user explicitly chose
  incremental add-and-test, not batch).
- Extracting rules from pure (1/1 or 0/1) clusters — deferred.

## Decisions

### Decision 1: Greedy incremental loop in `run_round`

Replace the current while-loop (steps.py:409-459) that breaks on first accept
with a loop that continues on accept and stops on degradation.

**Rationale**: The user wants "add cluster → run → compare → if good, add
next → if degraded, stop and save best." This is a greedy search that
accumulates rules monotonically.

**Alternative considered**: Compose all rules from all candidates at once,
then test once. Rejected — the user explicitly chose incremental add-and-test.

**Implementation**:
- Track `best_prompt` and `best_metrics` starting from the active version.
- Track `accumulated_source_candidates` across accepts for `in_prompt` flags.
- On accept (`decide` returns "accept"): update `best_prompt`,
  `best_metrics`; append to `accumulated_source_candidates`; continue.
- On degradation (`decide` returns "rollback"): break, keep `best_prompt`.
- After the loop: set `state.active_version = best_prompt`; update
  `in_prompt` flags for all `accumulated_source_candidates`; set rollback
  counter based on whether any acceptance occurred.

### Decision 2: `existing_rules` parameter in `compose()`

Add `existing_rules: list[str] | None = None` to `compose()`. When provided,
prepend them to `accepted_rules` before calling `assemble_prompt`.
`rules_with_lineage` contains only the newly formulated rule(s), so lineage
tracking is unaffected.

**Rationale**: Minimal API change. `assemble_prompt` already takes a
`list[str]` of rules, so prepending is a one-line change in `compose()`.

**Alternative considered**: Modify `assemble_prompt` to take
`existing_rules` and `new_rules` separately. Rejected — unnecessary
complexity; `assemble_prompt` just joins them into `PromptLayer.rules`.

### Decision 3: Extract existing rules in `compose_new_version`

`compose_new_version` (steps.py:165-200) currently calls
`load_base_layers(active_prompt)` which extracts only role, task,
output_contract, fallback — not rules. Add extraction of
`active_prompt.layers.rules` and pass them as `existing_rules` to `compose()`.

When `active_prompt.layers` is None (e.g., resumed from state dump where
layers are not serialized), default to an empty list — no rules to carry over.

### Decision 4: `max_rules` check in the accumulation loop

Before composing each candidate, check
`len(current_rules) >= pc_config.max_rules`. If true, stop accumulation.

**Rationale**: `validate_limits` is not called in the cycle (only in the
CLI). Adding the check in the loop prevents limit violations without
duplicating `validate_limits`.

### Decision 5: `skip_extraction` flag in `BaselineRunner`

Add `skip_extraction: bool = False` to `BaselineRunner.__init__`. In
`_process_example`, when true:
- Skip the `extract_theses_detailed` call.
- Set `theses_raw = []`, `theses_norm = []`, `extract_status = "skipped"`,
  `extract_latency_ms = 0.0`, `raw_extract = None`.
- Only run the classification scoring block.

`run_holdout` in steps.py passes `skip_extraction=True` when constructing the
runner.

**Rationale**: The holdout run only needs metrics. Thesis extraction is an
LLM call per example that produces data used only for rule discovery, which
happens exclusively on dev.

### Decision 6: Rollback counter semantics

- Round with at least one acceptance → reset rollback counter to 0.
- Round with zero acceptances → increment rollback counter by 1.

This preserves the existing `max_consecutive_rollbacks` stop condition: if
the cycle goes through multiple rounds without any rule being accepted, it
stops.

## Risks / Trade-offs

- **[Greedy suboptimality]** Greedy accumulation may miss a combination where
  an individually-degrading rule would help in combination with a later one.
  → Mitigation: the candidate selector ranks by precision, so earlier
  candidates are more likely to be beneficial. The trade-off is acceptable for
  the user's stated goal.

- **[Rules from previous rounds]** When a round starts, the active prompt may
  already have rules from previous rounds. `compose_new_version` extracts
  `active_prompt.layers.rules` as `existing_rules`. If `layers` is None
  (resumed state), existing rules are lost. → Mitigation: document that
  resumed cycles start fresh rule accumulation; the thesis bank's `in_prompt`
  flags still prevent re-selection of used clusters.

- **[Holdout result schema]** Holdout results with `skip_extraction=True` have
  `extract_status="skipped"` instead of "ok". Downstream consumers
  (thesis_analyzer) already filter on `extract_status == "ok"`, so skipped
  records are excluded from analysis. → No risk; this is the desired
  behavior.

- **[max_rules not enforced previously]** The cycle never enforced
  `max_rules`. Adding the check is a behavior change: rounds that would have
  exceeded the limit now stop earlier. → Mitigation: `max_rules=5` and the
  selector picks `top_n=5`, so the limit is rarely reached in practice.
