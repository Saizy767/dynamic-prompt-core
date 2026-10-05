# Design

## Context

The project has a linear optimization pipeline: `dataset-and-prompt` →
`stage1-baseline-runner` → `stage1-metrics` → `stage1-thesis-analyzer` →
`stage2-rule-candidate-selector`. The selector emits a
`rule_candidates_{run_id}_{prompt_version}_{timestamp}.json` artifact whose
`candidates` list carries `cluster_id`, `rank`, `precision`, `frequency`,
`positive_hits`, `negative_hits`, and `representative_theses` (each
`{text_norm, text_raw, frequency}`).

Prompts live in `prompts.py` as `PromptArtifact` dataclasses built from a
`PromptLayer(role, task, rules, output_contract, fallback)`. The baseline
`CLASSIFICATION_PROMPT_V0` (version `classify-v0`) is the only version today.
`render(layers)` joins the five layers in the fixed order
`role → task → rules → output contract → fallback`. The composer is not tied to
any specific upstream producer — it accepts any candidate artifact with the right
structure and any `PromptArtifact` with non-`None` layers.

Each existing component is a standalone Python module with a dataclass config
loaded from a TOML section, artifact load/validate helpers, and an argparse CLI
(`runner.py`, `metrics.py`, `thesis_analyzer.py`, `rule_candidate_selector.py`).
The composer follows the same pattern. It additionally uses `AsyncTask` for the
LLM formulation call and `sentence-transformers` (already a dependency via
`thesis_analyzer.py`) for the distortion-check embeddings.

## Goals / Non-Goals

**Goals:**
- Turn a candidate artifact plus the active prompt version into a new, validated
  prompt version with full lineage.
- Guarantee base layers are carried over unchanged.
- Reject rules that drift semantically from their source cluster.
- Produce a reloadable artifact so downstream stages do not re-run composition.

**Non-Goals:**
- Evaluating the new prompt version on the dev/holdout split (next stage).
- Optimizing the formulation prompt (it is fixed).
- Merging or deduplicating rules across versions.
- Interactive rule editing or human-in-the-loop review.

## Decisions

### D1: Single-module component (`prompt_composer.py`)
Mirrors `rule_candidate_selector.py` / `thesis_analyzer.py`: dataclass
`PromptComposerConfig.from_config`, `load_candidates(path)` with schema validation,
`load_base_layers(prompt_artifact)`, `formulate_rule(candidate, task, config)`
core LLM call, `check_distortion(rule, centroid, config)` embedding check,
`compose(candidates, base_layers, config)` orchestration, `write_prompt_version(...)`
/ `load_prompt_version(...)`, argparse CLI.

**Alternative**: split formulation logic from assembly logic into two modules.
Rejected — the module is small and the formulation, distortion check, and
assembly are tightly coupled in a single pass over the candidates.

### D2: Candidate schema validation — required fields
On load, validate that every candidate has `cluster_id`,
`representative_theses`, `precision`, `frequency`, `positive_hits`,
`negative_hits`, and `rank`. Raise an error naming the `cluster_id` and the
first missing field, matching the error style of
`rule_candidate_selector._validate_cluster`.

**Alternative**: validate via JSON Schema. Rejected — adds a dependency for a
flat field check that is clearer inline.

### D3: Base layers from `PromptArtifact.layers`
The composer reads `role`, `task`, `output_contract`, and `fallback` from the
active `PromptArtifact.layers` (a `PromptLayer`). When `layers` is `None` (as
with the extraction prompt) or any of the four base layers is an empty string,
composition fails with an error naming the missing layer. The `rules` list from
the active version is ignored — it is replaced entirely by the accepted rules.

**Alternative**: store base layers in a separate config file. Rejected —
`prompts.py` is the single source of truth for prompt content; duplicating it
invites drift.

### D4: Rule formulation via `AsyncTask` with a fixed prompt
A fixed system prompt (module constant `_FORMULATION_PROMPT`) instructs the LLM
to produce a single-sentence classification rule from the candidate's
representative theses, precision, and frequency. The LLM call uses a
`RuleFormulation` Pydantic schema (`{rule: str}`) added to `schemas.py` for
structured output. The call is synchronous from the CLI's perspective: the CLI
runs `asyncio.run` to drive the `AsyncTask` call, formulating rules sequentially
(formulation is not latency-critical — at most `max_rules` calls).

**Alternative**: batch formulation with `classify_many`. Rejected — the
formulation prompt differs per candidate (theses/precision/frequency are
injected into the user message), and the count is small (≤ `max_rules`).

### D5: Rule length enforcement — truncate, then log
When the LLM returns a rule longer than `max_rule_words` words, the composer
truncates it to the first `max_rule_words` words and logs a warning. Truncation
is preferred over rejection because the LLM's first `max_rule_words` words
usually carry the core condition; a hard reject would waste the formulation
call. The truncated rule still goes through the distortion check.

**Alternative**: reject oversize rules outright. Rejected — wastes the
formulation call and drops candidates that are often fixable by truncation.

### D6: Distortion check — cosine similarity to cluster centroid
The composer computes the embedding of the formulated rule (via the same
`sentence-transformers` model used by `thesis_analyzer.py`, configurable) and
compares it to the source cluster centroid with cosine similarity (reusing
`thesis_analyzer.cosine_similarity`). The centroid is read from the thesis-bank
artifact that produced the candidates; the candidate artifact does not carry it,
so the composer accepts an optional `--thesis-artifact` path to load centroids.
When the centroid is unavailable, the distortion check is skipped with a warning
(this keeps the composer usable when the thesis artifact is not at hand, at the
cost of losing the safety net).

When similarity is below `distortion_threshold` and `allow_reformulation` is
true, the composer makes one reformulation attempt (a second LLM call with a
"reformulate more faithfully" instruction appended). If the second attempt also
fails, the rule is rejected and logged.

**Alternative**: store the centroid in the candidate artifact. Rejected — the
candidate artifact is owned by the selector and changing its format is out of
scope; loading the centroid from the thesis artifact keeps this component
self-contained.

**Alternative**: skip the distortion check when the centroid is missing and
accept the rule unconditionally. Chosen for the no-centroid path, but logged at
WARNING so the operator knows the safety net was bypassed.

### D7: Token counting via the LLM tokenizer
`max_prompt_tokens` is checked against the token count from the `AsyncTask`
tokenizer (`AutoTokenizer` already loaded by `AsyncTask`). The composer reuses
the `AsyncTask` instance's tokenizer rather than loading its own. When the
tokenizer is unavailable, the check falls back to a word-count approximation
and logs a warning.

**Alternative**: a separate tokenizer instance. Rejected — `AsyncTask` already
loads one; reusing it avoids a second model load.

### D8: Artifact format — single JSON file
`prompt_v{version}_{run_id}_{timestamp}.json` with top-level keys `version`
(e.g. `classify-v1`), `text` (full rendered prompt), `hash` (sha256 of `text`,
first 16 hex chars matching `PromptArtifact.sha256` style), `rules` (list of
`{cluster_id, text}`), `source_candidates` (list of `cluster_id` values),
`base_version` (the active version the new one derives from), `created_at`
(ISO timestamp), and `metadata` (run_id, config snapshot, counters).

**Alternative**: store the full `PromptLayer` in the artifact. Rejected — the
`text` field is the renderable prompt; the layers are reconstructable from
`text` plus the rules list, and storing them duplicates content.

### D9: Version numbering — increment the active version's suffix
The new version is derived from the active version by incrementing the trailing
integer (e.g. `classify-v0` → `classify-v1`, `classify-v1` → `classify-v2`).
When the active version does not match the `classify-vN` pattern, the new
version is `{active_version}-next`.

**Alternative**: explicit version via CLI flag. Rejected for the default path —
auto-increment keeps the CLI simple; a `--version` override can be added later if
needed.

### D10: Composition counters logged via `logging.info`
Log `candidates_in`, `rules_formulated`, `rejected_distortion`,
`rejected_limits`, `final_rules_count` at INFO level, matching the logging style
of `rule_candidate_selector.py`. Also print a short CLI summary table.

## Risks / Trade-offs

- **[LLM formulation nondeterminism]** → At `temperature=0` the model is mostly
  deterministic, but different backends may still vary. Mitigation: the
  distortion check rejects rules that drift too far from the cluster centroid,
  and the formulation prompt is fixed.
- **[Distortion check skipped when centroid unavailable]** → Without the thesis
  artifact, rules are accepted without the semantic safety net. Mitigation: the
  CLI accepts `--thesis-artifact` and logs a WARNING when the check is skipped.
- **[Truncation may alter rule meaning]** → Truncating to `max_rule_words` could
  cut a condition. Mitigation: the truncated rule still passes the distortion
  check, which catches meaning drift; the truncation is logged.
- **[Token-count fallback to word count]** → When the tokenizer is unavailable,
  `max_prompt_tokens` is checked against word count, which underestimates tokens
  for multilingual text. Mitigation: the fallback is logged and the tokenizer is
  normally available via `AsyncTask`.
- **[Defaults tuned for small prompts]** → `max_prompt_tokens=300` and
  `max_rules=5` suit the current `classify-v0` scale (base layers alone are ~155
  tokens). Mitigation: all parameters are config-overridable.

## Open Questions

- **Should the composer deduplicate new rules against the active prompt's
  existing rules?** The candidate selector already excludes `in_prompt` clusters,
  so duplicates are unlikely. Deferred — add only if the formulation stage
  produces near-duplicate rules in practice.
- **Should `max_prompt_tokens` account for the user message or only the system
  prompt?** Currently only the system prompt. Deferred — the runner's
  `truncate_tokens` handles the user side separately.
- **Reformulation prompt wording**: the "reformulate more faithfully" instruction
  is a module constant. Deferred — tune after the first run if the second
  attempt rarely passes.
