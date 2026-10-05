# Proposal

## Why

The `stage2-rule-candidate-selector` emits a ranked candidate list with
representative theses, but nothing yet turns those candidates into a new
classification prompt version. A first iteration (`stage3-prompt-composer`,
complete) delivered `prompt_composer.py` with a `max_prompt_tokens=300` default.
The planning artifacts need to be re-established fresh for a v2 pass that
re-grounds the spec while keeping the 300 default. To close the optimization loop
repeatably, we need a composition step that formulates each candidate into a
single-sentence rule via an LLM, verifies the rule preserves the cluster's meaning
via embedding similarity, assembles a new prompt from the unchanged base layers
plus the accepted rules, validates the result against hard limits, and persists a
reloadable prompt-version artifact with lineage back to the source clusters.

## What Changes

- Re-specify the prompt composer component (`prompt_composer.py`) as a v2
  capability: load a rule-candidate artifact (JSON with `candidates` carrying
  `cluster_id`, `representative_theses`, `precision`, `frequency`,
  `positive_hits`, `negative_hits`, `rank`) and schema-validate it. The component
  is not tied to any specific upstream producer — it accepts any artifact with the
  right structure.
- Load the base layers (`role`, `task`, `output contract`, `fallback`) from the
  currently active prompt version (`prompts.PromptArtifact`). The base layers are
  carried over unchanged; only the `rules` layer is replaced.
- Formulate one rule per candidate via an LLM call using a fixed formulation
  prompt (not optimized). The LLM input is the candidate's
  `representative_theses`, `precision`, and `frequency`; the output is a single
  sentence of at most `max_rule_words` words.
- Verify each formulated rule for semantic distortion by comparing the rule
  embedding against the source cluster centroid with cosine similarity. Rules
  below `distortion_threshold` are rejected, or get one reformulation attempt when
  `allow_reformulation` is true before rejection.
- Assemble a new prompt version from the base layers plus accepted rules,
  preserving the layer order `role`, `task`, `rules`, `output contract`,
  `fallback`.
- Validate the assembled prompt against hard limits: `max_rules`,
  `max_prompt_tokens`, and non-empty `output contract` / `fallback`. Fail with an
  error naming the violated limit.
- Persist a `prompt_v{version}_{run_id}_{timestamp}.json` artifact containing
  `version`, `text`, `hash`, `rules`, `source_candidates`, `base_version`, and
  `created_at`, reloadable without re-running composition.
- Record lineage: each rule carries the `cluster_id` it derives from, so the chain
  `version → rules → clusters → theses` is reconstructable from stored artifacts.
- Log composition counters: `candidates_in`, `rules_formulated`,
  `rejected_distortion`, `rejected_limits`, `final_rules_count`.
- Keep `max_prompt_tokens` default at 300 (matching v1), so the assembled system
  prompt budget stays config-overridable without a behavioral break.
- Add a `[prompt_composer]` section to `config.toml` for `max_rules`,
  `max_prompt_tokens`, `max_rule_words`, `distortion_threshold`,
  `allow_reformulation`, `embedding_model`, `formulation_max_tokens`,
  `formulation_temperature`, and `output_dir`.

## Capabilities

### New Capabilities
- `stage3-prompt-composer-v2`: Loads a rule-candidate artifact, formulates each
  candidate into a single-sentence rule via an LLM, verifies semantic fidelity
  against the source cluster centroid, assembles a new classification prompt
  version from the unchanged base layers plus accepted rules, validates hard
  limits (with `max_prompt_tokens=300` default), and persists a
  reloadable prompt-version artifact with full lineage.

### Modified Capabilities
<!-- None — this change introduces a new v2 capability. The v1 change
     stage3-prompt-composer is complete but not yet archived to main specs, so
     there is no existing main spec to modify. -->

## Impact

- **Existing code**: `prompt_composer.py` (component + CLI) already exists from
  the v1 change. v2 re-grounds the spec and keeps the
  `DEFAULT_MAX_PROMPT_TOKENS` constant at 300, along with the
  `[prompt_composer]` default in `config.toml`. The module mirrors the structure
  of `rule_candidate_selector.py` and `thesis_analyzer.py` (dataclass config from
  TOML, artifact load/validate, CLI via argparse) and uses a `RuleFormulation`
  Pydantic schema in `schemas.py` for structured LLM output.
- **Config**: `[prompt_composer]` section in `config.toml` — default
  `max_prompt_tokens` stays at 300.
- **Dependencies**: reuses `sentence-transformers` (already present via
  `thesis_analyzer.py`) for the distortion-check embeddings, `numpy` for cosine
  similarity, `pydantic` for the formulation schema, and `asyncTask.AsyncTask`
  for the LLM call. No new dependencies.
- **Artifacts**: `prompt_v*_*_*.json` files written to `data/results/`.
- **Upstream**: consumes any `rule_candidates_*.json` artifact (e.g. from
  `stage2-rule-candidate-selector`) and the active `PromptArtifact` from
  `prompts.py`.
- **Downstream**: feeds the prompt-version runner/evaluator that measures
  incremental F1 on the dev split (`stage2-version-comparator`, out of scope for
  this change).
