# Proposal

## Why

The `stage2-rule-candidate-selector` emits a ranked candidate list with
representative theses, but nothing yet turns those candidates into a new
classification prompt version. To close the optimization loop, we need a
repeatable composition step that formulates each candidate into a single-sentence
rule via an LLM, verifies the rule preserves the cluster's meaning via embedding
similarity, assembles a new prompt from the unchanged base layers plus the
accepted rules, validates the result against hard limits, and persists a
reloadable prompt-version artifact with lineage back to the source clusters.

## What Changes

- Add a new component (`prompt_composer.py`) that loads a rule-candidate artifact
  (JSON with `candidates` carrying `cluster_id`, `representative_theses`,
  `precision`, `frequency`, `positive_hits`, `negative_hits`, `rank`) and
  schema-validates it. The component is not tied to any specific upstream
  producer — it accepts any artifact with the right structure.
- Load the base layers (`role`, `task`, `output_contract`, `fallback`) from the
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
  preserving the layer order `role`, `task`, `rules`, `output_contract`,
  `fallback`.
- Validate the assembled prompt against hard limits: `max_rules`,
  `max_prompt_tokens`, and non-empty `output_contract` / `fallback`. Fail with an
  error naming the violated limit.
- Persist a `prompt_v{version}_{run_id}_{timestamp}.json` artifact containing
  `version`, `text`, `hash`, `rules`, `source_candidates`, `base_version`, and
  `created_at`, reloadable without re-running composition.
- Record lineage: each rule carries the `cluster_id` it derives from, so the
  chain `version → rules → clusters → theses` is reconstructable from stored
  artifacts.
- Log composition counters: `candidates_in`, `rules_formulated`,
  `rejected_distortion`, `rejected_limits`, `final_rules_count`.
- Add a `[prompt_composer]` section to `config.toml` for `max_rules`,
  `max_prompt_tokens`, `max_rule_words`, `distortion_threshold`,
  `allow_reformulation`, `embedding_model`, `formulation_max_tokens`,
  `formulation_temperature`, and `output_dir`.

## Capabilities

### New Capabilities
- `stage3-prompt-composer`: Loads a rule-candidate artifact, formulates each
  candidate into a single-sentence rule via an LLM, verifies semantic fidelity
  against the source cluster centroid, assembles a new classification prompt
  version from the unchanged base layers plus accepted rules, validates hard
  limits, and persists a reloadable prompt-version artifact with full lineage.

### Modified Capabilities
<!-- None — this change introduces a new component without altering existing specs. -->

## Impact

- **New code**: `prompt_composer.py` (component + CLI), mirroring the structure of
  `rule_candidate_selector.py` and `thesis_analyzer.py` (dataclass config from
  TOML, artifact load/validate, CLI via argparse). Adds a `RuleFormulation` Pydantic
  schema to `schemas.py` for structured LLM output.
- **Config**: new `[prompt_composer]` section in `config.toml`.
- **Dependencies**: reuses `sentence-transformers` (already present via
  `thesis_analyzer.py`) for the distortion-check embeddings, `numpy` for cosine
  similarity, `pydantic` for the formulation schema, and `asyncTask.AsyncTask`
  for the LLM call. No new dependencies.
- **Artifacts**: new `prompt_v*_*_*.json` files written to `data/results/`.
- **Upstream**: consumes any `rule_candidates_*.json` artifact (e.g. from
  `stage2-rule-candidate-selector`) and the active `PromptArtifact` from
  `prompts.py`.
- **Downstream**: feeds the future prompt-version runner/evaluator that measures
  incremental F1 on the dev split (out of scope).
