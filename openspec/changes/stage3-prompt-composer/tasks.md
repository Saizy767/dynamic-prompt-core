# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[prompt_composer]` section to `config.toml` with keys
  `max_rules`, `max_prompt_tokens`, `max_rule_words`, `distortion_threshold`,
  `allow_reformulation`, `embedding_model`, `formulation_max_tokens`,
  `formulation_temperature`, `output_dir`, and inline comments documenting each
  default. Verify the file parses with `tomllib.load` and the section round-trips.
- [x] 1.2 Create `prompt_composer.py` with module docstring,
  `from __future__ import annotations`, a `PromptComposerConfig` dataclass with
  `from_config()` reading `[prompt_composer]` from `config.toml` (mirroring
  `RuleCandidateSelectorConfig`), and default constants (`DEFAULT_MAX_RULES=5`,
  `DEFAULT_MAX_PROMPT_TOKENS=150`, `DEFAULT_MAX_RULE_WORDS=15`,
  `DEFAULT_DISTORTION_THRESHOLD=0.75`, `DEFAULT_ALLOW_REFORMULATION=True`,
  `DEFAULT_EMBEDDING_MODEL="paraphrase-multilingual-MiniLM-L12-v2"`,
  `DEFAULT_OUTPUT_DIR="data/results"`). Verify
  `PromptComposerConfig.from_config()` returns the configured values and falls
  back to defaults for missing keys.
- [x] 1.3 Add a `RuleFormulation(BaseModel)` schema with a single `rule: str`
  field to `schemas.py` (design D4). Verify `RuleFormulation(rule="...")` validates
  and that an empty string is accepted (length is enforced by the composer, not
  the schema).

## 2. Artifact loading and validation

- [x] 2.1 Implement `load_candidates(path)` that reads the rule-candidate JSON
  artifact and validates that every candidate has `cluster_id`,
  `representative_theses`, `precision`, `frequency`, `positive_hits`,
  `negative_hits`, `rank` (design D2). Raise `PromptComposerError` naming the
  `cluster_id` and the first missing field on failure. Verify it loads a valid
  artifact and raises on a candidate missing `representative_theses`.
- [x] 2.2 Implement `load_base_layers(prompt_artifact)` that extracts `role`,
  `task`, `output_contract`, `fallback` from a `PromptArtifact.layers`
  (design D3). Raise `PromptComposerError` naming the missing layer when
  `layers` is `None` or any base layer is an empty string. Verify it returns the
  four layers from `CLASSIFICATION_PROMPT_V0` and raises when `layers` is `None`.
- [x] 2.3 Implement `load_centroids(thesis_artifact_path)` that loads a
  thesis-bank dump and returns a `{cluster_id: centroid_ndarray}` map (design D6).
  Return an empty map when the path is `None`. Verify it loads centroids from a
  real thesis dump and returns an empty map for `None`.

## 3. Rule formulation via LLM

- [x] 3.1 Define the fixed `_FORMULATION_PROMPT` system prompt constant
  (design D4) instructing the LLM to produce a single-sentence classification
  rule from the candidate's representative theses, precision, and frequency.
  Verify the constant is a non-empty string and does not reference config values
  (it is fixed, not optimized).
- [x] 3.2 Implement `formulate_rule(candidate, task, config, async_task)` that
  builds the user message from the candidate's `representative_theses`,
  `precision`, `frequency`, calls `AsyncTask.analyze` with
  `model=RuleFormulation` and the fixed formulation prompt, and returns the
  rule string (design D4). Verify the user message contains the theses and
  precision, and that a mocked `AsyncTask` returning `{"rule": "..."}` yields
  the rule string.
- [x] 3.3 Implement `enforce_rule_length(rule, max_words)` that truncates a rule
  to `max_words` words and returns `(truncated_rule, was_truncated)` (design D5).
  Verify a 10-word rule with `max_words=15` is returned unchanged, and a 20-word
  rule is truncated to 15 words with `was_truncated=True`.
- [x] 3.4 Implement `reformulate_rule(candidate, rule, config, async_task)` that
  appends a "reformulate more faithfully" instruction to the formulation prompt
  and makes a second LLM call (design D6). Verify the reformulation prompt
  contains the original rule and the faithfulness instruction, and that a mocked
  `AsyncTask` returns the new rule string.

## 4. Semantic distortion check

- [x] 4.1 Implement `compute_rule_embedding(rule, model_name)` reusing the
  lazy-loaded `sentence-transformers` model from `thesis_analyzer.py` (design D6).
  Verify it returns a float32 numpy array of the expected dimensionality for a
  non-empty rule string.
- [x] 4.2 Implement `check_distortion(rule, centroid, config, model_name)` that
  computes the cosine similarity between the rule embedding and the cluster
  centroid using `thesis_analyzer.cosine_similarity` (design D6). Return
  `(passed, similarity)`. When `centroid` is `None`, return `(True, None)` and
  log a WARNING that the safety net was bypassed. Verify: similarity at or above
  `distortion_threshold` returns `(True, sim)`; similarity below returns
  `(False, sim)`; `centroid=None` returns `(True, None)`.

## 5. Prompt assembly and validation

- [x] 5.1 Implement `assemble_prompt(base_layers, accepted_rules, base_version)`
  that builds a new `PromptLayer` from the base layers plus the accepted rules,
  renders it via `prompts.render`, and returns a `PromptArtifact` with the
  incremented version (design D9). Verify the layer order is
  `role, task, rules, output contract, fallback` and the base layers are
  identical to the input.
- [x] 5.2 Implement `increment_version(base_version)` that increments the
  trailing integer of a `classify-vN` version or appends `-next` when the
  pattern does not match (design D9). Verify `classify-v0` → `classify-v1`,
  `classify-v1` → `classify-v2`, and `custom` → `custom-next`.
- [x] 5.3 Implement `validate_limits(prompt_artifact, config, tokenizer)` that
  checks: rule count ≤ `max_rules`, prompt tokens ≤ `max_prompt_tokens`,
  non-empty `output_contract` and `fallback` (design D7). Raise
  `PromptComposerError` with a message naming the violated limit
  (e.g. "too many rules: 7 > 5", "prompt too long: 160 > 150",
  "fallback is empty"). Verify: 5 rules / 140 tokens passes; 7 rules with
  `max_rules=5` fails; 160 tokens with `max_prompt_tokens=150` fails; empty
  fallback fails. Verify the token-count fallback to word count logs a warning
  when the tokenizer is `None`.

## 6. Composition orchestration and artifact persistence

- [x] 6.1 Implement `compose(candidates, base_layers, centroids, config,
  async_task)` that chains formulation, length enforcement, distortion check,
  optional reformulation, and assembly over all candidates (design D1). Return
  `(prompt_artifact, rules_with_lineage, counters)` where each rule carries
  `cluster_id` and `text`, and `counters` has `candidates_in`,
  `rules_formulated`, `rejected_distortion`, `rejected_limits`,
  `final_rules_count`. Verify with mocked formulation returning 4 valid rules:
  counters show `candidates_in=4, rules_formulated=4, rejected_distortion=0,
  rejected_limits=0, final_rules_count=4` and each rule carries its `cluster_id`.
- [x] 6.2 Verify `compose` rejects a rule below `distortion_threshold` when
  `allow_reformulation=false`: with a mocked embedding returning similarity 0.5
  and `distortion_threshold=0.75`, the rule is rejected and
  `rejected_distortion` is incremented. Verify a reformulation attempt is made
  when `allow_reformulation=true` and the second attempt passes.
- [x] 6.3 Implement `write_prompt_version(prompt_artifact, rules_with_lineage,
  source_candidates, base_version, run_id, counters, config, output_dir)` that
  serializes to `prompt_v{version}_{run_id}_{timestamp}.json` with top-level keys
  `version`, `text`, `hash` (sha256 first 16 hex), `rules` (list of
  `{cluster_id, text}`), `source_candidates`, `base_version`, `created_at`,
  `metadata` (design D8). Verify the file is valid JSON, the filename matches the
  pattern, the hash matches `sha256(text)[:16]`, and `source_candidates` lists
  the `cluster_id` values used.
- [x] 6.4 Implement `load_prompt_version(path)` that restores the prompt version
  from a `prompt_v*_*_*.json` artifact without re-running composition (design D8).
  Verify that loading a file written by `write_prompt_version` reproduces the
  same `version`, `text`, `hash`, `rules`, and `source_candidates`.

## 7. CLI and integration

- [x] 7.1 Implement the argparse CLI (`python prompt_composer.py --artifact
  <candidates.json> [--thesis-artifact <thesis.json>] [--config config.toml
  --endpoint <url>]`) that loads config, loads the candidate artifact, loads the
  active `CLASSIFICATION_PROMPT_V0` base layers, loads centroids from the thesis
  artifact when provided, runs `compose` via `asyncio.run`, validates limits,
  writes the prompt-version artifact, logs the five composition counters at INFO
  level, and prints a summary table (design D10). Verify the CLI runs against an
  existing candidate artifact and prints the summary without errors.
- [x] 7.2 Run the composer end-to-end on a real candidate artifact with a running
  LLM server, then reload the written prompt-version artifact with
  `load_prompt_version` and confirm the reloaded `version`, `text`, `hash`, and
  `rules` match the in-memory result. Verify the reload path does not re-run
  composition or call the LLM. Verify the lineage chain: each reloaded rule's
  `cluster_id` appears in `source_candidates` and maps to a cluster in the thesis
  artifact.
