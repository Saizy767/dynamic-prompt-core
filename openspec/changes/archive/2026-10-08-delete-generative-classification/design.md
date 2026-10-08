# Design

## Context

The project has two parallel classification architectures:

1. **Candidate scoring** (current): `CandidateScorer.score(text, candidates)`
   → `Judgment[]` → `ClassificationPolicy.classify(judgments)` →
   `Classification`. Infrastructure provides `LLMLogitCandidateScorer` backed by
   `ScoringPromptBuilder`, batch tokenizer, `BatchedCausalLanguageModel`, and
   `LogitScorer`. This path is fully specified and tested.

2. **Generative classification** (legacy): `BaselineRunner` →
   `AsyncTask.classify_detailed(session, text, ClassificationResult, ...)`
   → `LLMClient` → structured JSON generation → `ClassificationResult(decision,
   confidence)`. This path uses `CLASSIFICATION_PROMPT_V0` which contains an
   `output_contract` requiring JSON generation.

The only production caller of `classify_detailed` is `BaselineRunner` in
`application/use_cases/run_baseline/runner.py:322`. `ClassificationResult` is
defined in `application/schemas/classification.py` and exported from
`application/schemas/__init__.py`. No tests reference `classify_detailed` or
`ClassificationResult`.

`LLMClient` (`infrastructure/llm/client.py`) provides both classification
methods (`classify`, `classify_detailed`, `classify_many`,
`classify_many_detailed`) and extraction methods (`extract_theses`,
`extract_theses_detailed`, `extract_theses_many`, `extract_theses_many_detailed`,
`analyze`, `analyze_raw`). The extraction methods serve thesis extraction, which
is not part of this change.

### ResultRow consumers

`ResultRow` (`runner.py:91-122`) is an internal application artifact serialized
to JSONL. Its consumers are:
- **Metrics module** (`application/services/metrics/metrics.py`): requires
  `predicted_decision` and `confidence` in `REQUIRED_FIELDS` (line 37).
  `predicted_decision` is used for accuracy, PR/F1, confusion matrix, prediction
  distribution, and version comparison. `confidence` is validated as present but
  not used in any computation.
- **`run_cycle/steps.py`**: calls `compute_metrics` which consumes
  `predicted_decision` indirectly.
- **Tests** (`tests/unit/run_cycle/test_steps.py`): synthetic result rows
  include `predicted_decision` and `confidence`.

No external API or system consumes the JSONL artifact. All consumers are
internal and will be migrated.

### Candidate source

There is no existing production mechanism that supplies `list[Candidate]` for
classification. `classify_input` is never called from production code. `Candidate`
is only constructed in tests. The `select_candidates` use case produces rule
candidates (thesis-cluster dicts for prompt composition), not classification
candidates. A candidate source must be specified for `BaselineRunner`.

### Scoring error boundary

The application-facing scoring error is `CandidateScoringError` in
`domain/errors/scoring.py`. The infrastructure-internal error is
`LLMScoringError` in `infrastructure/llm/scoring/errors.py`. The translation
happens at the port boundary in `LLMLogitCandidateScorer.score()`
(`candidate_scorer.py:92-97`), which catches `LLMScoringError` and re-raises as
`CandidateScoringError`.

### PromptLayer and CLASSIFICATION_PROMPT_V0

`PromptLayer` (`domain/prompts/base.py`) is a classification-specific prompt
structure with five required fields: `role`, `task`, `rules`,
`output_contract`, `fallback`. All fields are structurally required (no
defaults). `load_base_layers` in `composer.py` enforces that all four base
layers are non-empty strings.

`CLASSIFICATION_PROMPT_V0` (`domain/prompts/fixed.py:39-41`) is an **active
seed/base template** for the prompt versioning system. It seeds version 0 in
`prompt_store._seed_baseline()` and provides base layers for composition in
`compose_prompt` and `run_cycle`. It is not merely a historical artifact; new
prompt versions are composed from its base layers.

### Semantic criteria already migrated

The semantic classification criteria from `CLASSIFICATION_PROMPT_V0` are fully
migrated into `ScoringPromptBuilder._SEMANTIC_PREFIX`
(`infrastructure/llm/scoring/prompt_builder.py:19-35`). All four rules from the
legacy prompt are represented in candidate-evaluation language. The generative
protocol (`output_contract`, `fallback`) is deliberately excluded. This is
verified by `TestJudgmentPromptSemantics` in
`tests/unit/candidate_scoring/test_scoring_seams.py`.

## Goals / Non-Goals

**Goals:**
- Make candidate scoring the single supported classification mechanism.
- Remove `ClassificationResult`, `classify_detailed`, and all
  classification-specific structured-generation dependencies from production.
- Migrate `BaselineRunner` to use `CandidateScorer` → `ClassificationPolicy`
  for classification while preserving thesis extraction.
- Remove coexistence requirements from all specs.
- Prove the old path is gone via forbidden-reference tests.

**Non-Goals:**
- Change `CandidateScorer`, `Judgment`, `Classification`,
  `ClassificationPolicy`, or `ArgmaxClassificationPolicy` contracts.
- Change batch-scoring, tokenization, or logit-scoring semantics.
- Introduce score calibration or confidence estimation.
- Remove unrelated LLM generation functionality (thesis extraction, `analyze`,
  `analyze_raw`).
- Remove the prompt versioning system or its seed prompt artifact.
- Change `PromptLayer` structure or make its fields optional.
- Modify `CLASSIFICATION_PROMPT_V0` content.
- Change the model/provider abstraction beyond removing classification-specific
  methods.
- Introduce a new application architecture.

## Decisions

### Decision 1: Delete rather than deprecate

The generative classification path is deleted, not deprecated. No compatibility
wrapper, no runtime fallback, no `try: candidate_scoring except: generative`
pattern. This makes the architecture explicit and prevents two classification
mechanisms from silently diverging.

**Alternative considered**: Deprecate with a `DeprecationWarning` and keep the
old path for one release. Rejected because the proposal explicitly forbids a
hidden fallback and the candidate-scoring path is already fully operational.

### Decision 2: BaselineRunner migration approach

`BaselineRunner` is migrated in place, not replaced. Its `_process_example`
method changes the classification step from `classify_detailed` →
`ClassificationResult` to `CandidateScorer.score` →
`ClassificationPolicy.classify` → `Classification`. The thesis extraction step
(`extract_theses_detailed`) is preserved unchanged.

The runner gains `CandidateScorer` and `ClassificationPolicy` dependencies
alongside the existing `AsyncTask` dependency.

**Alternative considered**: Remove `BaselineRunner` entirely and use
`classify_input` use case. Rejected because the runner also performs thesis
extraction, checkpointing, and artifact writing that are outside the
`classify_input` use case's scope.

### Decision 2a: ResultRow artifact migration

`ResultRow` is an internal application artifact. All consumers are within the
codebase and will be migrated. The field changes are:

- `predicted_decision`: **Kept**, populated from `Classification.selected.value`
  (converted to `int` for binary classification). This avoids migrating the
  metrics module's core logic, which depends on `predicted_decision` for
  accuracy, PR/F1, confusion matrix, and version comparison.
- `confidence`: **Removed**. No calibrated confidence in candidate scoring. The
  field is not used in any metrics computation. Removed from `ResultRow`,
  `to_dict`, and metrics `REQUIRED_FIELDS`.
- `raw_classify`: **Removed**. No raw model response in candidate scoring.
- `selected_candidate`: **Added** (string, the selected `Candidate.value`).
  Diagnostic field alongside `predicted_decision`.
- `judgment_scores`: **Added** (dict mapping candidate values to scores).
  Diagnostic scoring output.

The metrics module (`application/services/metrics/metrics.py`) is updated to
remove `confidence` from `REQUIRED_FIELDS`. `predicted_decision` remains
unchanged in the metrics module — it is populated from candidate scoring
instead of generative classification, but the field name and semantics
(predicted label as int) are preserved.

Tests in `tests/unit/run_cycle/test_steps.py` that include `confidence` in
synthetic result rows are updated to omit it.

### Decision 2b: Candidate source for BaselineRunner

`BaselineRunner` receives the candidate set through a `candidates` parameter at
construction time, sourced from configuration. A new `[classification].candidates`
field in `config.toml` provides the candidate values as a list of strings,
defaulting to `["0", "1"]` for binary classification. The runner converts these
to `Candidate` objects.

The runner does not infer or generate candidates. The same `Candidate` objects
are passed to `CandidateScorer.score` and their ordering is preserved through
to `ClassificationPolicy`. The candidate set is fixed for the duration of a run.

**Alternative considered**: Derive candidates from dataset label values.
Rejected because it couples the candidate set to the dataset and assumes the
candidate labels match the dataset labels, which may not hold for non-binary
classification.

### Decision 3: LLMClient method removal scope

Remove from `AsyncTask`/`LLMClient`: `classify`, `classify_detailed`,
`classify_many`, `classify_many_detailed`. Preserve: `extract_theses`,
`extract_theses_detailed`, `extract_theses_many`, `extract_theses_many_detailed`,
`analyze`, `analyze_raw`, `analyze_many`.

Remove from the `LLMClient` outbound port (`application/ports/outbound/llm_client.py`):
`classify`, `classify_many`. Preserve: `extract_theses`, `extract_theses_many`.

`classify_many` is removed because it represents generative classification over
multiple classification requests. It is NOT replaced by a `CandidateScorer`
equivalent; candidate batching is already part of a single
`CandidateScorer.score(text, candidates)` invocation. There is no
`BatchCandidateScorer` — the existing `CandidateScorer` handles multiple
candidates in one call through internal batch scoring.

The `analyze` and `analyze_raw` methods remain because they are general-purpose
structured generation utilities not specific to classification. The
`call_type="classify"` parameter value is no longer used by any wrapper but
`analyze`/`analyze_raw` accept arbitrary `call_type` strings.

### Decision 4: CLASSIFICATION_PROMPT_V0 handling

`CLASSIFICATION_PROMPT_V0` is an active seed/base template for the prompt
versioning system, not an immutable historical artifact. It is kept **unchanged**.

The generation-only components (`output_contract`, `fallback`) remain in the
prompt artifact as legacy text. They are NOT used for generative classification
after this change because `classify_detailed` is removed. No code path reads
the prompt for generative classification purposes.

`PromptLayer` is NOT modified. Its fields remain required. The prompt
versioning system (`compose_prompt`, `compare_versions`, `cross_task_transfer`,
`prompt_store`, `run_cycle`) is NOT modified. This avoids broadening the
migration with unnecessary changes to the prompt model.

The semantic classification criteria currently represented primarily by `role`,
`task`, and `rules` are preserved in `CLASSIFICATION_PROMPT_V0`. Their content
is no longer used as a generation protocol; the scoring prompt
(`ScoringPromptBuilder._SEMANTIC_PREFIX`) contains the candidate-evaluation form
of those criteria.

**Alternative considered**: Make `output_contract` and `fallback` optional in
`PromptLayer` and empty them in `CLASSIFICATION_PROMPT_V0`. Rejected because
`PromptLayer` is classification-specific with structurally required fields, and
changing the schema broadens the migration unnecessarily. The generation
protocol text is inert once `classify_detailed` is removed.

**Alternative considered**: Remove `CLASSIFICATION_PROMPT_V0` entirely. Rejected
because it is the active seed for the prompt versioning system. Removing it
would break `prompt_store._seed_baseline`, `compose_prompt`,
`compare_versions`, `cross_task_transfer`, and `run_cycle`.

### Decision 5: Removal order

Follow the proposal's removal order (Section 19):
1. Identify all consumers (search-based verification, including `ResultRow`
   field consumers)
2. Migrate production callers (`BaselineRunner` + metrics module)
3. Migrate tests
4. Remove generative classification API (`classify_detailed`, `classify`, etc.)
5. Remove `ClassificationResult`
6. Remove generation-specific parsing
7. Remove dead infrastructure
8. Run full test suite
9. Repository-wide cleanup

### Decision 6: Error handling

Generative-specific error types are removed if they have no remaining consumers
after migration. Scoring errors continue to propagate through the established
scoring error boundary: `LLMScoringError` (infrastructure-internal,
`infrastructure/llm/scoring/errors.py`) is translated to
`CandidateScoringError` (application-facing, `domain/errors/scoring.py`) at the
port boundary in `LLMLogitCandidateScorer.score()`. The `BaselineRunner` catches
`CandidateScoringError` and maps it to `classify_status=failed` in `ResultRow`,
parallel to how it maps extraction failures to `extract_status=failed`.

### Decision 7: Semantic migration validation

The semantic classification criteria from `CLASSIFICATION_PROMPT_V0` are already
fully migrated into `ScoringPromptBuilder._SEMANTIC_PREFIX`. This is verified by
`TestJudgmentPromptSemantics` in `tests/unit/candidate_scoring/test_scoring_seams.py`.

Before the legacy prompt's generation protocol is considered fully removed, the
scoring prompt MUST contain all semantic classification requirements identified
during the Prompt Classification → Judgment Prompt migration. The existing tests
serve as this verification. No additional semantic migration is needed in this
change; this change only deletes the generative code path, not the semantic
content.

## Risks / Trade-offs

- **Hidden production consumer**: A production path may still depend on
  `classify_detailed` or `ClassificationResult` beyond `BaselineRunner`.
  → Mitigation: Repository-wide `grep` before deletion; full test suite must
  pass; forbidden-reference tests verify no deleted symbols remain.

- **Metrics module breakage**: Removing `confidence` from `ResultRow` and
  `REQUIRED_FIELDS` may affect the metrics module or tests that include it.
  → Mitigation: `confidence` is not used in any computation; removing it from
  `REQUIRED_FIELDS` is straightforward. Tests with synthetic `confidence` values
  are updated.

- **Behavioral difference**: Candidate scoring and generative classification
  produce different results. The selected class may change even with preserved
  semantic criteria. This is expected and not a regression.
  → Mitigation: No attempt to force score-based classification to reproduce
  generated `decision`/`confidence` behavior. Correctness is defined by the
  candidate-scoring flow, not byte-for-byte compatibility.

- **Loss of confidence information**: The old API exposed `confidence`. The new
  architecture intentionally does not provide calibrated confidence.
  → Mitigation: `Judgment.score` remains a ranking signal. Any future confidence
  API requires a separate design. `judgment_scores` in `ResultRow` provides
  diagnostic scoring output.

- **Accidental removal of unrelated generation**: `LLMClient` methods like
  `analyze` and `extract_theses` serve non-classification purposes.
  → Mitigation: Remove only classification-specific methods; preserve
  extraction and general-purpose generation.

## Migration Plan

1. **Search**: Search for all consumers of the legacy APIs and all consumers of
   affected `ResultRow` fields:
   - Legacy APIs: `classify_detailed`, `classify`, `classify_many`,
     `classify_many_detailed`, `ClassificationResult`
   - Affected `ResultRow` fields: `predicted_decision`, `confidence`,
     `raw_classify`
   Classify each occurrence as production dependency, test dependency, unrelated
   use, documentation, or dead code.
2. **Migrate `BaselineRunner`**: Add `CandidateScorer`, `ClassificationPolicy`,
   and candidate set dependencies. Replace `classify_detailed` call with
   `CandidateScorer.score` → `ClassificationPolicy.classify`. Update `ResultRow`
   schema: keep `predicted_decision` (populated from `Classification.selected`),
   remove `confidence` and `raw_classify`, add `selected_candidate` and
   `judgment_scores`.
3. **Migrate metrics module**: Remove `confidence` from `REQUIRED_FIELDS`.
   `predicted_decision` remains unchanged.
4. **Migrate tests**: Update tests asserting generative classification behavior
   to assert candidate-scoring behavior. Update synthetic result rows that
   include `confidence`. Remove tests for deleted methods.
5. **Remove `ClassificationResult`**: Delete
   `application/schemas/classification.py`, update
   `application/schemas/__init__.py`. Update `domain/models/classification.py`
   docstring only if it references `ClassificationResult` or generative
   classification terminology.
6. **Remove `LLMClient` classification methods**: Delete `classify`,
   `classify_detailed`, `classify_many`, `classify_many_detailed` from
   `infrastructure/llm/client.py`. Remove `classify`, `classify_many` from
   `application/ports/outbound/llm_client.py`.
7. **Remove dead code**: Clean imports, mocks, fixtures, error types,
   documentation referencing the generative path. `CLASSIFICATION_PROMPT_V0`
   and `PromptLayer` are NOT modified.
8. **Add forbidden-reference tests**: Verify no production code references
   deleted symbols (see Decision on test scope below).
9. **Run full test suite**: Verify candidate scoring is the only classification
   path.

**Rollback**: Repository-level revert of the migration commit(s). No runtime
fallback is retained. This keeps the architecture explicit.
