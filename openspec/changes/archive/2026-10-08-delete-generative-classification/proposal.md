# Proposal

## Why

The candidate-scoring architecture (`CandidateScorer` → `Judgment[]` →
`ClassificationPolicy` → `Classification`) is now the primary classification
mechanism. The old generative classification path (`classify_detailed` →
`LLMClient` → structured JSON generation → `ClassificationResult`) is no longer
required and must be removed so that candidate scoring becomes the single
supported classification mechanism, eliminating a hidden fallback and the
maintenance burden of two parallel inference architectures.

## What Changes

- **BREAKING**: Remove `ClassificationResult` schema (`application/schemas/classification.py`)
  and all its consumers.
- **BREAKING**: Remove `classify_detailed`, `classify`, `classify_many`, and
  `classify_many_detailed` from `AsyncTask`/`LLMClient` infrastructure and the
  `LLMClient` outbound port. Preserve `extract_theses*` and `analyze*` methods
  that serve non-classification purposes.
- **BREAKING**: Migrate `BaselineRunner` to use the candidate-scoring flow
  (`CandidateScorer` → `ClassificationPolicy`) for classification instead of
  `classify_detailed` → `ClassificationResult`. Thesis extraction within the
  runner is preserved.
- The legacy classification prompt (`CLASSIFICATION_PROMPT_V0`) and its
  `PromptLayer` structure are kept unchanged. The generation-only components
  (`output_contract`, `fallback`) remain as inert text in the prompt artifact
  for the versioning system; no code path uses them for generative
  classification after `classify_detailed` is removed. The semantic criteria
  are already represented by `ScoringPromptBuilder`.
- Remove classification-specific JSON parsing, structured-output schemas,
  response models, and confidence extraction from the production path.
- Remove "Existing generative classification preserved" requirements from all
  specs that currently mandate coexistence of the two paths.
- Remove the `LLMClient` port's `classify` and `classify_many` methods;
  preserve `extract_theses` and `extract_theses_many`.
- Remove obsolete generative error types with no remaining consumers.
- Remove dead imports, mocks, fixtures, and documentation referencing the
  generative classification path.
- Preserve `CandidateScorer`, `Judgment`, `Classification`,
  `ClassificationPolicy`, `ArgmaxClassificationPolicy`, and the batch-scoring
  architecture unchanged.

## Capabilities

### New Capabilities

(None — this change removes an existing path and migrates callers to the
already-specified candidate-scoring architecture.)

### Modified Capabilities

- `candidate-scoring-flow`: Remove the "Existing generative classification
  preserved" requirement and update `ClassificationResult` references in the
  classification domain concept to reflect that the generative path no longer
  exists.
- `candidate-scoring-contracts`: Remove the "Existing classification behavior
  preserved" requirement that mandates coexistence with the generative pipeline.
- `stage1-baseline-runner`: Migrate the classification step from
  `classify_detailed` → `ClassificationResult` to the candidate-scoring flow.
  The result schema changes: `predicted_decision` is kept (populated from
  `Classification.selected`) for metrics compatibility, `confidence` and
  `raw_classify` are removed, `selected_candidate` and `judgment_scores` are
  added. The metrics module is updated to remove `confidence` from required
  fields. Thesis extraction remains unchanged.
- `llm-transport-client`: Remove `classify_detailed` and
  `classify_many_detailed` from the rich-result method requirements. Preserve
  `extract_theses_detailed` and `extract_theses_many_detailed`.
- `judgment-prompt`: Remove the "Existing generative classification preserved"
  requirement that mandates coexistence with the generative pipeline.
- `llm-logit-scorer`: Remove the "generative classification path unchanged"
  clause from the batch-scoring preservation requirement.

## Impact

- **Code**: `application/schemas/classification.py` deleted;
  `application/schemas/__init__.py` updated; `application/ports/outbound/llm_client.py`
  classification methods removed; `infrastructure/llm/client.py` classification
  methods removed; `application/use_cases/run_baseline/runner.py` migrated to
  candidate-scoring (new `CandidateScorer`/`ClassificationPolicy` dependencies,
  candidate set from config); `application/services/metrics/metrics.py` updated
  to remove `confidence` from required fields; `domain/models/classification.py`
  docstring updated only if it references `ClassificationResult`;
  `CLASSIFICATION_PROMPT_V0` and `PromptLayer` kept unchanged; dead imports
  cleaned.
- **Specs**: Six existing capabilities receive delta specs removing
  coexistence requirements and migrating the baseline runner.
- **Dependencies**: Classification-specific Pydantic schema dependencies
  removed. No new runtime dependencies.
- **Public API**: **BREAKING** — `ClassificationResult`, `classify_detailed`,
  `classify`, `classify_many` removed. `CandidateScorer.score(text, candidates)`
  → `ClassificationPolicy.classify(judgments)` → `Classification` becomes the
  only supported classification path.
- **Tests**: Tests asserting generative classification behavior are migrated or
  removed. Forbidden-reference tests verify no production code references the
  deleted symbols.
