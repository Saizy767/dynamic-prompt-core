# Proposal

## Why

Classification and extraction are currently entangled inside `BaselineRunner`: extraction runs first via `AsyncTask`, then classification runs via `CandidateScorer` + `ClassificationPolicy`, both sharing the same `PromptArtifact`-based prompt infrastructure. This coupling prevents independent system prompts, independent model calls, and clean separation of concerns. The runner also raises on absent rules instead of gracefully defaulting classification to `0`, and the `CandidateScorer` / `ClassificationPolicy` split forces a two-step `score → classify` flow that the application must orchestrate rather than delegating to a single self-contained component.

## What Changes

- **New `ClassificationRules` domain model** — immutable `tuple[str, ...]` representation of the active rule set, with conversion from `PromptArtifact.layers.rules`. Distinguishes empty (no rules) from invalid (malformed) rules.
- **New `Classifier` component** — an outbound port in `application/ports/outbound/` with infrastructure implementation in `infrastructure/llm/`. Encapsulates the full classification process: rule validation, candidate handling, logit-based scoring, and final decision. Returns the existing `Classification` type. Absorbs `CandidateScorer` + `ClassificationPolicy` into one self-contained component with its own system prompt.
- **New `Extractor` component** — standalone component with its own system prompt and independent model call. Performs key-phrase extraction, thesis normalization, and existing status/error handling. Does not receive classification results as input.
- **New system-prompt resources** — two separate prompt files under `infrastructure/llm/prompts/` (one for `Classifier`, one for `Extractor`), packaged with the Python distribution via `package-data`.
- **`BaselineRunner` restructured** — receives `Classifier` and `Extractor` through DI. Runs classification first, then extraction. When rules are absent or empty, sets classification to `0` without calling `Classifier`, then still runs `Extractor`. On holdout, skips `Extractor` as today.
- **`CandidateScorer` port removed** — **BREAKING** at the port level. `CandidateScorer`, `LLMLogitCandidateScorer`, `ScoringPromptBuilder`, and `ClassificationPolicy` as separate architectural abstractions are removed/renamed. The logit-scoring algorithm is preserved inside the `Classifier` implementation.
- **`run_cycle/steps.py`** — minimal integration changes to wire new `Classifier` and `Extractor` interfaces. Cycle policy, sequence, acceptance criteria, and rollback logic unchanged.
- **`ResultRow` and JSONL** — field names, types, and semantics preserved. No new fields, no removed fields.

## Capabilities

### New Capabilities

- `classification-rules`: Immutable domain model representing the normalized set of classification rules, with conversion from `PromptArtifact` and validation at the boundary
- `classifier`: Self-contained classification component (port + infrastructure implementation) that performs the full classification process using logit-based scoring and returns the existing `Classification` type, with an independent system prompt
- `extractor`: Standalone key-phrase extraction component with an independent system prompt and model call, preserving existing extraction algorithm, normalization, statuses, and error handling
- `system-prompt-resources`: File-based system prompts for `Classifier` and `Extractor` under `infrastructure/llm/prompts/`, packaged with the Python distribution and loaded at component initialization

### Modified Capabilities

- `stage1-baseline-runner`: Runner orchestrates `Classifier` then `Extractor` (order reversed from current extraction-then-classification). Absent or empty rules set classification to `0` without calling `Classifier`. Receives `Classifier` and `Extractor` via DI instead of `CandidateScorer` + `ClassificationPolicy` + `AsyncTask`. `skip_extraction` still skips `Extractor`. `ResultRow` structure preserved.
- `candidate-scoring-contracts`: **BREAKING** — `CandidateScorer` port removed. `Classifier` port replaces it with a different contract: accepts classification input + `ClassificationRules`, returns `Classification` directly (not `list[Judgment]`). `Candidate` and `Judgment` remain as internal domain models used by the `Classifier` implementation.
- `candidate-scoring-flow`: **BREAKING** — the two-step `score → classify` flow (`CandidateScorer.score` → `ClassificationPolicy.classify`) is replaced by a single `Classifier.classify` call. `ClassificationPolicy` as a separate application-internal strategy is removed; the decision logic moves inside the `Classifier` implementation. `Classification` remains the return type.
- `candidate-scoring-validation`: Validation and architectural-invariant tests updated to reference `Classifier` instead of `CandidateScorer` + `ClassificationPolicy`. The single-production-construction-path invariant applies to `Classifier`. Logit-scoring, error-propagation, and no-partial-results invariants preserved.

## Impact

- **Domain layer**: New `classification_rules.py` model. `Candidate`, `Judgment`, `Classification` remain. `ClassificationPolicy` removed from `application/services/`.
- **Application layer**: New `Classifier` port in `application/ports/outbound/`. `BaselineRunner` restructured. `ClassifyInput` use case adapted or removed. `RunCycleDeps` swaps `scorer`/`policy` for `classifier`/`extractor`.
- **Infrastructure layer**: New `Classifier` and `Extractor` implementations in `infrastructure/llm/`. Scoring logic (`LogitScorer`, tokenizer/model adapters) preserved but repackaged under the `Classifier`. New `prompts/` resource directory. `ScoringPromptBuilder` renamed/adapted. `CandidateScorer` port and `LLMLogitCandidateScorer` removed after dependency audit.
- **Composition roots**: `interfaces/cli/main.py`, runner CLI, comparator CLI updated to construct `Classifier` and `Extractor` instead of `CandidateScorer` + `ClassificationPolicy`.
- **Packaging**: `pyproject.toml` gains `package-data` for prompt resource files.
- **Tests**: Unit, integration, contract, and regression tests updated. New unit tests for `ClassificationRules`, `Classifier`, `Extractor`. Existing `candidate-scoring-*` tests adapted.
- **No external API changes**: CLI interface, config format, and JSONL output unchanged.
