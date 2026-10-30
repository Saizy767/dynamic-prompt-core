# Proposal

## Why

The candidate-scoring infrastructure uses a minimal prompt (`{input}\nCandidate:
{candidate}`) that establishes the prefix/candidate boundary for logit scoring
but carries no semantic classification criteria. The existing generative
classification prompt (`CLASSIFICATION_PROMPT_V0`) contains both semantic
task instructions (what makes a candidate relevant, how to distinguish similar
classes) and generation-protocol instructions (return JSON, return a confidence
value, choose exactly one class). The scoring path needs the semantic content
but must not carry the generation protocol, because the scorer measures
candidate-token likelihood rather than asking the model to generate a
classification response. Without this migration, the judgment prompt lacks the
task context that makes candidate scores meaningful.

## What Changes

- Introduce a dedicated **Judgment Prompt** capability that defines the semantic
  contract for candidate-evaluation prompts: the prompt SHALL establish
  classification context for evaluating one explicit candidate, preserve
  semantic criteria from the existing classification task, and exclude
  generation-protocol instructions (JSON output, confidence values, class
  selection, structured-output schemas).
- Enrich the `ScoringPromptBuilder` so the prompt prefix includes semantic
  classification criteria extracted from the existing generative prompt, while
  the candidate continuation remains the exact `Candidate.value` scored by the
  logit scorer.
- Maintain the explicit prefix/candidate boundary so the tokenizer can
  determine candidate continuation positions structurally.
- Keep the prompt candidate-specific: each candidate receives its own logical
  prompt; the candidate set is never embedded into one multi-class prompt.
- Preserve the existing generative classification path unchanged during this
  migration step.
- Keep `CandidateScorer`, `Judgment`, `ClassificationPolicy`, `Classification`,
  and the batch-scoring architecture unchanged.

## Capabilities

### New Capabilities
- `judgment-prompt`: The semantic contract for candidate-evaluation prompts used
  by the logit scorer — prompt structure, semantic content requirements,
  generation-protocol exclusion, candidate-specificity, prefix/candidate
  boundary, and candidate-text preservation.

### Modified Capabilities
- `llm-logit-scorer`: The prompt construction requirement is strengthened to
  require the prompt to carry semantic classification criteria (not just
  represent the candidate) and to exclude generation-protocol instructions. The
  prompt builder collaborator contract is refined to reflect the judgment prompt
  semantics.

## Impact

- **Code**: `infrastructure/llm/scoring/prompt_builder.py` is modified to
  include semantic classification criteria in the prompt prefix. The
  `LLMLogitCandidateScorer` continues to use the prompt builder unchanged in
  its orchestration. No application, domain, or port changes. The generative
  classification path (`CLASSIFICATION_PROMPT_V0`, `LLMClient.classify`,
  `ClassificationResult`, `BaselineRunner`) remains unchanged.
- **Specs**: A new `judgment-prompt` capability spec is introduced. The
  `llm-logit-scorer` spec gains a delta strengthening the prompt construction
  requirement.
- **Dependencies**: No new runtime dependencies.
- **Public API**: Unchanged. `CandidateScorer.score(text, candidates)` remains
  the only application-facing contract.
- **Migration**: This is an incremental step in the candidate-scoring migration.
  The judgment prompt becomes the canonical prompt for candidate scoring. The
  generative classification prompt remains canonical only for the legacy
  generative path until that path is removed in a later step.
