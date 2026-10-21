# Proposal

## Why

The candidate scorer builds its scoring prompt from a hardcoded semantic prefix
(`_SEMANTIC_PREFIX`) that duplicates `CLASSIFICATION_PROMPT_V0`'s rules in
judgment language. The active prompt version's dynamic rules
(`PromptArtifact.layers.rules`) never reach the model input. The system can
compose, persist, and activate new prompt versions without their rules
influencing candidate scoring, so a prompt version is not operationally active
even when the application selects it.

## What Changes

- Extend the `CandidateScorer` port contract so `score()` accepts the active
  classification rules as an explicit argument (**BREAKING** for all callers of
  `CandidateScorer.score()`). Rules are `Sequence[str]` — semantic criteria from
  the active `PromptArtifact.layers.rules`, not prompt fragments.
- Update `ScoringPromptBuilder` to incorporate the supplied rules into the
  scoring prompt, placing them after the fixed candidate-evaluation
  instructions and before the input text and candidate continuation. The
  hardcoded `_SEMANTIC_PREFIX` criteria are replaced by the active rules.
- Update `LLMLogitCandidateScorer.score()` to forward rules into the prompt
  builder, preserving the single-batch forward pass and prefix-reuse pattern.
- Thread active rules from `BaselineRunner` (which already holds
  `self._classify_prompt`) into `self._scorer.score(...)`. The runner extracts
  rules from `PromptArtifact.layers.rules` and raises on malformed non-default
  prompts.
- Thread rules through `classify_input` and `ClassifyInputDeps` so the direct
  classification entry point also supplies rules.
- Update all test call sites of `CandidateScorer.score()` and add unit,
  integration, and regression tests verifying the full rule-propagation path
  from `PromptArtifact` to the actual model input.

## Capabilities

### New Capabilities

(None — no new capability is introduced. The scoring contracts, scoring flow,
scorer implementation, and baseline runner already exist; this change modifies
their existing requirements to propagate active rules.)

### Modified Capabilities

- `candidate-scoring-contracts`: The `CandidateScorer` port contract requires
  the active classification rules as an explicit argument to `score()`.
- `llm-logit-scorer`: The prompt construction and concrete scorer requirements
  change to include active rules in the model input and accept rules through
  the port.
- `candidate-scoring-flow`: The `ClassifyInput` use case orchestration changes
  to receive and forward active rules to the scorer.
- `stage1-baseline-runner`: The runner requirement changes to pass active
  prompt rules to the scorer.

## Impact

- **Code**: `CandidateScorer.score()` signature changes (breaking for all
  callers). `ScoringPromptBuilder`, `LLMLogitCandidateScorer`,
  `BaselineRunner`, `classify_input`, and `ClassifyInputDeps` are updated. All
  test fakes/mocks calling `score()` are updated.
- **Specs**: Delta specs for `candidate-scoring-contracts`,
  `llm-logit-scorer`, `candidate-scoring-flow`, and `stage1-baseline-runner`.
- **Dependencies**: No new runtime dependencies.
- **Public API**: `CandidateScorer.score()` gains a required `rules` parameter.
  `ClassifyInputDeps` gains a rule-source field or `classify_input` gains a
  rules parameter. CLI behavior is unchanged from the user's perspective.
- **Scoring semantics**: The scoring algorithm (mean log-probability, softmax
  normalization, candidate tokenization, batched forward pass) is unchanged.
  Scores will differ because the model input now contains active rules.
