# Design

## Context

The candidate-scoring architecture is implemented: `ScoringPromptBuilder`
(`infrastructure/llm/scoring/prompt_builder.py`) produces a minimal prompt
`{input}\nCandidate: {candidate}`, `LLMLogitCandidateScorer` orchestrates batch
tokenization and batched model inference, and `LogitScorer` computes mean
candidate-token log-probabilities with causal alignment. The application calls
`CandidateScorer.score(text, candidates)` and receives `Judgment[]`, which
`ClassificationPolicy` converts into a `Classification`.

The existing generative classification prompt (`CLASSIFICATION_PROMPT_V0` in
`domain/prompts/fixed.py`) is a layered system prompt with `role`, `task`,
`rules`, `output_contract`, and `fallback`. The `output_contract` instructs the
model to return a JSON object with `decision` and `confidence` fields. The
`task` and `rules` contain semantic classification criteria: what the
classification represents, what makes a candidate relevant, how to distinguish
similar intents, and how to handle ambiguous cases.

The current `ScoringPromptBuilder` carries no semantic classification criteria.
The prompt `{input}\nCandidate: {candidate}` establishes the prefix/candidate
boundary but provides no task context to the model. This change enriches the
prompt prefix with semantic criteria from the generative prompt while excluding
generation-protocol instructions, keeping the candidate continuation and
prefix/candidate boundary intact.

## Goals / Non-Goals

**Goals:**
- Enrich `ScoringPromptBuilder` with semantic classification criteria in the
  prompt prefix.
- Exclude generation-protocol instructions (JSON, confidence,
  structured-output schema, and response-format requirements) from the
  judgment prompt. Selection-oriented task instructions SHALL instead be
  rewritten into candidate-evaluation semantics where their underlying
  classification meaning is relevant.
- Preserve the existing prefix/candidate boundary so the tokenizer determines
  candidate continuation positions structurally.
- Keep the prompt candidate-specific: one prompt per candidate, no multi-class
  embedding.
- Preserve candidate text verbatim (no normalization).
- Keep `CandidateScorer`, `Judgment`, `ClassificationPolicy`, `Classification`,
  and the batch-scoring architecture unchanged.
- Keep the generative classification path operational.
- Make the new prompt independently testable.

**Non-Goals:**
- Remove the generative classification implementation or `ClassificationResult`.
- Change `CandidateScorer`, `Judgment`, `Classification`, or
  `ClassificationPolicy`.
- Change the scoring algorithm (mean candidate-token log-probability).
- Introduce score calibration, probability conversion, or confidence semantics.
- Introduce a new application-level prompt abstraction.
- Introduce provider-specific prompting behavior or few-shot examples.
- Change candidate ordering, tie-breaking, or batch scoring architecture.
- Version the prompt wording (deferred to a future change).

## Decisions

### Decision: Semantic criteria extracted by content, not by layer

The existing generative prompt is structured as `PromptLayer(role, task, rules,
output_contract, fallback)`, but the `PromptLayer` structure is not itself a
semantic guarantee. A `role` layer can contain model-behavior instructions; a
`rules` layer can contain output-format instructions. The judgment prompt
migration SHALL review the generative prompt by content rather than assuming
layer semantics. Content that defines the classification task is preserved;
content that exists solely to control generated output is excluded.

In the current prompt, this means the relevant semantic content is extracted
primarily from `role`, `task`, and `rules`, while `output_contract` and
`fallback` are excluded. The judgment prompt prefix SHALL include the preserved
semantic content, rewritten as candidate-evaluation framing where the original
framing is selection-oriented.

- **Alternative considered**: copy the entire generative prompt verbatim.
  Rejected: it contains JSON and confidence instructions that have no role in
  logit scoring and could confuse the model.
- **Alternative considered**: write entirely new semantic criteria from scratch.
  Rejected: the existing criteria encode domain knowledge accumulated through
  the prompt optimization pipeline; discarding them risks semantic drift.

### Decision: Prompt shape is `{semantic_prefix}{input}\nCandidate: {candidate}`

The enriched prompt prefix SHALL be `{semantic_prefix}{input}\nCandidate: `,
where `semantic_prefix` contains the classification criteria extracted from the
generative prompt. The candidate continuation remains `{candidate}` verbatim.
The `build_prefix(text)` method SHALL return `{semantic_prefix}{input}\nCandidate: `
and `build_candidate(candidate)` SHALL return `candidate.value` unchanged. This
preserves the existing prefix/candidate boundary contract used by the batch
tokenizer.

The candidate continuation MUST remain the final textual component of the
prompt. No semantic instructions or other generated text may appear between
`Candidate: ` and `{candidate}`. This is required because the logit scorer
measures the candidate continuation tokens specifically:

```text
semantic instructions
        ↓
input
        ↓
Candidate:
        ↓
candidate tokens  ← scored
```

- **Alternative considered**: `{input}\n{semantic_prefix}Candidate: {candidate}`.
  Rejected: placing semantic criteria after the input separates the criteria
  from the task framing and may reduce their influence on candidate scoring.
- **Alternative considered**: embed semantic criteria as a system prompt
  separate from the user prompt. Rejected: the logit scorer operates on a
  single continuation prompt; introducing system/user separation would require
  changes to the tokenizer and model adapter seams, which are out of scope.

### Decision: Semantic prefix is a fixed infrastructure constant

The semantic classification criteria SHALL be defined as a fixed string constant
inside `infrastructure/llm/scoring/prompt_builder.py`, not derived from the
generative prompt at runtime. This keeps the judgment prompt self-contained,
testable, and independent of the `domain/prompts/` machinery. The constant is
authored by extracting and rewriting the relevant semantic content from
`CLASSIFICATION_PROMPT_V0` during implementation.

The semantic prefix intentionally duplicates the relevant semantic content of
`CLASSIFICATION_PROMPT_V0`. This duplication is accepted for this migration
because the judgment prompt and generative prompt are separate inference
protocols. The two prompts MUST NOT be treated as automatically synchronized:
changes to the generative prompt do not automatically modify the judgment
prompt, and changes to the judgment prompt require explicit review.

- **Alternative considered**: import and transform `CLASSIFICATION_PROMPT_V0`
  at runtime. Rejected: it couples the scoring infrastructure to the generative
  prompt domain model, and the generative prompt may change independently.
- **Alternative considered**: make the semantic prefix configurable via the
  scorer constructor. Rejected: the application SHALL NOT provide prompt
  fragments; configurability would violate the port information boundary.

### Decision: Classification-selection instructions rewritten to evaluation

The generative prompt's `task` instructs the model to "classify the user text
into one of two classes." This SHALL be rewritten as "evaluate how well the
candidate describes or matches the input according to the classification
criteria." The `rules` (decide on content not hashtags, metaphorical cases,
factual matches, ambiguous cases) SHALL be preserved with minimal rewriting to
apply to candidate evaluation rather than binary selection.

### Decision: `ScoringPromptBuilder` API unchanged

The `build_prefix(text)`, `build_candidate(candidate)`, and
`build(text, candidate)` methods retain their existing signatures. The only
change is that `build_prefix` returns a longer string that includes semantic
criteria. `LLMLogitCandidateScorer` continues to call
`build_prefix(text)` and `build_candidate(candidate)` without modification. No
new methods are introduced.

### Decision: Tokenization boundary contract is preserved

The enriched prompt preserves the logical prefix/candidate boundary:

```text
prefix    = semantic_prefix + input + "\nCandidate: "
candidate = candidate.value
```

The tokenizer infrastructure remains responsible for determining the exact
candidate continuation tokens used for scoring.

The correctness invariant is:

> The candidate token IDs used for scoring MUST be exactly the token IDs
> occupying the candidate continuation positions in the model input sequence.

The implementation MUST NOT assume that separately encoding the prefix and
candidate (`encode(prefix) + encode(candidate)`) is universally equivalent to
encoding the concatenated prompt (`encode(prefix + candidate)`), because
tokenizer behavior may be boundary-sensitive.

The tokenizer may use separate encoding, combined encoding with offset
information, or another tokenizer-specific mechanism, provided that the
invariant above is satisfied.

This change does not alter the logical prefix/candidate boundary or expose
tokenization details to the application.

## Risks / Trade-offs

- **[Prompt semantic drift]** Extracting and rewriting semantic criteria may
  inadvertently change the classification task's meaning.
  → Mitigation: Extract criteria verbatim where possible; rewrite only
  selection-oriented instructions. Add semantic migration tests verifying that
  each identified semantic requirement from `CLASSIFICATION_PROMPT_V0` is
  represented in the judgment prompt and that generation-only requirements are
  absent. Tests should assert semantic prompt sections or stable authored
  requirements rather than requiring literal phrase preservation, because
  rewriting necessarily changes some phrases.

- **[Score distribution shift]** Adding semantic criteria to the prefix changes
  the model input, which changes token likelihoods and score distributions even
  if the classification semantics remain conceptually equivalent.
  → Mitigation: Treat this as a model-behavior change. Establish representative
  ranking regression tests. Exact numerical equality with the old minimal prompt
  is neither expected nor required.

- **[Candidate-as-instruction confusion]** A poorly structured prompt could
  cause the model to interpret the candidate as an instruction rather than the
  hypothesis being evaluated.
  → Mitigation: Keep the candidate in the `Candidate: {value}` position as the
  final textual component, clearly separated from the semantic criteria and
  input. The candidate label establishes the candidate as data, not instruction.

- **[Token-boundary sensitivity]** The longer prefix may change tokenization
  behavior at the prefix/candidate boundary for some tokenizers.
  → Mitigation: The tokenization boundary contract requires that candidate
  token IDs used for scoring are exactly the token IDs at candidate continuation
  positions in the model input sequence. The tokenizer implementation is
  responsible for satisfying this invariant regardless of the mechanism used
  (separate encoding, combined encoding with offsets, or another approach).
  Token-boundary tests verify actual candidate continuation positions, not
  assumed separate encoding equivalence.

- **[Semantic duplication drift]** The semantic prefix duplicates content from
  `CLASSIFICATION_PROMPT_V0`. The two prompts may diverge over time if one is
  updated without updating the other.
  → Mitigation: The duplication is explicitly documented. The two prompts MUST
  NOT be treated as automatically synchronized. Changes to either prompt require
  explicit review of the other.

## Migration Plan

1. **Extract semantic criteria**: Review `CLASSIFICATION_PROMPT_V0` by content.
   Identify classification-task semantics (what the classification represents,
   relevance criteria, ambiguity rules) for preservation; identify
   output-format instructions (JSON, confidence, schema, fallback) for
   exclusion; identify selection-oriented instructions for rewriting into
   candidate-evaluation semantics.
2. **Author semantic prefix**: Write the semantic prefix constant in
   `prompt_builder.py`, rewriting selection instructions as evaluation
   instructions.
3. **Update `build_prefix`**: Incorporate the semantic prefix into the prefix
   returned by `build_prefix(text)`.
4. **Add prompt construction tests**: Test that the prompt contains semantic
   criteria, excludes generation-protocol instructions, preserves candidate
   text, and maintains the prefix/candidate boundary.
5. **Add semantic migration tests**: Verify that each identified semantic
   requirement from `CLASSIFICATION_PROMPT_V0` is represented in the judgment
   prompt and that generation-only requirements are absent. Tests should assert
   semantic prompt sections or stable authored requirements rather than literal
   phrase preservation.
6. **Validate batch compatibility**: Run existing batch scoring tests to
   confirm the enriched prompt does not break batch tokenization, boundary
   detection, or score computation.
7. **Keep generative path unchanged**: Do not modify `CLASSIFICATION_PROMPT_V0`,
  `LLMClient.classify`, `ClassificationResult`, or `BaselineRunner`.

## Open Questions

- Should the semantic prefix be versioned once the first judgment prompt is
  established? Deferred — versioning is a separate concern; the initial
  implementation uses a fixed constant.
- Should representative legacy classification examples become permanent
  prompt-regression fixtures? Deferred — useful but not blocking; can be added
  after initial behavioral validation.
- Should few-shot examples be considered in a later optimization step? Deferred
  — explicitly out of scope per the non-goals.
