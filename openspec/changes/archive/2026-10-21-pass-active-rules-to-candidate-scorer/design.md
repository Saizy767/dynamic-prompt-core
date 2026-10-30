# Design

## Context

The candidate-scoring pipeline is fully implemented and wired into both the
baseline runner and the optimization cycle:

- `CandidateScorer` port (`application/ports/outbound/candidate_scorer.py`) with
  `score(text, candidates) -> list[Judgment]`.
- `LLMLogitCandidateScorer` (`infrastructure/llm/scoring/candidate_scorer.py`)
  constructs a batched forward pass using `ScoringPromptBuilder`.
- `ScoringPromptBuilder` (`infrastructure/llm/scoring/prompt_builder.py`) builds
  the prompt from a hardcoded `_SEMANTIC_PREFIX` + `text` + `Candidate: ` +
  `candidate.value`. The `_SEMANTIC_PREFIX` is a hand-maintained duplicate of
  `CLASSIFICATION_PROMPT_V0`'s rules rewritten in candidate-evaluation language.
- `BaselineRunner` (`application/use_cases/run_baseline/runner.py`) holds
  `self._classify_prompt` (a `PromptArtifact` with `.layers.rules`) but calls
  `self._scorer.score(example.text, self._candidates)` without passing rules.
- `classify_input` (`application/use_cases/classify_input/classify_input.py`)
  calls `deps.scorer.score(text, candidates)` without rules.
- `PromptLayer.rules: list[str]` (`domain/prompts/base.py`) is the structured
  source of classification rules. `PromptArtifact.layers` is
  `PromptLayer | None`.
- Three scoring backends (Hugging Face, GGUF, llama-server) all share the same
  `LLMLogitCandidateScorer` and `ScoringPromptBuilder` — only the tokenizer and
  model adapters differ. The factory (`infrastructure/llm/scoring/factory.py`)
  constructs `ScoringPromptBuilder()` with no arguments.

There are exactly two production call sites of `CandidateScorer.score()`:
`run_baseline/runner.py:447` and `classify_input/classify_input.py:30`.

## Goals / Non-Goals

**Goals:**

- Make the active `PromptArtifact.layers.rules` reach the actual model input in
  every scoring invocation.
- Extend the `CandidateScorer` port with an explicit `rules` parameter so rule
  propagation is visible in the type contract, not hidden in global state.
- Preserve the existing scoring algorithm, batched forward pass, prefix-reuse
  pattern, and all backend adapters.
- Handle missing/malformed prompt layers with a clear error, not silent
  default substitution.
- Keep the diff focused on rule propagation and its directly necessary
  validation.

**Non-Goals:**

- Change the scoring algorithm (mean log-probability, softmax normalization,
  candidate tokenization, causal logit alignment).
- Change candidate generation, selection, ranking, or classification policy.
- Redesign the prompt composition pipeline or change `PromptArtifact` structure.
- Introduce probability calibration or score rescaling.
- Guarantee that adding a rule improves classification accuracy.

## Decisions

### Decision: Add `rules: Sequence[str]` as a required parameter to `CandidateScorer.score()`

The port signature becomes:

```python
async def score(
    self,
    text: str,
    candidates: list[Candidate],
    rules: Sequence[str],
) -> list[Judgment]: ...
```

Rules are semantic classification criteria (plain strings from
`PromptArtifact.layers.rules`), not prompt fragments or rendered text. Making
the parameter required (not optional) forces every caller to consciously supply
the active rule set — a missing `rules` argument is a wiring bug, not a
fallback case.

- **Alternative considered**: optional `rules` parameter defaulting to `None`
  with the scorer falling back to hardcoded criteria. Rejected: a silent
  fallback is exactly the disconnect this change eliminates. Any call site that
  forgets to pass rules would silently use stale hardcoded criteria.
- **Alternative considered**: a typed `ScoringContext` request object wrapping
  `text`, `candidates`, and `rules`. Rejected: it adds a new type for a single
  field without enough structural benefit to justify the indirection. The
  existing two-parameter signature gains one parameter; a request object is
  more appropriate when the contract has many optional fields.
- **Alternative considered**: pass the entire `PromptArtifact` to the scorer.
  Rejected: the scorer is an infrastructure port that must not depend on
  domain prompt-composition types. Rules as `Sequence[str]` keeps the port
  boundary clean — the application extracts rules from the artifact, the
  scorer receives plain strings.

### Decision: `ScoringPromptBuilder.build_prefix(text, rules)` accepts rules per-call

The prompt builder remains stateless. `build_prefix` gains a `rules` parameter:

```python
def build_prefix(self, text: str, rules: Sequence[str]) -> str: ...
```

The prompt structure becomes:

```
<fixed candidate-evaluation instructions>

Classification rules:
- <rule 1>
- <rule 2>
...

Input text:
<text>
Candidate: <candidate.value>
```

The fixed instructions are the minimal candidate-evaluation framing ("You are a
binary text classifier. Evaluate how well the candidate matches the input text
according to the classification rules below."). The hardcoded
`_SEMANTIC_PREFIX` criteria bullets are removed — they are replaced by the
active rules supplied per-call.

- **Alternative considered**: make `ScoringPromptBuilder` stateful, holding
  rules set at construction time. Rejected: the factory constructs the builder
  once at process start, but rules change per prompt version. A stateful
  builder would require re-construction or mutation on every version change,
  breaking the shared-scorer pattern where one scorer instance serves all
  rounds.
- **Alternative considered**: keep the hardcoded prefix and append rules after
  it. Rejected: this would duplicate the default rules (hardcoded) alongside
  the active rules, confusing the model and violating the "no default rule
  duplication in builder" requirement. The fixed instructions stay; the
  criteria bullets become the active rules.

### Decision: Runner extracts rules from `PromptArtifact.layers.rules` with explicit error handling

`BaselineRunner` already holds `self._classify_prompt: PromptArtifact`. The
runner extracts rules once at construction time (not per-example):

```python
def _extract_rules(self) -> list[str]:
    layers = self._classify_prompt.layers
    if layers is None:
        if self._classify_prompt is CLASSIFICATION_PROMPT_V0:
            return list(_V0_LAYERS.rules)
        raise BaselineRunnerError(
            f"active prompt {self._classify_prompt.version} has no layers; "
            "cannot extract classification rules"
        )
    return list(layers.rules)
```

The default prompt (`CLASSIFICATION_PROMPT_V0`) always has valid layers, so the
`None` check only fires for non-default prompts with missing layers — a
malformed-artifact condition that should raise, not silently default.

Rules are extracted once at construction and stored as `self._rules`, then
passed to every `scorer.score(text, candidates, self._rules)` call. This avoids
re-extracting per example and makes the rule set visible in the runner's state.

- **Alternative considered**: extract rules per-example inside
  `_process_example`. Rejected: the active prompt does not change per-example
  within one runner instance; per-example extraction is unnecessary work and
  obscures the invariant that one runner uses one rule set.
- **Alternative considered**: fall back to `CLASSIFICATION_PROMPT_V0` rules
  when layers are missing for any prompt. Rejected: this silently substitutes
  default rules for a non-default prompt, violating FR-5 and AC-5/AC-10.

### Decision: `classify_input` gains a `rules` parameter

The direct classification entry point receives rules as an explicit argument:

```python
async def classify_input(
    deps: ClassifyInputDeps,
    text: str,
    candidates: list[Candidate],
    rules: Sequence[str],
) -> Classification: ...
```

`ClassifyInputDeps` is unchanged — it still carries only `scorer` and `policy`.
The caller (composition root or calling use case) is responsible for supplying
the active rules. This keeps the dependency object stable and makes the rule
source explicit at each call site.

- **Alternative considered**: add a rule-source port to `ClassifyInputDeps`
  (e.g., a `PromptRepository` that loads the active prompt). Rejected: this
  introduces an implicit dependency on prompt-store state inside the use case.
  The caller already knows which prompt version is active; passing rules
  explicitly is simpler and testable.

### Decision: `LLMLogitCandidateScorer.score()` forwards rules to the prompt builder

The scorer's `score()` gains the `rules` parameter and passes it to
`build_prefix`:

```python
async def score(self, text, candidates, rules):
    prefix = self._prompt_builder.build_prefix(text, rules)
    candidate_texts = [self._prompt_builder.build_candidate(c) for c in candidates]
    prefixes = [prefix] * len(candidates)
    # ... rest unchanged: tokenize batch, forward, score, softmax
```

The prefix is still constructed once and reused for all candidates. The batched
forward pass, tokenization, logit extraction, and softmax are unchanged. Only
the prefix content changes (it now contains active rules).

### Decision: Factory unchanged

`build_candidate_scorer()` constructs `ScoringPromptBuilder()` with no
arguments — this stays the same because the builder remains stateless. Rules
flow per-call through `score()`, not through construction. No factory change is
needed.

### Decision: Update all test fakes and mocks

There are ~35 test call sites of `CandidateScorer.score()`. Each must add the
`rules` parameter. Test fakes (`FakeCandidateScorer`, recording scorers) must
accept and (where relevant) record the rules. A recording model adapter or
prompt-builder double is used in integration tests to assert that the actual
model input contains the active rules.

## Risks / Trade-offs

- **[Breaking `CandidateScorer.score()` signature]** Every caller must add the
  `rules` parameter.
  → Mitigation: there are exactly two production call sites, both updated in
  this change. Test call sites are mechanical updates. A `TypeError` at call
  time is the desired failure mode for a missing `rules` argument.

- **[Score changes]** Scores will differ because the model input now contains
  active rules instead of hardcoded criteria. Existing golden-file or
  score-assertion tests may need updating.
  → Mitigation: the scoring algorithm is unchanged; only the prompt content
  changes. Tests that assert exact scores should use a recording adapter to
  verify prompt content, not real-model outputs. Regression tests for the
  algorithm (logit alignment, softmax, mean log-prob) are unaffected because
  they test `LogitScorer` directly, not through the prompt builder.

- **[Contract test for `_SEMANTIC_PREFIX` ↔ V0 correspondence]** An existing
  test (`tests/unit/candidate_scoring/test_scoring_seams.py`) verifies that the
  hardcoded semantic prefix represents V0 rules. This test becomes obsolete
  because the hardcoded prefix is removed.
  → Mitigation: replace it with a test verifying that the default prompt's
  rules produce the expected prompt structure through the new
  `build_prefix(text, rules)` path.

- **[Empty rules for default prompt]** `CLASSIFICATION_PROMPT_V0` always has
  4 rules (enforced by `build_classification_prompt`), so the default path
  never has empty rules. But a future prompt version could theoretically have
  zero rules if the 3–5 constraint is relaxed.
  → Mitigation: the scorer uses whatever rules are supplied. An empty list is
  valid (the prompt simply has no criteria section); it is not silently
  defaulted. This is the documented contract.

## Migration Plan

1. Update `CandidateScorer` port: add `rules: Sequence[str]` to `score()`.
2. Update `ScoringPromptBuilder`: replace `_SEMANTIC_PREFIX` criteria with
   `build_prefix(text, rules)` that incorporates supplied rules.
3. Update `LLMLogitCandidateScorer.score()`: add `rules` parameter, forward to
   `build_prefix`.
4. Update `BaselineRunner`: extract rules from `self._classify_prompt` at
   construction, pass to `scorer.score()`.
5. Update `classify_input`: add `rules` parameter, forward to `scorer.score()`.
6. Update all test fakes, mocks, and call sites.
7. Add unit tests for `ScoringPromptBuilder` (rules inclusion, order, placement,
   no substitution, determinism).
8. Add unit tests for `LLMLogitCandidateScorer` (rules forwarded to builder,
   prefix reuse, batch behavior preserved).
9. Add integration tests with a recording adapter verifying rules reach the
   model input through `run_cycle`, `BaselineRunner`, and `classify_input`.
10. Add regression tests for default prompt, empty candidates, and
    no-implicit-substitution.
11. Run `pytest`, `mypy`, `ruff check`, and `lint-imports`.

No runtime fallback is required. Revert via repository-level revert.

## Open Questions

(None — the data flow is already established: `PromptArtifact.layers.rules` is
available at every `BaselineRunner` construction site, and the port extension
is a single required parameter.)
