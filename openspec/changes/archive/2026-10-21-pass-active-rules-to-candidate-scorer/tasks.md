# Tasks

## 1. Extend the `CandidateScorer` port contract

- [x] 1.1 Add `rules: Sequence[str]` as a required parameter to
  `CandidateScorer.score()` in
  `src/dynamic_prompt_core/application/ports/outbound/candidate_scorer.py`.
  Update the docstring to describe the rules as semantic classification
  criteria from the active prompt version's `PromptArtifact.layers.rules`.
  Verify with `mypy` that the port type-checks and that existing callers now
  produce a `TypeError` until updated.
- [x] 1.2 Run `ruff check` on the modified port file and verify no lint errors.

## 2. Update `ScoringPromptBuilder` to incorporate active rules

- [x] 2.1 In
  `src/dynamic_prompt_core/infrastructure/llm/scoring/prompt_builder.py`,
  replace the hardcoded `_SEMANTIC_PREFIX` criteria bullets with a fixed
  candidate-evaluation instruction header. Change `build_prefix(self, text)` to
  `build_prefix(self, text, rules)` so the prefix is:
  `<fixed instructions>\n\nClassification rules:\n- <rule 1>\n...\n\nInput text:\n<text>\nCandidate: `.
  Update `build(self, text, candidate)` to accept and forward `rules`.
  Remove the `_SEMANTIC_PREFIX` constant. Verify the module imports cleanly.
- [x] 2.2 Add unit tests in
  `tests/unit/candidate_scoring/test_scoring_prompt_builder.py` (or the
  existing test file) verifying: all supplied rules appear in the prompt;
  rule order is preserved; rules appear after fixed instructions and before
  input text and candidate; Unicode and multiline rules are handled; an empty
  rule list does not produce default rules; identical inputs produce identical
  output. Verify all tests pass.

## 3. Update `LLMLogitCandidateScorer` to forward rules

- [x] 3.1 In
  `src/dynamic_prompt_core/infrastructure/llm/scoring/candidate_scorer.py`,
  add `rules: Sequence[str]` to `score()` and pass it to
  `self._prompt_builder.build_prefix(text, rules)`. The rest of the method
  (batch tokenize, forward, logit score, softmax, judgment construction) is
  unchanged. Verify the prefix is still constructed once and reused for all
  candidates (`prefixes = [prefix] * len(candidates)`).
- [x] 3.2 Add unit tests in
  `tests/unit/candidate_scoring/test_llm_logit_candidate_scorer.py` verifying:
  the supplied rules are passed to the prompt builder (use a recording prompt
  builder double); the rule-aware prefix is used for every candidate in the
  batch; candidate tokenization and score calculation are unchanged; invalid
  scoring context is rejected per the existing error contract. Verify all
  tests pass.

## 4. Thread active rules through `BaselineRunner`

- [x] 4.1 In
  `src/dynamic_prompt_core/application/use_cases/run_baseline/runner.py`, add a
  rule-extraction step at construction time that reads
  `self._classify_prompt.layers.rules`. If `layers` is `None` and the prompt is
  `CLASSIFICATION_PROMPT_V0`, use its default rules. If `layers` is `None` and
  the prompt is not the default, raise a clear error
  (`BaselineRunnerError` or equivalent). Store the extracted rules as
  `self._rules`. Verify by reading the constructor and confirming rules are
  extracted once and stored.
- [x] 4.2 Update the `scorer.score()` call in `_process_example` to pass
  `self._rules` as the `rules` argument. Verify by reading the call site and
  confirming `rules=self._rules` (or positional) is present.
- [x] 4.3 Add a unit test verifying that a non-default prompt with `None`
  layers raises a clear error at runner construction. Add a unit test
  verifying the default prompt's rules are extracted and passed to the scorer.
  Add a unit test verifying a non-default prompt with valid layers has its
  rules passed to the scorer. Verify all tests pass.

## 5. Thread rules through `classify_input`

- [x] 5.1 In
  `src/dynamic_prompt_core/application/use_cases/classify_input/classify_input.py`,
  add `rules: Sequence[str]` as a parameter to `classify_input()`. Forward it
  to `deps.scorer.score(text, candidates, rules)`. `ClassifyInputDeps` is
  unchanged. Verify with `mypy` that the function type-checks.
- [x] 5.2 Add a unit test verifying `classify_input` forwards the supplied
  rules to the scorer (use a fake scorer that records its arguments). Verify
  the test passes.

## 6. Update all test fakes, mocks, and call sites

- [x] 6.1 Update every test fake/mock implementing `CandidateScorer` to accept
  the `rules` parameter in their `score()` method. Search for all test classes
  implementing the `CandidateScorer` protocol (e.g., `FakeCandidateScorer`,
  recording scorers, mock scorers in `conftest.py` files). Verify each fake
  accepts `rules` without error.
- [x] 6.2 Update every test call site of `scorer.score(...)` to pass a `rules`
  argument. Use the default prompt's rules
  (`CLASSIFICATION_PROMPT_V0.layers.rules`) where tests previously relied on
  hardcoded behavior. Verify by running
  `pytest tests/unit/candidate_scoring/ tests/unit/run_baseline/ tests/unit/run_cycle/ -x --tb=short`
  and confirming no `TypeError` from missing `rules`.

## 7. Integration tests for rule propagation

- [x] 7.1 Add an integration test verifying that active prompt rules propagate
  through `BaselineRunner` to the model adapter. Use a recording model adapter
  or recording prompt builder that captures the actual prompt text. Assert the
  rules from `PromptArtifact.layers.rules` appear in the captured prompt.
  Verify the test passes.
- [x] 7.2 Add an integration test verifying that different prompt versions
  produce different rule content in the model input. Construct two
  `PromptArtifact`s with different rules, run each through the runner, and
  assert the captured prompts contain their respective rules and not each
  other's. Verify the test passes.
- [x] 7.3 Add an integration test verifying that `run_cycle`'s
  `run_active_on_dev` propagates `state.active_version.layers.rules` to the
  model input. Use a fake scorer that records rules. Verify the test passes.
- [x] 7.4 Add an integration test verifying that `classify_input` propagates
  the supplied rules to the model input. Use a recording fake scorer. Verify
  the test passes.

## 8. Regression tests

- [x] 8.1 Add a regression test verifying the default classification prompt
  produces a scoring prompt containing its default rules and the pipeline
  completes successfully. Verify the test passes.
- [x] 8.2 Add a regression test verifying existing batch scoring behavior is
  preserved: one forward call for N candidates, judgments in candidate order,
  scores sum to 1.0. Verify the test passes.
- [x] 8.3 Add a regression test verifying that an empty or missing rule set
  for a non-default prompt does not silently become the default rules. Assert
  the scorer receives exactly the supplied (possibly empty) rules. Verify the
  test passes.
- [x] 8.4 Update or replace the existing contract test
  (`tests/unit/candidate_scoring/test_scoring_seams.py`) that verified
  `_SEMANTIC_PREFIX` corresponds to V0 rules. Replace it with a test verifying
  the default prompt's rules produce the expected prompt structure through
  `build_prefix(text, rules)`. Verify the test passes.

## 9. Full validation

- [x] 9.1 Run `pytest` and verify the full test suite passes, including
  contract tests and all scoring tests.
- [x] 9.2 Run `mypy` and `ruff check` and verify they pass with no
  regressions.
- [x] 9.3 Run `lint-imports` and verify all import-linter contracts pass,
  including `application-ports-isolated` and the domain stdlib-only allowlist.
  Confirm no infrastructure module is imported by the application port.
- [x] 9.4 Run
  `openspec validate --changes pass-active-rules-to-candidate-scorer`
  and verify the spec deltas are valid: every requirement has at least one
  scenario, MODIFIED requirements use `## MODIFIED Requirements` with full
  updated content, and ADDED requirements use `## ADDED Requirements`.
- [x] 9.5 Verify the acceptance criteria: inspect every production call site
  of `CandidateScorer.score()` and confirm each supplies rules; confirm
  `ScoringPromptBuilder` does not contain `CLASSIFICATION_PROMPT_V0` or
  hardcoded default rules; confirm missing layers on a non-default prompt
  raises; confirm all three backends share the same `ScoringPromptBuilder`.
