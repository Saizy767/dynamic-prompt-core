# Tasks

## 1. Extract semantic criteria and author semantic prefix

- [x] 1.1 Review `CLASSIFICATION_PROMPT_V0` in
  `src/dynamic_prompt_core/domain/prompts/fixed.py` by content, not by layer.
  Classify each piece of content as: (a) classification-task semantics to
  preserve, (b) output-format instructions to exclude, or (c) selection-oriented
  task instructions to rewrite into candidate-evaluation semantics. Document the
  classification in a table:

  | Legacy content | Judgment Prompt treatment |
  |---|---|
  | Classification meaning (what the task represents) | Preserve |
  | Relevance criteria (what makes a candidate relevant) | Preserve |
  | Ambiguity rules (how to handle ambiguous cases) | Preserve/rewrite |
  | "Choose one class" / "classify into one of two classes" | Rewrite as candidate evaluation |
  | JSON schema / "Reply with a single JSON object" | Remove |
  | "confidence" / "decision" fields | Remove |
  | Fallback JSON response | Remove |

  Verify the table covers all content in the generative prompt with nothing left
  unclassified.
- [x] 1.2 Define a `_SEMANTIC_PREFIX` string constant in
  `src/dynamic_prompt_core/infrastructure/llm/scoring/prompt_builder.py`
  containing the extracted and rewritten semantic classification criteria. The
  constant SHALL include task framing rewritten as candidate evaluation ("evaluate
  how well the candidate describes or matches the input") and the classification
  rules adapted to candidate evaluation. The constant SHALL NOT contain JSON
  instructions, confidence value instructions, response-format requirements, or
  structured-output schema references. Verify by inspecting the constant for
  absence of the strings "JSON", "confidence", "decision", and "Reply with".

## 2. Update ScoringPromptBuilder

- [x] 2.1 Update `ScoringPromptBuilder.build_prefix` in
  `src/dynamic_prompt_core/infrastructure/llm/scoring/prompt_builder.py` to
  return `{_SEMANTIC_PREFIX}{text}\n{_CANDIDATE_LABEL}` instead of
  `{text}\n{_CANDIDATE_LABEL}`. The `build_candidate` and `build` methods SHALL
  remain unchanged. Verify `build_prefix("test")` starts with `_SEMANTIC_PREFIX`
  and ends with `"Candidate: "`.
- [x] 2.2 Update the `ScoringPromptBuilder` docstring to reflect the enriched
  prompt format `{semantic_prefix}{input}\nCandidate: {candidate}`. Verify the
  docstring accurately describes the new prompt shape.

## 3. Add prompt construction tests

- [x] 3.1 Add tests to `TestScoringPromptBuilder` in
  `tests/unit/candidate_scoring/test_scoring_seams.py` verifying that the prefix
  contains semantic classification criteria (e.g., task framing about evaluating
  a candidate against the input) and that the prefix still ends with
  `"Candidate: "`. Verify the tests pass.
- [x] 3.2 Add tests verifying that the full prompt does not contain
  generation-protocol instructions: assert that `"JSON"`, `"confidence"`,
  `"decision"`, and `"Reply with"` do not appear in `build_prefix` or `build`
  output. Verify the tests pass.
- [x] 3.3 Add tests verifying candidate text preservation with the enriched
  prefix: candidates with leading/trailing spaces, mixed case, punctuation, and
  Unicode are preserved verbatim in `build_candidate` and in `build`. Verify the
  tests pass.
- [x] 3.4 Add a test verifying the prefix/candidate boundary is maintained:
  `build(text, candidate) == build_prefix(text) + build_candidate(candidate)`.
  Verify the test passes.
- [x] 3.5 Add a test verifying the candidate continuation is the final textual
  component of the prompt: no text appears after `build_candidate(candidate)`
  in `build(text, candidate)`, and no semantic instructions appear between the
  candidate label and the candidate value. Verify the test passes.

## 4. Add semantic migration tests

- [x] 4.1 Add a test class `TestJudgmentPromptSemantics` in
  `tests/unit/candidate_scoring/test_scoring_seams.py` (or a new
  `test_judgment_prompt.py` file in the same directory) that verifies each
  identified semantic requirement from `CLASSIFICATION_PROMPT_V0` is represented
  in the judgment prompt. Tests should assert semantic prompt sections or stable
  authored requirements (e.g., "the prompt contains a task-framing section that
  describes candidate evaluation", "the prompt contains relevance criteria based
  on text content") rather than requiring literal phrase preservation, because
  rewriting necessarily changes some phrases (e.g., "classify into one of two
  classes" becomes "evaluate how well the candidate matches the input"). Verify
  the tests pass.
- [x] 4.2 Add tests verifying that generation-protocol phrases from
  `CLASSIFICATION_PROMPT_V0`'s `output_contract` and `fallback` are absent from
  the judgment prompt: assert that "Reply with a single JSON object", "decision",
  "confidence", and "No markdown" do not appear in the prompt. Verify the tests
  pass.
- [x] 4.3 Add a test verifying the prompt is candidate-specific: constructing
  prompts for candidates `["A", "B", "C"]` with the same input produces three
  distinct prompts, each containing exactly one candidate, and no prompt
  contains another candidate's value. Verify the test passes.

## 5. Validate batch compatibility and existing tests

- [x] 5.1 Run the existing batch scoring tests
  (`tests/unit/candidate_scoring/test_batch_scoring.py` and
  `test_batch_integration.py`) and verify they pass unchanged. The enriched
  prefix must not break batch tokenization, boundary detection, or score
  computation. Confirm that `LLMLogitCandidateScorer` still calls
  `build_prefix` and `build_candidate` without modification.
- [x] 5.2 Run the full test suite (`pytest`) and verify all tests pass,
  including existing `TestScoringPromptBuilder` tests, `LLMLogitCandidateScorer`
  tests, and generative classification tests. Confirm that no existing test
  required modification due to the prompt change (existing tests that assert
  `prefix.endswith("Candidate: ")` or `candidate.value` verbatim should still
  pass).

## 6. Validate lint, typecheck, and specs

- [x] 6.1 Run `ruff check` and `mypy` and verify they pass with no new
  violations. Inspect `git diff --stat` and confirm the only modified source
  file is `src/dynamic_prompt_core/infrastructure/llm/scoring/prompt_builder.py`
  and the only modified test file is under
  `tests/unit/candidate_scoring/`.
- [x] 6.2 Run `lint-imports` and verify all import-linter contracts pass.
  Confirm that no new imports were introduced in `prompt_builder.py` (the
  semantic prefix is a string constant, not an import from `domain/prompts/`).
- [x] 6.3 Run `openspec validate --changes judgment-prompt` and verify the spec
  deltas (`judgment-prompt` and `llm-logit-scorer`) are valid: every requirement
  has at least one scenario, the new capability has a `## Purpose` section, and
  the modified `llm-logit-scorer` delta uses `## MODIFIED Requirements` with no
  `## Purpose`.
- [x] 6.4 Verify the generative classification path is unchanged: inspect
  `git diff` and confirm that `src/dynamic_prompt_core/domain/prompts/fixed.py`,
  `src/dynamic_prompt_core/infrastructure/llm/client.py`,
  `src/dynamic_prompt_core/application/schemas/classification.py`, and
  `src/dynamic_prompt_core/application/use_cases/run_baseline/runner.py` are not
  modified.
