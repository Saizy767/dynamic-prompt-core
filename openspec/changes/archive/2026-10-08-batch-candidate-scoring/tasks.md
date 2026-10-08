# Tasks

## 1. Batch-capable tokenizer seam

- [x] 1.1 Add a `BatchTokenizerAdapter` protocol and a batch result type to
  `infrastructure/llm/scoring/tokenizer_adapter.py` that encodes multiple
  prompts together into a padded batch with `input_ids`, `attention_mask`, and
  per-item `prefix_token_count` / `candidate_token_ids` boundaries. For each
  prompt, the adapter produces the complete model input token sequence, the
  exact positions of candidate continuation tokens, and the candidate token IDs
  at those positions. The correctness invariant: candidate token IDs used for
  scoring MUST be exactly the token IDs occupying the candidate continuation
  positions in the model input sequence. The implementation MUST NOT assume
  that `encode(prefix) + encode(candidate)` equals `encode(prefix + candidate)`
  — if the tokenizer adapter has a proven-safe separate-encoding strategy it
  MAY use it, otherwise it SHALL determine the boundary from the full encoded
  sequence. The tokenizer SHALL use right padding (padding tokens appended after
  real content); candidate-token positions are calculated from each item's
  unpadded sequence length. No stripping, lowercasing, or whitespace collapsing
  is introduced. Verify with a unit test using a deterministic stub tokenizer
  that the batch contains one item per prompt, per-item boundaries are recorded,
  right padding is applied, and the attention mask covers non-padding positions.
- [x] 1.2 Add a `HuggingFaceBatchTokenizerAdapter` concrete implementation
  wrapping `transformers.AutoTokenizer` batch encoding (`encode_plus` /
  `pad`) that produces padded `input_ids` and `attention_mask` tensors. Verify
  with a unit test using a stub that the concrete adapter satisfies the
  `BatchTokenizerAdapter` protocol and preserves prompt content verbatim.

## 2. Batch-capable model seam

- [x] 2.1 Add a `BatchedCausalLanguageModel` protocol with a `forward_batch`
  method and a `BatchedLogits` result type to
  `infrastructure/llm/scoring/model_adapter.py`. `BatchedLogits` has the
  semantic shape `[batch_size, sequence_length, vocabulary_size]` where item
  `i` corresponds to input batch item `i`; the concrete tensor type remains
  infrastructure-specific. The seam accepts padded `input_ids` and
  `attention_mask` and returns per-item logits preserving the
  `batch item i → logits i` mapping without reordering. Verify with a unit test
  using a deterministic stub model that `forward_batch` returns one logits entry
  per input item in input order and the result shape matches `[B, S, V]`.
- [x] 2.2 Add a `TorchBatchedCausalLMAdapter` concrete implementation that runs
  the PyTorch model in `eval()` mode under `torch.no_grad()` with batched input
  tensors and attention masks, returning per-item `SequenceLogits`. Verify with
  a unit test using a stub that the concrete adapter satisfies the
  `BatchedCausalLanguageModel` protocol and preserves batch item order.

## 3. Batched logit scoring

- [x] 3.1 Extend `LogitScorer` in `infrastructure/llm/scoring/logit_scorer.py`
  with a `score_batch` method that accepts per-item `prefix_token_counts`,
  `candidate_token_ids`, and `BatchedLogits`, and returns one score per batch
  item using the existing causal alignment and mean log-probability algorithm.
  Padding tokens must never contribute to any item's score. Verify with a unit
  test using deterministic fake logits that per-item alignment matches the
  sequential `score` method for multi-token candidates and variable lengths.
- [x] 3.2 Add numerical validation to `score_batch`: reject `NaN`, positive
  infinity, and negative infinity per batch item. If any item produces an
  invalid score, raise `LLMScoringError` for the entire batch (no partial
  results). Verify with a unit test that one invalid score among valid items
  raises `LLMScoringError` and returns no scores.

## 4. Switch LLMLogitCandidateScorer to batch execution

- [x] 4.1 Refactor `LLMLogitCandidateScorer.score` in
  `infrastructure/llm/scoring/candidate_scorer.py` to build all candidate
  prompts, tokenize as a batch, run one `forward_batch` call behind a single
  `asyncio.to_thread` boundary, score all batch items with `score_batch`, and
  reconstruct ordered `Judgment` objects in the original candidate order. The
  scorer constructor accepts the batch tokenizer and batched model seams. Verify
  with a unit test using stub seams that `N` candidates produce exactly `1`
  `forward_batch` call and judgments are in input order.
- [x] 4.2 Preserve the existing error translation in the batch path: translate
  `LLMScoringError` and raw infrastructure exceptions into
  `CandidateScoringError` with the original as `__cause__`. A batch failure
  must not return a partial judgment list. Verify with a unit test using a
  failing batch model that `CandidateScoringError` is raised and no partial
  result is returned.
- [x] 4.3 Add a sequential compatibility adapter that implements `forward_batch`
  by delegating to `forward` per item, for model adapters that do not support
  native batching. This is a temporary compatibility mechanism only, not the
  target implementation. Verify with a unit test that the compatibility adapter
  produces the same scores as native batch scoring with a stub model.

## 5. Equivalence and ordering tests

- [x] 5.1 Add a unit test that runs the same input and candidate set through
  both the sequential implementation and the batch implementation, comparing
  candidate-to-score mappings by candidate value within floating-point tolerance
  (`pytest.approx`). Cover variable-length candidates (`"tax"`, `"financial
  regulation"`, `"international financial regulation"`). Verify the test passes.
- [x] 5.2 Add a unit test that verifies batch output preserves exact candidate
  order for inputs like `[candidate_a, candidate_b, candidate_c]`, including
  when candidates have different token lengths. Verify the returned judgment
  order matches the input candidate order exactly AND that for every index `i`,
  `judgments[i].candidate == candidates[i]` (the strengthened invariant that
  catches tensor indexing errors).
- [x] 5.3 Add a unit test that verifies a single-candidate invocation still
  works through the batch path (batch of size 1) and produces the same score as
  the sequential path. Verify the test passes.
- [x] 5.4 Add a unit test that verifies padding does not change a candidate's
  score: score a short candidate alone, then score it in a batch with a much
  longer candidate, and verify the short candidate's score is the same in both
  cases (within `pytest.approx` tolerance). This test simultaneously validates
  padding, attention mask, candidate boundary, causal alignment, and logit
  indexing.

## 6. Integration and validation

- [x] 6.1 Add an opt-in real-model integration test (separate from the default
  suite, gated by a marker or environment variable) that verifies the tokenizer
  accepts a batch, the model accepts batched input, output dimensions match the
  input batch, multiple candidates are scored, scores are finite, and candidate
  ordering is preserved. Verify the test is skipped by default and runs when
  explicitly enabled.
- [x] 6.2 Run `openspec validate --change batch-candidate-scoring` and verify
  the spec delta is valid: every requirement has at least one scenario, and the
  modified `llm-logit-scorer` delta uses `## ADDED Requirements` with no
  `## Purpose`.
- [x] 6.3 Run `pytest tests/unit/candidate_scoring/` and verify all existing
  and new tests pass, confirming the batch implementation preserves the
  existing scoring contract.
- [x] 6.4 Run `mypy`, `ruff check`, and `lint-imports` and verify they pass,
  confirming no type, lint, or import-boundary regressions. Inspect `git diff
  --stat` and confirm only `infrastructure/llm/scoring/` source files and
  `tests/unit/candidate_scoring/` test files are modified — no `domain/`,
  `application/`, or `interfaces/` changes.
- [x] 6.5 Verify the `CandidateScoringError` error boundary: confirm that
  `CandidateScoringError` is imported from its current location
  (`domain/errors/scoring.py`) by the batch scorer, that infrastructure
  exceptions (`LLMScoringError`, torch, transformers) are translated into it
  with `__cause__` set, and that no raw infrastructure exception leaks to the
  application. Note: the `llm-logit-scorer` spec states this error SHOULD reside
  in `application/errors/scoring.py` but it currently lives in
  `domain/errors/scoring.py` — flag this pre-existing discrepancy in the
  implementation summary but do not move the error in this change.
