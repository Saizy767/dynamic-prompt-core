# Proposal

## Why

The `LLMLogitCandidateScorer` evaluates candidates sequentially: one model
forward pass per candidate. For an input with `N` candidates, scoring performs
`N` independent model inferences. The existing infrastructure seams were
intentionally designed to allow batching later; this change introduces actual
batched model inference so one `score()` invocation evaluates all candidates
through a single batched forward pass whenever the underlying adapter supports
it, without changing the `CandidateScorer` application port or exposing batching
mechanics to the domain.

## What Changes

- Replace sequential per-candidate model inference with batched model inference
  inside `LLMLogitCandidateScorer`: build all candidate prompts, tokenize as a
  padded batch, run one batched forward pass, score all batch items, and
  reconstruct ordered judgments.
- Introduce a batch-capable model seam (`forward_batch`) in infrastructure
  alongside the existing single-sequence `forward` seam.
- Extend the tokenizer adapter to encode multiple prompts as a padded batch with
  explicit attention masks while preserving per-item candidate-token boundaries.
- Extend the logit scorer to calculate candidate-token log-probabilities for all
  batch items simultaneously using the existing causal alignment and mean
  log-probability algorithm.
- Preserve exact candidate ordering in returned judgments, variable candidate
  token lengths via padding, and the existing error boundary (no partial results,
  `CandidateScoringError` translation).
- Keep the `CandidateScorer` port, `ClassifyInput`, `Classification`,
  `Judgment`, `ClassificationPolicy`, and the generative classification path
  unchanged. No model, tokenizer, tensor, or padding details leak into domain or
  application.

## Capabilities

### New Capabilities

(None. Batch scoring is an infrastructure optimization behind the existing
`CandidateScorer` port. No new application or domain capability is introduced.)

### Modified Capabilities

- `llm-logit-scorer`: Add requirements specifying that the standard
  implementation SHALL evaluate all candidates from one scoring invocation
  through a single batched model forward pass, with batch-capable tokenizer and
  model seams, padded batch construction with attention masks, per-item
  candidate-token boundary tracking, batch output ordering preservation, and
  batch failure semantics consistent with the existing atomic failure contract.

## Impact

- **Code**: Infrastructure-only changes to `infrastructure/llm/scoring/`
  (`candidate_scorer.py`, `tokenizer_adapter.py`, `model_adapter.py`,
  `logit_scorer.py`). No `domain/`, `application/`, or `interfaces/` changes.
  The generative classification path is untouched.
- **Specs**: `llm-logit-scorer` gains batch-inference requirements as a delta
  over the existing spec. No other capability spec changes.
- **Dependencies**: No new runtime dependencies. Batching uses the existing
  model and tokenizer libraries through new infrastructure seams.
- **Public API**: Unchanged. `CandidateScorer.score(text, candidates)` retains
  its signature and semantics. The application cannot observe whether batching
  occurred.
- **Migration**: Additive. The scorer switches from sequential to batch
  execution internally. Rollback is infrastructure-only: revert
  `LLMLogitCandidateScorer` to sequential execution without domain or
  application changes.
