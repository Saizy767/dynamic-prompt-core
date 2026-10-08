# Design

## Context

The candidate-scoring flow and the initial `LLMLogitCandidateScorer` are
implemented as separate application and infrastructure concerns. The current
scorer evaluates candidates sequentially:

`ClassifyInput` → `CandidateScorer.score(text, candidates)` →
`LLMLogitCandidateScorer` → one model inference per candidate.

This preserves the application contract, but it requires a separate model
forward pass for every candidate. For an input with `N` candidates, scoring
currently performs `N` independent model inferences (see
`candidate_scorer.py:62` — the `score` method loops over candidates calling
`_score_one`).

The current infrastructure seams were intentionally designed to allow batching
later (the `llm-logit-scorer` spec's "Multi-candidate API is an application
contract not a model batch" requirement explicitly says the implementation MAY
batch). This change introduces actual batch scoring inside
`infrastructure/llm/scoring/` without changing the `CandidateScorer`
application port or exposing batching mechanics to the domain.

## Goals / Non-Goals

**Goals:**

- Replace sequential per-candidate model inference with batched model inference.
- Keep the existing `CandidateScorer.score(text, candidates)` application
  contract unchanged.
- Batch all candidate prompts belonging to one scoring invocation.
- Preserve the exact existing candidate ordering in the returned `Judgment`
  list.
- Preserve the existing mean candidate-token log-probability algorithm.
- Support candidates with different token lengths through padding and
  per-example candidate-token boundaries.
- Keep tokenization, padding, attention masks, logits, and batch execution
  entirely inside infrastructure.
- Keep the async application-facing scorer contract unchanged.
- Ensure a failure in batch inference is translated through the existing
  scoring error boundary.
- Keep a straightforward path for models that do not support the optimized
  batch adapter.

**Non-Goals:**

- Change `CandidateScorer` or introduce a new application-level
  `BatchCandidateScorer` port.
- Score multiple independent input texts in one invocation.
- Introduce prefix/KV-cache reuse.
- Introduce provider-specific batch APIs.
- Introduce dynamic batching across concurrent application requests.
- Change the scoring algorithm or its score semantics.
- Introduce score calibration or probability normalization.
- Change `ClassificationPolicy`.
- Modify the existing generative classification path.
- Optimize GPU memory management beyond what is required for basic batching.

## Decisions

### Decision: Batch scoring is an infrastructure optimization behind the existing `CandidateScorer` port

The application continues to call `scorer.score(text, candidates)` and receives
the same ordered collection of `Judgment` objects. The application does not know
whether the scorer evaluates candidates sequentially, in one batch, in several
batches, or falls back to sequential execution. Batching is an implementation
detail of `LLMLogitCandidateScorer`. This preserves the dependency direction
`application → CandidateScorer → infrastructure` and prevents model execution
mechanics from leaking into the application layer.

- **Alternative considered:** introduce a `BatchCandidateScorer` application
  port. Rejected: the existing `CandidateScorer` already accepts multiple
  candidates. Introducing a second port would expose an infrastructure
  optimization as a domain/application capability without semantic value.

### Decision: One scoring invocation represents one input and a candidate batch

The batch consists of all candidates passed to `score(text, candidates)`. The
scorer constructs one model input per candidate (`"{input}\nCandidate:
{candidate}"`), tokenizes them as a batch, and evaluates them in one batched
forward pass. The initial implementation does not combine different `text`
values into the same batch.

- **Reason:** the current application contract represents scoring for one input
  against a candidate set. General-purpose request batching is a separate
  performance problem and is not required to remove the current per-candidate
  inference overhead.

### Decision: Introduce a batched model seam in infrastructure

The existing synchronous model abstraction (`CausalLanguageModel.forward`) is
extended with a batch-capable seam. Conceptually:

```python
class BatchedCausalLanguageModel(Protocol):
    def forward_batch(
        self,
        input_ids,
        attention_mask=None,
    ) -> BatchedLogits:
        ...
```

`BatchedLogits` has the semantic shape `[batch_size, sequence_length,
vocabulary_size]`, where item `i` corresponds to input batch item `i`. The
concrete tensor type (e.g. `torch.Tensor`) remains infrastructure-specific;
only the semantic shape and item mapping are fixed.

The batch model seam is not exposed through `application/ports/`. The concrete
model adapter translates the batch representation into the underlying framework
call. The adapter continues to keep the model in evaluation mode, disable
gradient calculation, and preserve the externally managed model lifecycle. The
batch seam preserves the mapping `batch item i → logits for batch item i`
without reordering.

A separate `BatchedCausalLanguageModel` protocol is used rather than adding
`forward_batch` to the existing `CausalLanguageModel` protocol because the
existing single-sequence adapters (e.g. `TorchCausalLMAdapter`) must continue
to work without implementing a batch API. The scorer depends on the batch
protocol; a sequential compatibility adapter can satisfy it by delegating to
`forward` per item.

- **Alternative considered:** unify `forward` and `forward_batch` into one
  `CausalLanguageModel` protocol. Rejected: this would force every existing
  adapter to implement `forward_batch` even if it only supports
  single-sequence inference. The separate protocol keeps the batch capability
  opt-in at the adapter level while the scorer always requires it.

### Decision: Tokenization produces a padded batch with explicit attention masks

Each candidate prompt is tokenized independently (encoding prefix and candidate
separately, as the existing `Tokenization boundary determination` requirement
mandates) and then assembled into a batch. Because candidate prompts may have
different token lengths, the tokenizer adapter pads them to a common sequence
length. The resulting batch contains at least `input_ids` and
`attention_mask`. Padding is infrastructure-only — the application and domain
never see token IDs, padding IDs, attention masks, tensor shapes, or
tokenizer-specific batch structures. The tokenizer adapter preserves the exact
candidate prompt content: no `.strip()`, lowercasing, case folding, or
whitespace collapsing is introduced. `Candidate.value` remains unchanged before
tokenization.

### Decision: Padding direction is right padding

The tokenizer adapter SHALL use right padding: padding tokens are appended
after the real sequence content. This keeps real tokens at positions `0 ..
unpadded_length - 1`, so candidate-token positions are calculated from each
batch item's unpadded sequence length, not from the padded batch length. This
is critical because for a causal LM, logit positions directly determine which
token predictions are extracted — an ambiguous padding side would make
candidate-token positions depend on batch composition, silently corrupting
scores.

The padding direction is an explicit contract of the tokenizer adapter, not an
emergent property of the tokenizer library. If a specific model requires left
padding (e.g. some decoder-only models for generation), that SHALL be an
explicit contract of that model's adapter, not a silent behavior of the
tokenizer.

- **Alternative considered:** left padding. Rejected for the initial
  implementation: right padding keeps real tokens at the start of the
  sequence, making position calculation straightforward. Left padding may be
  required for specific models and can be introduced as an explicit adapter
  contract if needed.

### Decision: Candidate-token boundaries are tracked per batch item

The scorer must not infer candidate-token positions from the padded batch
length. For every candidate, the tokenizer adapter produces: the complete
model input token sequence, the exact positions corresponding to candidate
continuation tokens, and the candidate token IDs at those positions. The
candidate score is computed only from the candidate continuation tokens for
that batch item. Padding tokens never contribute to the score. This is required
because prompts have different total lengths, candidates have different token
lengths, and candidate scoring must remain independent of batch composition.

The correctness invariant is: **the candidate token IDs used for scoring MUST
be exactly the token IDs occupying the candidate continuation positions in the
model input sequence.** This invariant is authoritative — it is the
correctness contract, not the encoding mechanism. How the boundary is
determined is tokenizer-specific and remains inside infrastructure. The
implementation MUST NOT assume that separately encoding the prefix and the
candidate (`encode(prefix) + encode(candidate)`) is equivalent to encoding the
concatenated prompt (`encode(prefix + candidate)`), because BPE/SentencePiece
tokenizers may merge or split tokens across the boundary. If a concrete
tokenizer adapter has a proven-safe separate-encoding strategy, it MAY use it;
otherwise it SHALL determine the boundary by inspecting the full encoded
sequence. The entire batch-optimization must remain semantically equivalent to
the sequential implementation, and this invariant is what makes that
equivalence hold.

### Decision: Causal logit alignment remains explicit in the batched implementation

The model follows causal next-token prediction semantics. For a token sequence
`[p0, p1, p2, c0, c1, c2]`, the relevant predictions are `logit[p2] → c0`,
`logit[c0] → c1`, `logit[c1] → c2`. For every batch item, the scorer computes
`log_softmax(logits[position_of_previous_token])[candidate_token_id]` for each
candidate continuation token. The candidate score is the arithmetic mean of
these token log-probabilities. Batching must not alter this alignment. The
implementation may use tensorized operations to calculate the values for all
batch items simultaneously, but the semantic definition remains identical to
the sequential implementation.

### Decision: Mean candidate-token log-probability remains the scoring algorithm

Batch scoring does not change the score definition. For candidate `c` with
continuation tokens `t1, ..., tn` and log-probabilities `l1, ..., ln`, the
score remains `score(c) = (l1 + l2 + ... + ln) / n`. This remains a ranking
signal. The implementation does not convert scores into probabilities,
normalize scores across candidates, apply softmax across candidates, calibrate
scores, or introduce temperature scaling. The only change is that multiple
candidates are evaluated simultaneously.

### Decision: Batch output is mapped back to the original candidate order

The candidate list order is semantically observable because the current
`ArgmaxClassificationPolicy` uses first-in-order tie-breaking. If the input is
`[candidate_a, candidate_b, candidate_c]`, the scorer returns
`[Judgment(candidate_a, ...), Judgment(candidate_b, ...),
Judgment(candidate_c, ...)]` even if the infrastructure internally changes
tensor layout or uses implementation-specific indexing. Batching must never
reorder candidates. The strengthened invariant is: for every input candidate
at index `i`, the returned `Judgment` at index `i` MUST refer to that same
candidate (`judgments[i].candidate == candidates[i]`). This is stronger than
"order preserved" and catches tensor indexing errors that could silently swap
candidates. This guarantees that switching between sequential and batch
implementations does not change classification behavior except for numerical
differences caused by the underlying computation.

### Decision: The scorer performs one asynchronous blocking boundary per batch

The current scorer is asynchronous while local model inference is synchronous.
The batch implementation continues to ensure that synchronous model execution
does not block the event loop. Instead of calling the blocking model once per
candidate, the scorer constructs the complete batch and performs the batch
model call behind a single non-blocking execution boundary. The exact executor
mechanism is an infrastructure implementation detail. The architectural
requirement is: synchronous model execution MUST NOT block the application's
async event loop. `asyncio.to_thread` may remain the initial implementation,
but it is not itself the architectural contract.

### Decision: Batch failure has the same error boundary as sequential scoring

The application continues to receive scoring-layer errors rather than raw
framework/provider exceptions. Tokenizer, tensor construction, model execution,
and output-shape failures are translated into the existing
`CandidateScoringError` (with the original as `__cause__`). No batch-specific
exception type is introduced unless implementation proves that callers need to
distinguish batch failures from ordinary scoring failures. A batch failure must
not silently produce partial judgments — if the batch cannot be evaluated
reliably, the entire scoring invocation fails.

**Pre-existing note:** The `llm-logit-scorer` spec states that
`CandidateScoringError` SHALL reside in `application/errors/scoring.py`, but
the current implementation has it in `domain/errors/scoring.py`. This is a
pre-existing discrepancy, not introduced by this change. The batch scoring
implementation uses `CandidateScoringError` wherever it currently lives and
translates infrastructure exceptions (`LLMScoringError`, torch, transformers)
into it. Resolving the location discrepancy is a separate concern; this change
adds a verification task to flag it.

### Decision: No partial batch results

A single `score(text, candidates)` invocation is atomic from the application's
perspective. If one candidate produces an invalid score or the model returns an
invalid batch shape, the scorer does not return scores for the remaining
candidates — the entire scoring operation raises a scoring error. This keeps
the application contract simple and avoids ambiguous partial classifications.

### Decision: Batch size is initially equal to the candidate count

The first implementation sends all candidates from one scoring invocation
through one model batch (one `forward_batch` call). There is no configurable
`max_batch_size` in the application contract. A future infrastructure
implementation may split a scoring batch into multiple chunks (model batches)
for memory constraints; such chunking must remain invisible to the application.
The introduction of configurable batch sizing is deferred until there is
evidence that candidate sets can exceed practical model-memory limits.

### Decision: Scoring batch, model batch, and chunk are distinct concepts

A **scoring batch** is the logical group of all candidates passed to one
`score(text, candidates)` invocation. A **model batch** is the concrete set of
items passed to one `forward_batch` call. A **chunk** is a subset of a scoring
batch passed as one model batch when the scoring batch is split for memory
constraints. The initial implementation uses one model batch per scoring batch
(no chunking). Separating these concepts eliminates the potential contradiction
between "one batch" and future chunking: the scoring batch is always the full
candidate set, while the number of model batches is an infrastructure execution
decision.

### Decision: Sequential fallback is a temporary compatibility mechanism only

The preferred implementation path is `CandidateScorer → LLMLogitCandidateScorer
→ batch tokenizer → batched model adapter → batched logit scoring`. If a
concrete model adapter cannot support batching, that limitation is handled
inside infrastructure. A compatibility adapter may temporarily implement batch
execution by delegating to sequential execution (`forward_batch(items)` calls
`forward(item)` for each item). This is permitted ONLY as a temporary
transitional mechanism — it preserves the semantic contract but does not solve
the performance problem. The production local-model adapter used by
`LLMLogitCandidateScorer` MUST use native tensor batching; sequential
delegation is permitted only for explicitly supported compatibility adapters.
The application must not branch on whether batching is supported.

### Decision: No additional top-level `batching/` package

The existing structure under `infrastructure/llm/scoring/` is extended rather
than replaced. Batch-specific behavior belongs in the existing scoring seams
because batching is an execution strategy for candidate scoring, not a separate
domain capability. If the implementation becomes sufficiently complex, internal
helpers may be extracted later without changing the public architecture.

## Risks / Trade-offs

- **[Memory usage increases]** A batch requires holding multiple sequences and
  their logits simultaneously.
  → Mitigation: initially batch one candidate set; introduce
  infrastructure-level chunking if real workloads demonstrate memory pressure.

- **[Padding creates computation overhead]** Candidates have different sequence
  lengths, so shorter prompts may be padded to the longest prompt.
  → Mitigation: keep the first implementation simple; optimize bucketing only
  if profiling demonstrates a meaningful benefit.

- **[Batch and sequential floating-point results may differ slightly]**
  → Mitigation: compare within an explicit numerical tolerance and avoid tests
  requiring bit-for-bit equality.

- **[Batching does not automatically provide prefix reuse]**
  → Mitigation: prefix/KV-cache optimization remains a separate future change.
  This specification only removes the per-candidate model-call overhead.

- **[Large candidate sets may exceed model memory]**
  → Mitigation: keep the infrastructure boundary capable of internal batch
  chunking without changing the application port.

- **[The model adapter may not support batching uniformly]**
  → Mitigation: keep batch capability behind infrastructure and allow a
  sequential compatibility implementation where required.

## Migration Plan

This change is additive.

1. **Introduce batch-capable tokenizer seam** — Extend the tokenizer
   infrastructure so multiple prompts can be encoded together while preserving
   per-item token boundaries. No application changes.
2. **Introduce batch-capable model seam** — Add the infrastructure abstraction
   for batched model inference. The existing single-input model seam remains
   available.
3. **Implement batched logit scoring** — Extend the logit scoring implementation
   to calculate candidate-token log-probabilities for all batch items while
   preserving the existing algorithm.
4. **Switch `LLMLogitCandidateScorer`** — Change the scorer from per-candidate
   `tokenize → forward → score` to `build all prompts → tokenize batch →
   forward batch → score batch → reconstruct ordered judgments`.
5. **Add equivalence tests** — Verify that batch scoring produces the same
   candidate-to-score semantics as sequential scoring.
6. **Add performance benchmark** — Measure representative candidate sets before
   and after batching (total latency, model forward-call count, throughput,
   peak memory where practical).
7. **Keep application flow unchanged** — `ClassifyInput`, `CandidateScorer`,
   `Classification`, `Judgment`, and `ClassificationPolicy` require no semantic
   changes. The generative classification path remains untouched.

**Rollback:** Rollback is limited to infrastructure. If batch scoring
introduces correctness or performance problems, `LLMLogitCandidateScorer` can
temporarily revert to sequential execution while retaining the existing
application contract. No domain or application rollback is required because the
public scoring API does not change.

## Open Questions

- Should infrastructure expose a configurable maximum batch size once real
  candidate-set sizes are known?
- Should batch chunking be introduced immediately or only after memory
  profiling?
- Should the tokenizer adapter expose an explicit batch result type, or should
  the concrete tokenizer library's batch structure remain
  infrastructure-internal?
- Should future optimization group candidates by prompt length to reduce
  padding?
- At what candidate-set size does batching stop providing a meaningful latency
  improvement?
- Should prefix/KV-cache reuse be considered as the next optimization after
  basic batching?

These questions do not block the initial batch-scoring implementation.
