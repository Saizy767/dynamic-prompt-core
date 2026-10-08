# Spec Delta

## ADDED Requirements

### Requirement: Batched model inference for one scoring invocation
The standard `LLMLogitCandidateScorer` implementation SHALL evaluate all
candidates from one `score(text, candidates)` invocation through a single
batched model forward pass rather than one forward pass per candidate. For
`N` candidates, the scorer SHALL perform exactly `1` batched model forward
call. The batch SHALL consist of all candidate prompts belonging to that
scoring invocation. The batch SHALL NOT combine candidates from different
`text` values. The application SHALL NOT know whether scoring used one batched
call, several calls, or sequential execution; batching remains an
infrastructure implementation detail behind the `CandidateScorer` port.

#### Scenario: One forward call for multiple candidates
- **WHEN** the scorer evaluates `N` candidates in one invocation
- **THEN** the model seam is invoked exactly once with a batch containing all `N` candidate prompts

#### Scenario: Batch contains all candidates from one invocation
- **WHEN** the scorer is invoked with candidates `[a, b, c]`
- **THEN** the batch forwarded to the model contains exactly the prompts for `a`, `b`, and `c`

#### Scenario: Batch does not combine different inputs
- **WHEN** the scorer is invoked with one `text` and a candidate set
- **THEN** the batch contains only prompts derived from that `text`

#### Scenario: Application unaware of batching
- **WHEN** the application inspects the scoring result
- **THEN** it cannot determine whether batching occurred

### Requirement: Batch-capable model seam
The infrastructure SHALL provide a batch-capable model seam that executes one
batched forward pass over multiple padded token sequences and returns logits
for every batch item. The batch seam SHALL preserve the mapping from batch
item `i` to the logits for batch item `i` without reordering. The batch seam
SHALL remain infrastructure-internal; the application SHALL NEVER import it.
The existing single-sequence model seam MAY remain available for non-batch
use. The batch seam SHALL keep the model in inference/evaluation mode, avoid
retaining gradients, and preserve the externally managed model lifecycle,
consistent with the existing inference-mode requirements.

The batch seam SHALL return a `BatchedLogits` result with the semantic shape
`[batch_size, sequence_length, vocabulary_size]`, where `item i` corresponds to
input batch item `i`. The concrete tensor type (e.g. `torch.Tensor`) SHALL
remain infrastructure-specific; only the semantic shape and item mapping are
fixed by this requirement.

#### Scenario: Batch seam returns logits per item
- **WHEN** the batch seam is invoked with `N` padded sequences
- **THEN** it returns logits for all `N` batch items

#### Scenario: BatchedLogits has semantic shape B x S x V
- **WHEN** the batch seam returns `BatchedLogits` for a batch of `B` sequences padded to length `S`
- **THEN** the result has shape `[B, S, V]` where `V` is the vocabulary size, and item `i` corresponds to input batch item `i`

#### Scenario: Batch item order preserved
- **WHEN** the batch seam returns logits
- **THEN** batch item `i` logits correspond to input batch item `i` without reordering

#### Scenario: Batch seam is infrastructure-internal
- **WHEN** the application is inspected
- **THEN** it never imports the batch-capable model seam

#### Scenario: Batch seam runs in inference mode
- **WHEN** the batch seam executes a batched forward pass
- **THEN** the model is in evaluation mode and gradients are not retained

### Requirement: Batch-capable tokenizer seam
The infrastructure SHALL provide a batch-capable tokenizer seam that encodes
multiple prompts together into a padded batch with explicit attention masks.
The batch tokenizer SHALL preserve the exact prompt content of each item: no
`.strip()`, lowercasing, case folding, or whitespace collapsing SHALL be
introduced as part of batch encoding. The `Candidate.value` SHALL remain
unchanged before tokenization. The batch tokenizer SHALL produce, at minimum,
`input_ids` and `attention_mask` for the batch. Padding SHALL be
infrastructure-only; the application and domain SHALL NEVER see token IDs,
padding IDs, attention masks, tensor shapes, or tokenizer-specific batch
structures. The batch tokenizer SHALL preserve per-item token boundaries so
the scorer can distinguish prefix tokens from candidate tokens for each batch
item independently.

#### Scenario: Multiple prompts encoded as padded batch
- **WHEN** the batch tokenizer encodes prompts of different token lengths
- **THEN** it produces one padded batch with `input_ids` and `attention_mask` covering all items

#### Scenario: Prompt content preserved verbatim
- **WHEN** the batch tokenizer encodes a prompt
- **THEN** the prompt content is preserved without stripping, lowercasing, or whitespace collapsing

#### Scenario: Padding is infrastructure-only
- **WHEN** the batch tokenizer produces a padded batch
- **THEN** token IDs, padding IDs, attention masks, and tensor shapes are not exposed to domain or application

#### Scenario: Per-item token boundaries recorded
- **WHEN** the batch tokenizer encodes a batch of prefix and candidate token sequences
- **THEN** the prefix token count and candidate token IDs are tracked independently for each batch item

### Requirement: Padding direction is explicit and fixed
The tokenizer adapter SHALL use right padding: padding tokens are appended
after the sequence content, so the real tokens occupy positions `0 ..
unpadded_length - 1` in the padded batch. Candidate-token positions SHALL be
calculated from the unpadded sequence length of each batch item, not from the
padded batch length. Padding tokens SHALL NEVER participate in candidate
scoring. The padding direction SHALL be an explicit contract of the tokenizer
adapter, not an emergent property of the tokenizer library. If a specific model
requires left padding, that SHALL be an explicit contract of that model's
adapter, not a silent behavior of the tokenizer.

#### Scenario: Right padding used
- **WHEN** the batch tokenizer pads sequences of different lengths to a common length
- **THEN** padding tokens are appended after the real tokens (right padding)

#### Scenario: Candidate positions from unpadded length
- **WHEN** candidate-token log-probabilities are extracted for a batch item
- **THEN** token positions are calculated from that item's unpadded sequence length, not the padded batch length

#### Scenario: Padding direction is explicit adapter contract
- **WHEN** the tokenizer adapter is inspected
- **THEN** the padding direction is explicitly documented, not inherited silently from the tokenizer library

### Requirement: Candidate-token boundaries tracked per batch item
The scorer SHALL NOT infer candidate-token positions from the padded batch
length. For every batch item, infrastructure SHALL retain the exact boundary
between the prefix tokens and the candidate continuation tokens, tracked
independently as `prefix_token_count` and `candidate_token_ids` per item. The
candidate score SHALL be computed only from the candidate continuation tokens
for that batch item. Padding tokens SHALL NEVER contribute to any candidate
score. This SHALL hold regardless of differing prompt lengths, differing
candidate token lengths, or batch composition.

The tokenizer adapter SHALL produce, for each candidate prompt: the complete
model input token sequence, the exact positions corresponding to candidate
continuation tokens, and the candidate token IDs at those positions. The
candidate token IDs used for scoring SHALL be exactly the token IDs occupying
the candidate continuation positions in the model input sequence. This
invariant is the correctness contract for batch scoring: it ensures that logits
are extracted at the positions that correspond to the candidate's actual tokens.

How the boundary is determined is tokenizer-specific and remains inside
infrastructure. The implementation SHALL NOT assume that separately encoding
the prefix and the candidate (`encode(prefix) + encode(candidate)`) is
equivalent to encoding the concatenated prompt (`encode(prefix + candidate)`),
because BPE/SentencePiece tokenizers may merge or split tokens across the
boundary. If a concrete tokenizer adapter has a proven-safe separate-encoding
strategy, it MAY use it; otherwise it SHALL determine the boundary by
inspecting the full encoded sequence.

#### Scenario: Boundaries tracked per item
- **WHEN** a batch contains candidates with different prefix and candidate token lengths
- **THEN** each batch item has its own `prefix_token_count` and `candidate_token_ids` independent of other items

#### Scenario: Padding tokens do not contribute to score
- **WHEN** a batch item is padded to match a longer sequence
- **THEN** padding tokens are excluded from that item's candidate score

#### Scenario: Score independent of batch composition
- **WHEN** the same candidate is scored in different batches with different co-candidates
- **THEN** its score is the same regardless of which other candidates are in the batch

#### Scenario: Candidate token IDs match continuation positions
- **WHEN** the scorer extracts logits for a batch item's candidate tokens
- **THEN** the token IDs used for scoring are exactly the token IDs at the candidate continuation positions in that item's model input sequence

#### Scenario: Padding does not change score
- **WHEN** a short candidate is scored alone and then scored in a batch with a much longer candidate
- **THEN** the short candidate's score is the same in both cases (within floating-point tolerance)

### Requirement: Batched causal logit alignment
The causal logit alignment SHALL remain identical to the sequential
implementation for every batch item. For each batch item, given prefix tokens
`p_0 ... p_m` and candidate tokens `c_0 ... c_{n-1}`, the candidate-token
log-probabilities SHALL be `log_softmax(logits[m])[c_0_id]`,
`log_softmax(logits[m+1])[c_1_id]`, ..., `log_softmax(logits[m+n-2])[c_{n-1}_id]`.
The implementation MAY use tensorized operations to calculate values for all
batch items simultaneously, but the semantic definition SHALL remain identical
to the sequential implementation. Batching SHALL NOT alter the alignment.

#### Scenario: Per-item alignment matches sequential
- **WHEN** a batch item's logits are scored
- **THEN** the causal alignment is identical to scoring that item alone sequentially

#### Scenario: Multi-token candidates aligned correctly in batch
- **WHEN** a batch item has a multi-token candidate
- **THEN** each candidate token's log-probability is read from the preceding position's logit distribution

### Requirement: Batched score algorithm equivalence
The mean candidate-token log-probability scoring algorithm SHALL remain
unchanged. For each batch item, the score SHALL be the arithmetic mean of the
candidate-token log-probabilities. The implementation SHALL NOT convert scores
into probabilities, normalize scores across candidates, apply softmax across
candidates, calibrate scores, or introduce temperature scaling. The only change
from sequential scoring SHALL be that multiple candidates are evaluated
simultaneously. Batch and sequential implementations SHALL produce equivalent
candidate-to-score mappings within floating-point tolerance.

#### Scenario: Mean log-probability per batch item
- **WHEN** a batch item with candidate-token log-probabilities `l1, ..., ln` is scored
- **THEN** its score is `(l1 + ... + ln) / n`

#### Scenario: No cross-candidate normalization
- **WHEN** multiple candidates are scored in one batch
- **THEN** scores are not normalized, softmaxed, or calibrated across candidates

#### Scenario: Batch equivalent to sequential
- **WHEN** the same input and candidate set are scored with the batch implementation and the sequential implementation
- **THEN** the candidate-to-score mappings are equivalent within floating-point tolerance

### Requirement: Batch output preserves candidate order
The scorer SHALL return judgments in the same order as the supplied candidates.
For every input candidate at index `i`, the returned `Judgment` at index `i`
SHALL refer to that same candidate (`judgments[i].candidate == candidates[i]`).
If the input is `[candidate_a, candidate_b, candidate_c]`, the scorer SHALL
return `[Judgment(candidate_a, ...), Judgment(candidate_b, ...),
Judgment(candidate_c, ...)]` even if the infrastructure internally changes
tensor layout or uses implementation-specific indexing. Batching SHALL NEVER
reorder candidates. This guarantees that switching between sequential and
batch implementations does not change classification behavior except for
numerical differences from the underlying computation.

#### Scenario: Judgments match candidate order
- **WHEN** the scorer evaluates candidates `[a, b, c]` as a batch
- **THEN** the returned judgments are in order `[a, b, c]`

#### Scenario: Judgment at index i references candidate at index i
- **WHEN** the scorer evaluates candidates `[c0, c1, ..., cn-1]`
- **THEN** for every index `i`, `judgments[i].candidate == candidates[i]`

#### Scenario: Order preserved despite internal reordering
- **WHEN** the infrastructure internally reorders batch items for execution efficiency
- **THEN** the returned judgments are still in the original candidate order

### Requirement: Batch failure is atomic
A single `score(text, candidates)` invocation SHALL be atomic from the
application's perspective. If one candidate produces an invalid score or the
model returns an invalid batch shape, the scorer SHALL NOT return scores for
the remaining candidates. The scorer SHALL NOT return a partially populated
judgment list. Tokenizer, tensor construction, model execution, and
output-shape failures SHALL be translated into the existing
`CandidateScoringError` boundary with the original exception as `__cause__`.
No batch-specific exception type SHALL be introduced unless callers need to
distinguish batch failures from ordinary scoring failures.

#### Scenario: One invalid score fails the entire invocation
- **WHEN** one candidate in a batch produces an invalid score (NaN, infinity)
- **THEN** the scorer raises `CandidateScoringError` and returns no judgments

#### Scenario: Invalid batch shape fails the entire invocation
- **WHEN** the model returns a batch shape that does not match the input batch
- **THEN** the scorer raises `CandidateScoringError` and returns no judgments

#### Scenario: No partial judgment list on failure
- **WHEN** a batch cannot be evaluated reliably
- **THEN** the scorer raises an exception rather than returning a partial judgment list

#### Scenario: Batch failure translated through existing error boundary
- **WHEN** a batch inference failure occurs
- **THEN** the application receives `CandidateScoringError`, not a raw infrastructure or batch-specific exception

### Requirement: One async blocking boundary per batch
The scorer SHALL perform synchronous batch model execution behind a single
non-blocking execution boundary so the application's async event loop is not
blocked. The exact executor mechanism SHALL be infrastructure-specific and
SHALL NOT be prescribed by the application port contract. The scorer SHALL
construct the complete batch before performing the batch model call, rather
than performing multiple blocking calls.

#### Scenario: Event loop not blocked during batch inference
- **WHEN** the scorer performs batched model inference
- **THEN** the application's event loop is not blocked

#### Scenario: Single blocking boundary for the batch
- **WHEN** the scorer evaluates a batch of candidates
- **THEN** synchronous model execution occurs behind one non-blocking boundary, not one per candidate

### Requirement: Scoring batch, model batch, and chunk terminology
A **scoring batch** is the logical group of all candidates passed to one
`score(text, candidates)` invocation. A **model batch** is the concrete set of
items passed to one `forward_batch` call. A **chunk** is a subset of a scoring
batch passed as one model batch when the scoring batch is split for memory
constraints. The initial implementation uses one model batch per scoring batch
(no chunking). Future implementations MAY split a scoring batch into multiple
chunks (model batches); chunking SHALL remain invisible to the application and
SHALL preserve candidate ordering and score semantics across chunks.

#### Scenario: Scoring batch maps to one model batch initially
- **WHEN** the scorer evaluates a scoring batch of `N` candidates
- **THEN** one model batch containing all `N` items is passed to `forward_batch`

#### Scenario: Chunking is invisible to the application
- **WHEN** a future implementation splits a scoring batch into multiple chunks
- **THEN** the application sees no chunking boundary, and the returned judgments are ordered as if no chunking occurred

### Requirement: Batch size equals candidate count initially
The initial implementation SHALL send all candidates from one scoring
invocation through one model batch (one `forward_batch` call). There SHALL be
no configurable `max_batch_size` in the application contract. A future
infrastructure implementation MAY split a scoring batch into multiple chunks
for memory constraints; such chunking SHALL remain invisible to the
application and SHALL preserve candidate ordering and score semantics.

#### Scenario: All candidates in one model batch
- **WHEN** the scorer evaluates `N` candidates
- **THEN** all `N` candidates are sent in a single model batch with no application-visible chunking

#### Scenario: No configurable batch size in application contract
- **WHEN** the `CandidateScorer` port contract is inspected
- **THEN** it exposes no `max_batch_size` or batch-sizing configuration

### Requirement: Sequential compatibility adapter is temporary
If a concrete model adapter cannot support native batching, that limitation
SHALL be handled inside infrastructure. A compatibility adapter MAY implement
batch execution by delegating to sequential execution (`forward_batch(items)`
calls `forward(item)` for each item). This is permitted ONLY as a temporary
transitional compatibility mechanism for adapters that lack native batch
support. The production local-model adapter used by
`LLMLogitCandidateScorer` MUST use native tensor batching. Sequential
delegation is permitted only for explicitly supported compatibility adapters.
The application SHALL NOT branch on whether batching is supported.

#### Scenario: Compatibility adapter preserves contract
- **WHEN** a model adapter does not support native batching
- **THEN** a temporary compatibility adapter delegates to sequential execution and preserves the scoring contract

#### Scenario: Production adapter uses native batching
- **WHEN** the production local-model adapter is used
- **THEN** it performs native tensor batching, not sequential delegation

#### Scenario: Application does not branch on batch support
- **WHEN** the application invokes the scorer
- **THEN** it does not check whether the model adapter supports batching

### Requirement: Batch scoring preserves existing port and information boundary
The `CandidateScorer` port, `ClassifyInput`, `Classification`, `Judgment`,
and `ClassificationPolicy` SHALL remain unchanged. No model, tokenizer,
tensor, padding, attention mask, or batch structure details SHALL leak into
domain or application. The existing generative classification path SHALL
continue to operate unchanged. Batch scoring SHALL be additive to the
infrastructure layer only.

#### Scenario: Port contract unchanged
- **WHEN** the `CandidateScorer` port is inspected after batch scoring is introduced
- **THEN** its signature and semantics are identical to before

#### Scenario: No batch internals leak to domain or application
- **WHEN** the domain and application layers are inspected
- **THEN** they contain no token IDs, padding, attention masks, tensor shapes, or batch structures

#### Scenario: Generative classification path unchanged
- **WHEN** batch scoring is introduced
- **THEN** the existing generative classification pipeline continues to operate unchanged
