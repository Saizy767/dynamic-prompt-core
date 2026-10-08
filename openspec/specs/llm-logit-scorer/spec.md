# LLM Logit Scorer Specification

## Purpose

Define the infrastructure `CandidateScorer` implementation that derives
candidate scores from LLM logits, owning prompt construction, tokenizer and
model seams, logit extraction, the logit-to-score algorithm, inference-mode
enforcement, numerical validation, and error translation — all contained inside
the infrastructure layer behind the application port.

## Requirements

### Requirement: Concrete CandidateScorer implementation
A concrete `CandidateScorer` implementation SHALL be provided in infrastructure
that derives candidate scores from LLM logits. The implementation SHALL satisfy
the `CandidateScorer` port contract: `async def score(text, candidates) ->
list[Judgment]`, returning exactly one `Judgment` per supplied candidate. The
implementation's constructor MAY depend on infrastructure-specific
collaborators (tokenizer adapter, model adapter, prompt builder, logit scorer);
those collaborators SHALL NOT appear in the `CandidateScorer` port. The
application SHALL see only `await scorer.score(text, candidates)`.

#### Scenario: Port contract satisfied
- **WHEN** the concrete scorer is invoked
- **THEN** it satisfies the `CandidateScorer` port contract

#### Scenario: One judgment per candidate
- **WHEN** the concrete scorer successfully evaluates N candidates
- **THEN** it returns exactly N judgments

#### Scenario: Infrastructure collaborators not in port
- **WHEN** the `CandidateScorer` port is inspected
- **THEN** it exposes no tokenizer adapter, model adapter, prompt builder, or logit scorer

#### Scenario: Application sees only score
- **WHEN** the application invokes the scorer
- **THEN** it calls `await scorer.score(text, candidates)` and receives judgments

### Requirement: Prompt construction in infrastructure
Prompt construction SHALL be owned entirely by infrastructure. The scorer
SHALL construct a model-facing scoring prompt from the classification input text
and the candidate. The prompt SHALL represent the candidate explicitly. The
domain and application layers SHALL NOT contain prompt text. The scorer SHALL
NOT accept prompt fragments or prompt templates from the application; the
application provides semantic data (`text`, `candidate`), not prompt parts. The
prompt SHALL carry semantic classification criteria that define what constitutes
a good classification, extracted from the existing classification task
semantics. The prompt SHALL NOT contain generation-protocol instructions
instructing the model to return JSON, return a confidence value, or use a
structured-output schema. Selection-oriented task instructions SHALL be
rewritten into candidate-evaluation semantics rather than excluded. The prompt
SHALL be candidate-specific: each candidate receives its own logical prompt, and
the candidate set SHALL NOT be embedded into one multi-class prompt. The prompt
SHALL expose the boundary between the contextual prefix and the candidate
continuation so the tokenizer can determine candidate continuation positions
structurally.

#### Scenario: Prompt constructed in infrastructure
- **WHEN** the scorer prepares a model input
- **THEN** the prompt is constructed inside infrastructure

#### Scenario: Candidate represented in prompt
- **WHEN** a scoring prompt is constructed
- **THEN** the candidate is represented explicitly in the prompt

#### Scenario: No prompt text in domain or application
- **WHEN** the domain and application layers are inspected
- **THEN** they contain no scoring prompt text

#### Scenario: Application provides semantic data not prompt parts
- **WHEN** the scorer port contract is inspected
- **THEN** it accepts text and candidates, not prompt fragments or prompt templates

#### Scenario: Prompt carries semantic classification criteria
- **WHEN** a scoring prompt is constructed
- **THEN** it contains semantic classification criteria that define what constitutes a good classification

#### Scenario: Prompt excludes generation-protocol instructions
- **WHEN** a scoring prompt is inspected
- **THEN** it does not contain instructions to return JSON, return a confidence value, or use a structured-output schema

#### Scenario: Prompt is candidate-specific
- **WHEN** the scorer evaluates multiple candidates
- **THEN** each candidate receives its own logical prompt and no prompt contains more than one candidate

#### Scenario: Prompt exposes prefix/candidate boundary
- **WHEN** a scoring prompt is constructed
- **THEN** the boundary between the contextual prefix and the candidate continuation is explicitly exposed

### Requirement: Candidate value used verbatim for tokenization
The scorer SHALL use the exact `Candidate.value` string as supplied for
tokenization. The scorer SHALL NOT strip, lowercase, casefold, normalize
whitespace, or otherwise transform `Candidate.value` before tokenization. This
is consistent with the `Candidate` domain model's no-auto-normalization rule:
the candidate's semantic value is stored verbatim, and the scorer preserves that
value through to the tokenizer. Leading or trailing whitespace in the candidate
value SHALL be tokenized as supplied; the resulting tokenization semantics are
an intentional consequence of the prompt format.

#### Scenario: Candidate value not transformed before tokenization
- **WHEN** a candidate with value `"  sports  "` is scored
- **THEN** the scorer tokenizes `"  sports  "` verbatim, without stripping or normalizing

#### Scenario: Leading whitespace preserved
- **WHEN** a candidate with leading whitespace is scored
- **THEN** the leading whitespace is preserved through to tokenization

#### Scenario: No case folding before tokenization
- **WHEN** candidates `"Sports"` and `"sports"` are scored
- **THEN** they are tokenized as distinct strings, not case-folded to a common form

### Requirement: Tokenization boundary determination
The backend SHALL determine the boundary between the prefix tokens and the
candidate tokens by encoding the prefix and the candidate as separate tokenizer
inputs and composing the full sequence, NOT by encoding the full prompt and
then searching for the candidate substring or candidate tokens via string
matching. The prefix SHALL be the prompt text up to and including the candidate
label (e.g., `"{input}\nCandidate: "`); the candidate SHALL be `candidate.value`.
The backend SHALL record `prefix_token_count` and `candidate_token_ids` from the
two encodings. The full sequence SHALL be `prefix_tokens + candidate_tokens`.
Determining candidate token positions by substring search, token subsequence
matching, or any post-hoc alignment heuristic SHALL NOT be permitted, because
tokenizers may merge or split tokens across the boundary in ways that make such
searches unreliable.

#### Scenario: Boundary determined by separate encoding
- **WHEN** the backend tokenizes a scoring prompt
- **THEN** it encodes the prefix and the candidate separately and composes the full sequence

#### Scenario: Prefix token count recorded
- **WHEN** the backend tokenizes a scoring prompt
- **THEN** the prefix token count is known from the prefix encoding, not from post-hoc search

#### Scenario: No substring search for candidate positions
- **WHEN** the backend determines candidate token positions
- **THEN** it does not use substring search, token subsequence matching, or post-hoc alignment

### Requirement: Tokenizer and model seams
The implementation SHALL introduce narrow infrastructure-facing seams for the
tokenizer and the causal language model so the scoring algorithm is testable
with deterministic doubles and decoupled from a concrete model library. The
tokenizer seam SHALL convert text into model tokens. The model seam SHALL
execute model inference and expose output logits. These seams SHALL be
infrastructure-internal; the application SHALL NEVER import them. The seams
SHALL NOT be required to be `typing.Protocol` if a concrete adapter is more
practical, but the scoring algorithm SHALL depend on the seam, not a concrete
library class, where practical.

#### Scenario: Tokenizer seam converts text to tokens
- **WHEN** the tokenizer seam is invoked
- **THEN** it converts text into model tokens

#### Scenario: Model seam executes inference and exposes logits
- **WHEN** the model seam is invoked
- **THEN** it executes model inference and exposes output logits

#### Scenario: Seams are infrastructure-internal
- **WHEN** the application is inspected
- **THEN** it never imports the tokenizer or model seams

#### Scenario: Scoring algorithm depends on seams not concrete library
- **WHEN** the scoring algorithm is inspected
- **THEN** it depends on the seams, not a concrete model library class, where practical

### Requirement: Logit extraction in inference mode
The backend SHALL execute the model in inference-only mode. Training behavior
SHALL NOT be enabled. The implementation SHALL avoid retaining gradients when
the underlying framework supports gradient tracking. The scorer SHALL NOT mutate
model training state in a way that surprises other infrastructure components.
If the model lifecycle is owned externally, the scorer SHALL follow the existing
lifecycle convention rather than taking ownership implicitly.

#### Scenario: Model run in inference mode
- **WHEN** the scorer invokes the model
- **THEN** the model is in inference/evaluation mode

#### Scenario: No gradients retained
- **WHEN** the underlying framework supports gradient tracking
- **THEN** the implementation avoids retaining gradients

#### Scenario: No training state mutation
- **WHEN** the scorer runs
- **THEN** it does not mutate model training state in a way that surprises other infrastructure components

### Requirement: Causal logit alignment semantics
For a causal language model, the logits at position `i` predict the token at
position `i + 1`. The backend SHALL extract candidate-token log-probabilities
using this alignment: for each candidate token `c_k` at sequence position `k`,
the log-probability SHALL be `log_softmax(logits[position k - 1])[c_k_id]`,
where `position k - 1` is the position of the preceding token (the last prefix
token for the first candidate token, or the preceding candidate token for
subsequent candidate tokens). Formally, given prefix tokens `p_0 ... p_m` and
candidate tokens `c_0 ... c_{n-1}`:

```
candidate_logprob_0 = log_softmax(logits[m])[c_0_id]
candidate_logprob_1 = log_softmax(logits[m+1])[c_1_id]
...
candidate_logprob_{n-1} = log_softmax(logits[m+n-2])[c_{n-1}_id]
```

The backend SHALL NOT read logits at candidate token positions as if they
predict the candidate token at the same position. The alignment SHALL be
documented in the implementation.

#### Scenario: First candidate token scored from last prefix position
- **WHEN** the first candidate token `c_0` is scored
- **THEN** its log-probability is `log_softmax(logits[m])[c_0_id]` where `m` is the last prefix position

#### Scenario: Subsequent candidate token scored from preceding candidate position
- **WHEN** candidate token `c_k` (k > 0) is scored
- **THEN** its log-probability is `log_softmax(logits[m+k])[c_k_id]` using the preceding position

#### Scenario: Alignment documented
- **WHEN** the implementation is inspected
- **THEN** the causal logit alignment strategy is explicitly documented

### Requirement: Logit-to-score algorithm
The backend SHALL define one deterministic logit-derived score per candidate.
The score SHALL be the arithmetic mean of the candidate-token log-probabilities
as defined by the causal logit alignment:

```
score = mean(candidate_logprob_0, ..., candidate_logprob_{n-1})
```

Mean (not sum) normalizes by candidate token count, avoiding systematic length
bias. The score SHALL be a ranking/evidence signal where higher means stronger
support. The implementation SHALL NOT claim the score is a probability merely
because logits were transformed with softmax. The chosen strategy SHALL be
documented in the implementation.

#### Scenario: Deterministic score per candidate
- **WHEN** the same input and candidate are scored twice with the same model
- **THEN** the resulting scores are equal

#### Scenario: Higher score means stronger support
- **WHEN** candidate A has stronger model support than candidate B
- **THEN** candidate A receives a higher score than candidate B

#### Scenario: Mean not sum
- **WHEN** a candidate tokenizes into tokens with log-probs `[-0.2, -0.4]`
- **THEN** the score is `-0.3` (the mean), not `-0.6` (the sum)

#### Scenario: Strategy documented
- **WHEN** the implementation is inspected
- **THEN** the logit-to-score strategy is explicitly documented

#### Scenario: No unjustified probability claim
- **WHEN** the score is produced
- **THEN** it is not labeled a probability unless the scoring procedure establishes that interpretation

### Requirement: Multi-token candidate handling
The backend SHALL NOT assume every candidate maps to a single token. If a
candidate tokenizes into multiple tokens, the backend SHALL score the sequence
using the causal logit alignment and mean log-probability defined above. The
normalization strategy (mean over candidate tokens) SHALL remain
infrastructure-specific and SHALL NOT leak into `Judgment` or the application
port. The backend SHALL NOT systematically reward longer candidate strings
solely because they contain more tokens.

#### Scenario: Multi-token candidate scored
- **WHEN** a candidate tokenizes into multiple tokens
- **THEN** the backend scores the sequence using causal alignment and mean log-probability

#### Scenario: No systematic length bias
- **WHEN** candidates of different token lengths are scored
- **THEN** longer candidates are not systematically rewarded solely for having more tokens

#### Scenario: Normalization stays in infrastructure
- **WHEN** the normalization strategy is applied
- **THEN** it does not leak into `Judgment` or the application port

### Requirement: Numerical validation before judgment construction
The backend SHALL NOT return invalid floating-point values. Before constructing
a `Judgment`, the backend SHALL reject `NaN`, positive infinity, and negative
infinity as scores. The backend SHALL NOT silently clamp invalid values into a
valid range. `Judgment`'s own domain invariant SHALL remain the final guard.

#### Scenario: NaN score rejected
- **WHEN** the logit-to-score calculation produces NaN
- **THEN** the backend raises an exception rather than constructing a judgment

#### Scenario: Positive infinity rejected
- **WHEN** the logit-to-score calculation produces positive infinity
- **THEN** the backend raises an exception rather than constructing a judgment

#### Scenario: Negative infinity rejected
- **WHEN** the logit-to-score calculation produces negative infinity
- **THEN** the backend raises an exception rather than constructing a judgment

#### Scenario: No silent clamping
- **WHEN** the calculation produces an out-of-range finite value
- **THEN** the backend does not silently clamp it into a different range

### Requirement: Multi-candidate API is an application contract not a model batch
`CandidateScorer.score()` SHALL accept multiple candidates in one invocation.
This is an application-level contract: one invocation evaluates a set of
candidates. The implementation MAY evaluate candidates sequentially (one model
forward per candidate) or batch them; the application SHALL NOT know or care
which strategy is used. The multi-candidate API SHALL NOT be interpreted as a
requirement that the implementation perform a single batched model forward.
Batching, prefix reuse, and provider batch APIs are implementation
optimizations that SHALL NOT change the application contract. The backend design
SHALL NOT prevent future batching or prefix reuse.

#### Scenario: Multiple candidates in one call
- **WHEN** the scorer is invoked with multiple candidates
- **THEN** all candidates are evaluated in a single invocation

#### Scenario: Sequential evaluation is valid
- **WHEN** the implementation evaluates candidates one model forward at a time
- **THEN** the multi-candidate application contract is satisfied

#### Scenario: Batching is internal
- **WHEN** the implementation evaluates multiple candidates
- **THEN** the application does not know whether batching occurred

#### Scenario: Backend does not prevent batching
- **WHEN** the backend design is inspected
- **THEN** future batching and prefix reuse are not prevented

### Requirement: Ordering of returned judgments
The scorer SHALL return judgments in the same order as the supplied candidates
unless there is a strong infrastructure reason not to. Callers SHALL NOT rely
solely on ordering because each `Judgment` explicitly identifies its candidate.
The classification policy SHALL operate on candidate association, not positional
assumptions.

#### Scenario: Judgments follow candidate order
- **WHEN** the scorer successfully evaluates candidates in a given order
- **THEN** the returned judgments are in the same order, unless a strong infrastructure reason dictates otherwise

#### Scenario: Callers do not rely on position
- **WHEN** the application reads judgments
- **THEN** it associates judgments by candidate, not by list position

### Requirement: Async contract without event loop blocking
`CandidateScorer.score()` SHALL be asynchronous. The infrastructure
implementation SHALL NOT block the application's event loop during model
inference. The concrete mechanism for executing synchronous model inference
(thread offloading, executor, dedicated worker, async-native model, or any other
strategy) SHALL be infrastructure-specific and SHALL NOT be prescribed by the
application port contract. The application SHALL NOT care whether the
underlying implementation is synchronous, asynchronous, remote, local, or
batched.

#### Scenario: Score is async
- **WHEN** the port contract is inspected
- **THEN** `score` is an async method

#### Scenario: Event loop not blocked
- **WHEN** the scorer performs model inference
- **THEN** the application's event loop is not blocked

#### Scenario: Execution mechanism is infrastructure-specific
- **WHEN** the implementation executes synchronous model inference
- **THEN** the mechanism (thread offloading, executor, etc.) is chosen by infrastructure, not prescribed by the port

### Requirement: Application-facing scoring error
The application-facing scoring error SHALL reside in the application layer
(`application/errors/scoring.py`) as `CandidateScoringError`. The application
SHALL NOT import infrastructure error classes. The infrastructure SHALL define
its own internal error (`LLMScoringError` in
`infrastructure/llm/scoring/errors.py`) for wrapping library-specific failures
and invalid scores before translation. The `LLMLogitCandidateScorer` SHALL
translate `LLMScoringError` and raw infrastructure exceptions (torch,
transformers, provider SDK, tokenizer) into `CandidateScoringError` with the
original as `__cause__`. The application SHALL receive only
`CandidateScoringError`, never raw infrastructure exceptions or
`LLMScoringError`.

#### Scenario: Application error in application layer
- **WHEN** the application-facing scoring error is located
- **THEN** it resides in `application/errors/scoring.py`

#### Scenario: Infrastructure error is infrastructure-internal
- **WHEN** the infrastructure scoring error is located
- **THEN** it resides in `infrastructure/llm/scoring/errors.py` and is not imported by the application

#### Scenario: Infrastructure exceptions translated
- **WHEN** the model or tokenizer raises an infrastructure-specific exception
- **THEN** the backend translates it into `CandidateScoringError`

#### Scenario: No raw infrastructure exceptions leak
- **WHEN** the application receives a scoring failure
- **THEN** it is a `CandidateScoringError`, not a raw torch, transformers, provider SDK, tokenizer, or `LLMScoringError` exception

### Requirement: Atomic scoring failure semantics
A scoring invocation SHALL be atomic from the application's perspective. If the
backend cannot produce a valid judgment for any requested candidate, the call
SHALL fail. The backend SHALL NOT silently return fewer judgments than
candidates. This ensures classification never operates on an incomplete
candidate set unless such behavior is explicitly introduced later.

#### Scenario: Partial failure raises
- **WHEN** the backend cannot evaluate one or more candidates
- **THEN** the call raises an exception rather than returning a partial result

#### Scenario: No silent omission
- **WHEN** a candidate cannot be evaluated
- **THEN** the call raises an exception rather than omitting that candidate's judgment

### Requirement: Backend information boundary
The concrete scorer SHALL NOT expose logits, token IDs, tokenizer APIs, model
objects, or provider-specific types through the `CandidateScorer` port. All
model internals SHALL remain behind the port boundary. The port SHALL remain
valid if the underlying implementation changes between logits-based scoring,
another local model, a remote model, a deterministic test scorer, or another
future scoring mechanism.

#### Scenario: No logits exposed
- **WHEN** the port contract is inspected
- **THEN** it exposes no logits

#### Scenario: No token IDs exposed
- **WHEN** the port contract is inspected
- **THEN** it exposes no token IDs

#### Scenario: No tokenizer or model objects exposed
- **WHEN** the port contract is inspected
- **THEN** it exposes no tokenizer APIs or model objects

### Requirement: Backend dependency direction
The concrete scorer and its seams SHALL reside in `infrastructure`. The
infrastructure implementation MAY import `domain`, `application.ports`, and
`application.errors` and SHALL NOT import `application.use_cases` or
`interfaces`. The domain SHALL NOT import infrastructure. The application port
SHALL NOT import infrastructure.

#### Scenario: Scorer resides in infrastructure
- **WHEN** the concrete scorer is located
- **THEN** it resides in `infrastructure`

#### Scenario: Infrastructure imports ports and errors not use cases
- **WHEN** the infrastructure is inspected
- **THEN** it may import `application.ports` and `application.errors` but does not import `application.use_cases` or `interfaces`

#### Scenario: Domain does not import infrastructure
- **WHEN** the domain is inspected
- **THEN** it does not import infrastructure

### Requirement: Backend testability without real model
The backend SHALL be testable with deterministic test doubles for the tokenizer
and model seams. The standard test suite SHALL NOT require downloading or
loading a large model. A real-model integration test, if added, SHALL be opt-in,
isolated from the default unit-test suite, and explicit about its
model/tokenizer dependencies.

#### Scenario: Deterministic doubles usable
- **WHEN** the backend is unit-tested
- **THEN** the tokenizer and model seams are replaceable with deterministic test doubles

#### Scenario: Standard suite needs no model download
- **WHEN** the standard test suite is run
- **THEN** it does not download or load a large model

#### Scenario: Real model test is opt-in
- **WHEN** a real-model integration test exists
- **THEN** it is opt-in, isolated from the default suite, and explicit about dependencies

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
domain or application. Batch scoring SHALL be additive to the infrastructure
layer only. The generative classification path SHALL NOT exist; candidate
scoring is the sole classification mechanism.

#### Scenario: Port contract unchanged
- **WHEN** the `CandidateScorer` port is inspected after batch scoring is introduced
- **THEN** its signature and semantics are identical to before

#### Scenario: No batch internals leak to domain or application
- **WHEN** the domain and application layers are inspected
- **THEN** they contain no token IDs, padding, attention masks, tensor shapes, or batch structures

#### Scenario: No generative classification path
- **WHEN** the codebase is inspected for generative classification
- **THEN** no `classify_detailed`, `ClassificationResult`, or generative classification fallback exists
