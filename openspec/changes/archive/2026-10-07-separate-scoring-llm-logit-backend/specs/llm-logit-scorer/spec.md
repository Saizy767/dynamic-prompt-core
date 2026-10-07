# Spec Delta

## Purpose

Define the infrastructure `CandidateScorer` implementation that derives
candidate scores from LLM logits, owning prompt construction, tokenizer and
model seams, logit extraction, the logit-to-score algorithm, inference-mode
enforcement, numerical validation, and error translation — all contained inside
the infrastructure layer behind the application port.

## ADDED Requirements

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
application provides semantic data (`text`, `candidate`), not prompt parts.

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
