# Spec Delta

## Purpose

HTTP-based candidate scoring against a running llama-server instance: the
model seam forces generation of each candidate token via `logit_bias` on the
OpenAI-compatible `/v1/completions` endpoint and reads back the token's true
full-vocabulary log-probability, so scoring requires no in-process model and
no Python-side logit materialization.

## ADDED Requirements

### Requirement: Forced candidate-token generation via logit_bias
The model seam SHALL score each candidate token by issuing one
`/v1/completions` request per token with: the prompt as a list of token IDs
(prefix tokens plus already-scored candidate tokens), `max_tokens: 1`,
`logprobs: 1`, greedy sampling, and `logit_bias` raising the candidate token's
logit so the server MUST generate exactly that token. The reported log-prob
of the generated token SHALL be the token's true full-vocabulary log-prob
under the unmodified model distribution.

#### Scenario: Forced token returns its true log-prob
- **WHEN** a request forces candidate token `t` after prefix `p`
- **THEN** the server generates `t` and reports the log-prob of `t` given `p` under the full unmodified vocabulary distribution

#### Scenario: Bias does not distort the reported log-prob
- **WHEN** the same token's log-prob is measured with and without the forcing bias
- **THEN** the reported log-prob values are identical

### Requirement: No top-N window, floor, or renormalization
The scoring mechanism SHALL NOT depend on the candidate token appearing in any
`top_logprobs` window, SHALL NOT apply floor values for tokens outside a
top-N list, and SHALL NOT derive one candidate's log-prob arithmetically from
another's. Every candidate token's log-prob SHALL be read directly from its
own forced-generation response.

#### Scenario: Unlikely candidate token is still exact
- **WHEN** a candidate token's true log-prob is far below the top-20 tokens
- **THEN** its score is its exact log-prob, not a floor value

#### Scenario: Both candidates of a pair are scored independently
- **WHEN** two candidates are scored against the same prefix
- **THEN** each candidate's score comes from its own forced-generation request, with no cross-candidate arithmetic

### Requirement: Generated token identity verification
The model seam SHALL verify that the token ID returned by the server matches
the candidate token ID that was forced. On mismatch the seam SHALL raise a
scoring error naming both token IDs, rather than silently scoring a different
token.

#### Scenario: Identity mismatch is rejected
- **WHEN** the server returns a generated token ID different from the forced candidate token ID
- **THEN** the scoring call raises a scoring error that names the expected and actual token IDs

### Requirement: Multi-token candidate iteration
For a candidate spanning multiple tokens, the seam SHALL score token `k` by
sending the prefix plus candidate tokens `0..k-1` as the prompt and forcing
token `k`. The candidate's score SHALL be the arithmetic mean of its tokens'
log-probs, preserving the existing mean-log-prob scoring semantics.

#### Scenario: Multi-token candidate scored step by step
- **WHEN** a candidate tokenizes to tokens `c0, c1`
- **THEN** the seam issues one request forcing `c0` after the prefix and one request forcing `c1` after the prefix plus `c0`, and the score is the mean of the two log-probs

### Requirement: Required batch boundary metadata
The model seam's batched forward call SHALL require per-item prefix token
counts and candidate token IDs. When either is missing, the call SHALL be
rejected with an explicit error; the seam SHALL NOT fall back to full-sequence
logit retrieval.

#### Scenario: Missing boundary metadata is rejected
- **WHEN** the batched forward call is invoked without prefix token counts or candidate token IDs
- **THEN** the call raises an explicit error stating the required inputs

### Requirement: Tokenization via tokenize endpoint
The tokenizer seam SHALL encode prefixes and candidates through the server's
`/tokenize` endpoint so token IDs match the model's own tokenizer. The seam
SHALL encode `prefix + candidate` as one string, verify the prefix token
sequence is an exact prefix of the full sequence, and reject with an error on a
cross-boundary merge. Candidate values SHALL be used verbatim.

#### Scenario: Server tokenizer used for encoding
- **WHEN** a prefix and candidate are encoded
- **THEN** the token IDs come from the server's `/tokenize` endpoint

#### Scenario: Boundary merge is rejected
- **WHEN** the prefix token sequence is not an exact prefix of the full sequence
- **THEN** encoding raises an error rather than scoring a different token sequence

### Requirement: Async HTTP transport without thread offloading
All scoring and tokenization I/O SHALL be asynchronous over HTTP. The model
seam SHALL NOT load a model in-process, hold inference locks, or offload
blocking model calls to threads. Concurrent batch items SHALL be evaluated
with concurrent HTTP requests bounded by a configured concurrency limit.

#### Scenario: No in-process model
- **WHEN** the scoring backend is constructed
- **THEN** no model weights are loaded into the Python process

#### Scenario: Batch items evaluated concurrently
- **WHEN** a batch contains multiple items
- **THEN** their HTTP requests are issued concurrently, bounded by the concurrency limit

### Requirement: HTTP error translation and transient retry
Transport-level failures (connection refused, server disconnect, timeout)
SHALL be retried up to a bounded number of attempts with delay. Non-200
responses and malformed response bodies SHALL surface as scoring errors
carrying the endpoint, status, and a response excerpt. No raw HTTP client
exceptions SHALL escape the seam.

#### Scenario: Transient disconnect is retried
- **WHEN** the server disconnects mid-request
- **THEN** the seam retries up to the bounded attempt count before failing

#### Scenario: HTTP error surfaces as scoring error
- **WHEN** the server returns a non-200 status
- **THEN** a scoring error is raised carrying the status code and a body excerpt

### Requirement: Single scorer backend configuration
The scorer backend configuration SHALL accept exactly one backend:
`llama_server`. Requests for any other backend SHALL be rejected with a
configuration error. The configuration SHALL carry the server endpoint and
request timeout, with defaults, and SHALL validate the endpoint scheme and
timeout positivity. When the `[model]` section is absent, the configuration
SHALL default to `llama_server` with the default endpoint.

#### Scenario: Only llama_server is valid
- **WHEN** a configuration names any backend other than `llama_server`
- **THEN** a configuration error is raised listing the valid backend

#### Scenario: Defaults without a model section
- **WHEN** no `[model]` section is present
- **THEN** the scorer uses the llama_server backend with the default endpoint and timeout

#### Scenario: Invalid endpoint rejected
- **WHEN** the configured endpoint does not start with `http://` or `https://`
- **THEN** a configuration error is raised

### Requirement: Backend metadata for run identity
The model seam SHALL expose backend metadata for experiment recording and
resume identity: backend name, endpoint, and the size of the logit-bias
forcing window. The metadata SHALL NOT include model weights or in-process
model state.

#### Scenario: Metadata describes the HTTP backend
- **WHEN** backend metadata is queried
- **THEN** it identifies the llama_server backend, its endpoint, and the forcing configuration, with no in-process model state
