# Spec Delta

## Purpose

HTTP-based scoring adapter that implements the tokenizer and model seams by calling llama-server REST endpoints, enabling GPU-accelerated logit extraction without an in-process model dependency.

## ADDED Requirements

### Requirement: Endpoint configuration
The adapter SHALL connect to a configurable `llama-server` HTTP endpoint. The
endpoint URL and request timeout SHALL be read from the scorer configuration.
The default endpoint SHALL be `http://127.0.0.1:8080`. The default request
timeout SHALL be 30 seconds. The endpoint URL SHALL be validated as a
well-formed HTTP or HTTPS URL before the adapter is constructed.

#### Scenario: Default endpoint used
- **WHEN** no endpoint is specified in configuration
- **THEN** the adapter connects to `http://127.0.0.1:8080`

#### Scenario: Custom endpoint from config
- **WHEN** configuration specifies `endpoint = "http://192.168.1.10:9090"`
- **THEN** the adapter connects to `http://192.168.1.10:9090`

#### Scenario: Invalid endpoint rejected
- **WHEN** configuration specifies a malformed endpoint
- **THEN** the adapter construction raises a configuration error naming the invalid endpoint

### Requirement: Logit retrieval via completion endpoint
The model adapter SHALL retrieve logits for a tokenized prompt by calling the
`llama-server` `/completion` endpoint with `n_predict` set to 0 (no token
generation) and logits exposure enabled, so the server computes a single
forward pass and returns logits for all input tokens. The adapter SHALL send
the prompt as raw token IDs when the endpoint supports token-input mode, or as
decoded text otherwise. The adapter SHALL extract the logits array from the
response and return it in the `BatchedLogits` shape
`[batch_size, sequence_length, vocabulary_size]` expected by the orchestrator.

#### Scenario: Forward pass returns logits
- **WHEN** the adapter calls the completion endpoint with a prompt and `n_predict: 0`
- **THEN** the server returns logits for all input tokens without generating new tokens

#### Scenario: Logits converted to BatchedLogits shape
- **WHEN** the adapter receives the logits array from the server
- **THEN** it returns a `BatchedLogits` result with shape `[batch_size, sequence_length, vocabulary_size]`

#### Scenario: No token generation requested
- **WHEN** the adapter sends a request to the completion endpoint
- **THEN** `n_predict` is set to 0 so the server performs only a forward pass

### Requirement: Tokenization via tokenize endpoint
The tokenizer adapter SHALL encode text by calling the `llama-server`
`/tokenize` endpoint. The adapter SHALL encode the prefix and the candidate as
separate tokenizer inputs and compose the full sequence, preserving the
separate-encoding boundary-determination strategy required by the scoring
specification. The adapter SHALL record `prefix_token_count` and
`candidate_token_ids` from the two encodings. The full sequence SHALL be
`prefix_tokens + candidate_tokens`.

#### Scenario: Text encoded via HTTP tokenize endpoint
- **WHEN** the tokenizer adapter encodes text
- **THEN** it calls the `/tokenize` endpoint and returns the token IDs from the response

#### Scenario: Separate encoding preserves boundary
- **WHEN** the adapter encodes a prefix and a candidate
- **THEN** it encodes them separately and composes the full sequence, recording `prefix_token_count` and `candidate_token_ids`

#### Scenario: Candidate value not transformed
- **WHEN** a candidate with leading or trailing whitespace is tokenized
- **THEN** the value is sent to the tokenize endpoint verbatim, without stripping or normalizing

### Requirement: Async HTTP transport without thread offloading
The adapter SHALL use `aiohttp` for all HTTP requests. The model seam's
`forward_batch` SHALL be an async method that performs HTTP I/O without
blocking the event loop and without thread offloading (`asyncio.to_thread`).
The adapter SHALL NOT acquire `threading.Lock` or `threading.BoundedSemaphore`
around HTTP requests. The adapter SHALL follow the established `AsyncTask` /
`client.py` pattern for session management, connector limits, and timeouts.

#### Scenario: Event loop not blocked
- **WHEN** the adapter performs HTTP requests during scoring
- **THEN** the application's event loop is not blocked

#### Scenario: No threading locks
- **WHEN** the adapter implementation is inspected
- **THEN** it does not acquire `threading.Lock` or `threading.BoundedSemaphore` for HTTP requests

#### Scenario: aiohttp session reused
- **WHEN** the adapter makes multiple HTTP requests
- **THEN** it reuses a shared `aiohttp.ClientSession` with configured connector limits and timeouts

### Requirement: Batch handling via concurrent HTTP requests
The model adapter SHALL implement `forward_batch` by issuing one HTTP request
per batch item concurrently using `asyncio.gather` (or equivalent). The
adapter SHALL preserve batch item ordering: the returned `BatchedLogits` item
`i` SHALL correspond to input batch item `i` without reordering. The adapter
SHALL NOT require the server to support a native batched-logit tensor API.

#### Scenario: One HTTP request per batch item
- **WHEN** `forward_batch` is called with N batch items
- **THEN** the adapter issues N HTTP requests to the completion endpoint

#### Scenario: Batch items requested concurrently
- **WHEN** `forward_batch` is called with multiple batch items
- **THEN** the HTTP requests are issued concurrently, not sequentially

#### Scenario: Batch item order preserved
- **WHEN** `forward_batch` returns `BatchedLogits` for a batch of N items
- **THEN** item `i` in the result corresponds to input batch item `i` without reordering

### Requirement: HTTP error translation
HTTP errors (non-200 status, connection refused, timeout) SHALL be translated
into the infrastructure-internal `LLMScoringError` (or `GGUFProviderError`),
which the orchestrator translates into `CandidateScoringError` at the
application boundary. The error SHALL include the endpoint URL, HTTP status
code (when available), and response body excerpt. The application SHALL never
receive raw `aiohttp` exceptions or HTTP status codes.

#### Scenario: Non-200 status translated
- **WHEN** the server returns a non-200 HTTP status
- **THEN** the adapter raises `LLMScoringError` with the status code and response body excerpt

#### Scenario: Connection refused translated
- **WHEN** the server is not running and the connection is refused
- **THEN** the adapter raises `LLMScoringError` with the endpoint URL and connection error detail

#### Scenario: Request timeout translated
- **WHEN** an HTTP request exceeds the configured timeout
- **THEN** the adapter raises `LLMScoringError` naming the timeout and endpoint URL

#### Scenario: No raw aiohttp exceptions leak
- **WHEN** an HTTP error occurs during scoring
- **THEN** the application receives `CandidateScoringError`, not a raw `aiohttp` exception

### Requirement: Response logit parsing
The adapter SHALL parse the `llama-server` response format to extract the
logits array. The adapter SHALL validate that the logits array has the expected
shape (sequence length matches the input token count, vocabulary dimension
matches the server-reported vocabulary size). The adapter SHALL convert logits
to the floating-point type expected by the orchestrator. The adapter SHALL
reject responses with missing or malformed logits arrays.

#### Scenario: Logits extracted from response
- **WHEN** the server returns a response with a logits array
- **THEN** the adapter extracts the logits and converts them to the expected floating-point type

#### Scenario: Shape mismatch rejected
- **WHEN** the returned logits array shape does not match the input token count
- **THEN** the adapter raises `LLMScoringError` naming the expected and actual shapes

#### Scenario: Missing logits rejected
- **WHEN** the server response does not contain a logits array
- **THEN** the adapter raises `LLMScoringError` indicating the logits field is missing

### Requirement: No in-process model dependency
The adapter SHALL NOT import or load `llama-cpp-python` or any native model
library. The scoring path through the `"llama_server"` backend SHALL depend
only on core dependencies already present in the project (`aiohttp`, `numpy`).
The adapter SHALL NOT require the optional `[gguf]` extra to be installed.

#### Scenario: No llama-cpp-python import
- **WHEN** the adapter module is imported
- **THEN** it does not import `llama_cpp` or any native model library

#### Scenario: No optional dependency required
- **WHEN** the `"llama_server"` backend is used
- **THEN** the `[gguf]` optional extra is not required to be installed

### Requirement: Backend configuration support
The scorer configuration SHALL accept `"llama_server"` as a valid backend value
in addition to the existing `"huggingface"` and `"gguf"` backends. The
configuration SHALL parse connection fields (`endpoint`, `timeout`) from the
`[model]` config section when the backend is `"llama_server"`. The factory
`build_candidate_scorer` SHALL dispatch to the HTTP adapter when
`config.backend == "llama_server"`. The existing `"gguf"` and `"huggingface"`
backends SHALL remain unchanged.

#### Scenario: llama_server backend accepted
- **WHEN** configuration specifies `backend = "llama_server"`
- **THEN** the config parser accepts it and the factory builds the HTTP adapter

#### Scenario: Connection fields parsed
- **WHEN** configuration specifies `backend = "llama_server"` with `endpoint` and `timeout`
- **THEN** the config parser reads those fields into the adapter configuration

#### Scenario: Existing backends unchanged
- **WHEN** configuration specifies `backend = "gguf"` or `backend = "huggingface"`
- **THEN** the factory dispatches to the existing in-process adapters without change

### Requirement: Backend metadata for run identity
The adapter SHALL provide backend metadata (endpoint URL, served model name or
alias, vocabulary size) consistent with the existing adapter's `describe()`
contract, so run identity and resume logic can distinguish runs by backend and
endpoint. The metadata SHALL NOT expose internal HTTP session objects or
connection pool details.

#### Scenario: Metadata includes endpoint
- **WHEN** the adapter's backend metadata is queried
- **THEN** it includes the endpoint URL and served model name

#### Scenario: Metadata does not expose internals
- **WHEN** the adapter's backend metadata is queried
- **THEN** it does not expose HTTP session objects or connection pool details
