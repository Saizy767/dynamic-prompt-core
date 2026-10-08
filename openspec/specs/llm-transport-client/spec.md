# Spec

## Purpose

Structured transport, parsing, and jsonl logging for LLM requests through
OpenAI-compatible servers (llama.cpp, vLLM), providing the request-level
foundation for the prompt-optimization pipeline.

## Requirements

### Requirement: Transport and parsing separation
The client SHALL provide an `analyze_raw` method that returns a `RawResponse`
containing raw content and transport metadata without parsing, and an `analyze`
method that parses the raw response into a validated model instance.

#### Scenario: analyze_raw returns unparsed content
- **WHEN** `analyze_raw` is called with a text and a Pydantic model
- **THEN** it returns a `RawResponse` whose `content` field is the raw string from the model, not a parsed object

#### Scenario: analyze composes parsing over analyze_raw
- **WHEN** `analyze` is called with the same text and model
- **THEN** the result equals parsing `analyze_raw(text, model).content` through the model's validator

#### Scenario: RawResponse captures transport metadata
- **WHEN** a request completes (success or failure)
- **THEN** the `RawResponse` includes `content`, `model`, `latency_ms`, `status`, `attempts`, `finish_reason`, `usage`, `http_status`, and `error`

### Requirement: Call-type wrappers
The client SHALL provide `extract_theses` and `extract_theses_many` methods
that pass a `call_type` identifier into log entries and use per-call defaults
for `system_prompt`, `max_tokens`, and `truncate_tokens`. The client SHALL NOT
provide `classify` or `classify_many` methods; classification is performed
through the candidate-scoring architecture, not through generative LLM calls.

#### Scenario: extract_theses uses extraction defaults
- **WHEN** `extract_theses` is called without explicit `truncate_tokens` or `max_tokens`
- **THEN** it uses defaults appropriate for extraction (truncate_tokens=2000, max_tokens=512)

#### Scenario: per-call system_prompt overrides constructor default
- **WHEN** a wrapper is called with an explicit `system_prompt`
- **THEN** that prompt is used for the request, overriding the constructor's `system_prompt`

#### Scenario: call_type appears in log
- **WHEN** any wrapper is called and logging is enabled
- **THEN** the log entry's `call_type` field is `"extract_theses"` matching the wrapper called

### Requirement: Structured jsonl logging
The client SHALL log one jsonl line per logical request (not per retry attempt)
when a `log_path` is provided, written under an async lock to prevent
interleaving.

#### Scenario: One log entry per request
- **WHEN** a request completes after N retry attempts
- **THEN** exactly one jsonl line is written with `attempts=N`

#### Scenario: Log entry contains required fields
- **WHEN** a log entry is written
- **THEN** it includes `timestamp`, `run_id`, `call_type`, `system_prompt_hash`, `text_hash`, `model_name`, `params`, `raw_content`, `latency_ms`, `finish_reason`, `usage`, `status`, `parse_status`, `attempts`, `http_status`, `error`, and `truncated`

#### Scenario: run_id defaults to smoke
- **WHEN** no `run_id` is provided to the client
- **THEN** log entries use `"smoke"` as the `run_id`

### Requirement: Parse status taxonomy
The client SHALL classify parse outcomes as `ok`, `invalid_json`,
`schema_mismatch`, or `truncated` to distinguish failure modes that retry
cannot fix from transport errors.

#### Scenario: Valid JSON matching schema
- **WHEN** the model returns valid JSON that matches the Pydantic schema
- **THEN** `parse_status` is `ok`

#### Scenario: Invalid JSON
- **WHEN** the model returns content that is not valid JSON
- **THEN** `parse_status` is `invalid_json`

#### Scenario: Valid JSON wrong shape
- **WHEN** the model returns valid JSON that does not match the schema
- **THEN** `parse_status` is `schema_mismatch`

#### Scenario: Truncated output
- **WHEN** `finish_reason` is `"length"` and JSON parsing fails
- **THEN** `parse_status` is `truncated`

### Requirement: Backend from config
The client SHALL read the backend from `[llm].backend` in config.toml when no
backend is passed explicitly to the constructor.

#### Scenario: Backend from config when not overridden
- **WHEN** the constructor is called without a `backend` argument and config.toml has `[llm].backend = "llamacpp"`
- **THEN** the client uses the llamacpp adapter and its default structured mode

#### Scenario: Explicit backend overrides config
- **WHEN** the constructor is called with `backend="vllm"`
- **THEN** the client uses the vllm adapter regardless of config.toml's `[llm].backend`

### Requirement: Retry policy at zero temperature
The client SHALL NOT retry on parse validation failures when `temperature` is
`0.0`, because the same input produces the same output.

#### Scenario: Parse failure not retried at temp=0
- **WHEN** `temperature=0.0` and the model returns content that fails validation
- **THEN** the client returns the failure result without retrying, and `attempts=1`

#### Scenario: Transport error retried at temp=0
- **WHEN** `temperature=0.0` and the request fails with a transient transport error (5xx, timeout, connection error)
- **THEN** the client retries up to `max_retries`

### Requirement: Truncation with per-call limits
The client SHALL truncate input text to a token limit that can be specified per
call, with defaults appropriate for `extract_theses`, and SHALL log whether
truncation occurred.

#### Scenario: Text within limit not truncated
- **WHEN** the input text is within the token limit
- **THEN** the text is sent in full and `truncated=false` in the log

#### Scenario: Text exceeding limit truncated
- **WHEN** the input text exceeds the token limit
- **THEN** the text is truncated to the first N tokens and `truncated=true` in the log

### Requirement: Deterministic defaults
The client SHALL default to `temperature=0.0` and `top_p=1.0` to ensure
deterministic model output.

#### Scenario: Default temperature is zero
- **WHEN** the constructor is called without specifying `temperature`
- **THEN** `temperature` is `0.0`

#### Scenario: Default top_p is one
- **WHEN** the constructor is called without specifying `top_p`
- **THEN** `top_p` is `1.0`

### Requirement: Backend detection failure raises error
The `detect_backend` method SHALL raise an error when it cannot identify the
backend from the server's model list, rather than silently defaulting to a
generic OpenAI adapter.

#### Scenario: Unknown owned_by value
- **WHEN** `detect_backend` queries a server whose `owned_by` field is not recognized
- **THEN** it raises an error indicating the backend could not be determined, rather than silently returning a default

### Requirement: Smoke test readiness gate
The project SHALL include a `smoke_test.py` script that verifies the transport
layer end-to-end against a running server and exits with code 0 on success,
non-zero on failure. The smoke test SHALL exercise `extract_theses` calls and
SHALL NOT exercise `classify` calls, because classification is performed through
candidate scoring, not generative LLM calls.

#### Scenario: Successful smoke test
- **WHEN** the server is running and `smoke_test.py` is executed
- **THEN** it performs extract_theses calls, checks the jsonl log, verifies analyze_raw composition, checks determinism, and exits 0

#### Scenario: Smoke test on unreachable server
- **WHEN** the server is not running and `smoke_test.py` is executed
- **THEN** it exits with a non-zero code and an error message

#### Scenario: Structured mode probe
- **WHEN** the smoke test starts
- **THEN** it probes whether the server accepts `json_schema` response_format, falling back to `json_object` if rejected, and logs which mode was accepted

### Requirement: Rich-result call return type
The client SHALL provide a `CallResult` return type that bundles the parsed
model instance, the raw response content, `latency_ms`, `parse_status`,
transport `status`, `finish_reason`, `error`, and `truncated` for a single call.
The existing `extract_theses` / `extract_theses_many` wrappers SHALL remain
unchanged and continue to return the parsed model or `None`.

#### Scenario: CallResult carries parsed and raw
- **WHEN** a rich-result call succeeds with valid JSON matching the schema
- **THEN** the `CallResult` has the parsed model in `parsed`, the raw string in `raw_content`, `parse_status=ok`, and `latency_ms` populated

#### Scenario: CallResult on parse failure
- **WHEN** a rich-result call receives invalid JSON
- **THEN** the `CallResult` has `parsed=None`, `raw_content` holding the raw string, and `parse_status` set to the failure category

#### Scenario: Existing wrappers unchanged
- **WHEN** `extract_theses` is called as before
- **THEN** it returns the parsed model or `None` with no change to its signature or behavior

### Requirement: Rich-result single-call methods
The client SHALL provide `extract_theses_detailed` method that returns a
`CallResult` and writes one jsonl log entry per request through the existing
`_LogWriter` when a `log_path` is configured. This method SHALL use the same
per-call defaults (`max_tokens`, `truncate_tokens`) as its non-detailed sibling.
The client SHALL NOT provide `classify_detailed`.

#### Scenario: extract_theses_detailed uses extraction defaults
- **WHEN** `extract_theses_detailed` is called without explicit `truncate_tokens` or `max_tokens`
- **THEN** it uses extraction defaults (`truncate_tokens=2000`, `max_tokens=512`)

### Requirement: Rich-result batch methods
The client SHALL provide `extract_theses_many_detailed` method that returns a
list of `CallResult` in the same order as the input texts, executes with a
configurable concurrency limit, and logs every request when a `log_path` is
configured. The client SHALL NOT provide `classify_many_detailed`.

#### Scenario: Batch returns rich results in order
- **WHEN** `extract_theses_many_detailed` is called with N texts
- **THEN** it returns N `CallResult` objects whose order matches the input texts

#### Scenario: Batch concurrency honored
- **WHEN** `extract_theses_many_detailed` is called with `concurrency=4`
- **THEN** no more than 4 requests execute simultaneously

#### Scenario: Batch logs every request
- **WHEN** `extract_theses_many_detailed` is called with a `log_path` configured on N texts
- **THEN** exactly N jsonl log entries are written
