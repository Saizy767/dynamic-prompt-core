# Spec Delta

## MODIFIED Requirements

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
