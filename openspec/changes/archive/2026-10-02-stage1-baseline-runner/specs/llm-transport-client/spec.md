# Spec Delta

## ADDED Requirements

### Requirement: Rich-result call return type
The client SHALL provide a `CallResult` return type that bundles the parsed
model instance, the raw response content, `latency_ms`, `parse_status`,
transport `status`, `finish_reason`, `error`, and `truncated` for a single call.
The existing `classify` / `extract_theses` / `classify_many` /
`extract_theses_many` wrappers SHALL remain unchanged and continue to return the
parsed model or `None`.

#### Scenario: CallResult carries parsed and raw
- **WHEN** a rich-result call succeeds with valid JSON matching the schema
- **THEN** the `CallResult` has the parsed model in `parsed`, the raw string in `raw_content`, `parse_status=ok`, and `latency_ms` populated

#### Scenario: CallResult on parse failure
- **WHEN** a rich-result call receives invalid JSON
- **THEN** the `CallResult` has `parsed=None`, `raw_content` holding the raw string, and `parse_status` set to the failure category

#### Scenario: Existing wrappers unchanged
- **WHEN** `classify` or `extract_theses` is called as before
- **THEN** it returns the parsed model or `None` with no change to its signature or behavior

### Requirement: Rich-result single-call methods
The client SHALL provide `classify_detailed` and `extract_theses_detailed`
methods that return a `CallResult` and write one jsonl log entry per request
through the existing `_LogWriter` when a `log_path` is configured. These methods
SHALL use the same per-call defaults (`max_tokens`, `truncate_tokens`) as their
non-detailed siblings.

#### Scenario: classify_detailed logs and returns rich result
- **WHEN** `classify_detailed` is called with a `log_path` configured
- **THEN** exactly one jsonl log entry is written and a `CallResult` is returned

#### Scenario: extract_theses_detailed uses extraction defaults
- **WHEN** `extract_theses_detailed` is called without explicit `truncate_tokens` or `max_tokens`
- **THEN** it uses extraction defaults (`truncate_tokens=2000`, `max_tokens=512`)

### Requirement: Rich-result batch methods
The client SHALL provide `classify_many_detailed` and
`extract_theses_many_detailed` methods that return a list of `CallResult` in the
same order as the input texts, execute with a configurable concurrency limit, and
log every request when a `log_path` is configured.

#### Scenario: Batch returns rich results in order
- **WHEN** `classify_many_detailed` is called with N texts
- **THEN** it returns N `CallResult` objects whose order matches the input texts

#### Scenario: Batch concurrency honored
- **WHEN** `classify_many_detailed` is called with `concurrency=4`
- **THEN** no more than 4 requests execute simultaneously

#### Scenario: Batch logs every request
- **WHEN** `classify_many_detailed` is called with a `log_path` configured on N texts
- **THEN** exactly N jsonl log entries are written
