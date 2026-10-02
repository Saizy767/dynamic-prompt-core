# Tasks

## 1. RawResponse and transport/parsing split

- [x] 1.1 Add `RawResponse` dataclass to `asyncTask.py` with fields `content`, `model`, `latency_ms`, `status`, `attempts`, `finish_reason`, `usage`, `http_status`, `error`. Verify by inspecting the dataclass fields match the spec.
- [x] 1.2 Refactor `_post_once` to capture `finish_reason` and `usage` from the API response and return raw content + metadata without parsing. Verify `finish_reason` is no longer discarded by checking the returned data includes it.
- [x] 1.3 Implement `analyze_raw` that does HTTP transport with retry, returns a `RawResponse`, and does NOT parse content. Verify it returns a `RawResponse` with `content` as a string, not a parsed object.
- [x] 1.4 Refactor `analyze` to call `analyze_raw` then parse the content through the model's validator. Verify `analyze(text, model)` equals `model.model_validate_json(analyze_raw(text, model).content)` for a successful request.
- [x] 1.5 Add a `ResponseStatus` enum (`ok`, `empty`, `http_error`, `timeout`, `network_error`, `unexpected_shape`) used by `RawResponse.status`. Verify the enum values match the design.

## 2. Backend and defaults fixes

- [x] 2.1 Change `AsyncTask.__init__` backend default from `Backend.VLLM` to `None`; when `None`, read `self._config["llm"]["backend"]`. Raise `ConfigError` if the key is missing. Verify that constructing `AsyncTask` without `backend` uses the config's `llamacpp` adapter.
- [x] 2.2 Change constructor defaults: `temperature=0.0`, `top_p=1.0` (was `0.1` / `0.9`). Verify the new defaults are applied when no explicit values are passed.
- [x] 2.3 Change `detect_backend` to raise `ConfigError` on unrecognized `owned_by` instead of returning `Backend.OPENAI`. Verify that an unknown `owned_by` raises an error.
- [x] 2.4 Resolve `config_path` to an absolute path in `__init__` via `os.path.abspath`. Verify the client works when invoked from a different working directory.

## 3. Retry policy

- [x] 3.1 Update `_is_retryable` to accept `temperature` as a parameter and return `False` for `ValidationError` when `temperature == 0.0`. Verify that at `temp=0` a parse failure results in `attempts=1` with no retry.
- [x] 3.2 Verify that transient transport errors (5xx, timeout, connection error) are still retried at `temp=0` up to `max_retries`.

## 4. Parse status taxonomy

- [x] 4.1 Add a `ParseStatus` enum (`ok`, `invalid_json`, `schema_mismatch`, `truncated`) to `asyncTask.py`. Verify the enum values match the spec.
- [x] 4.2 Implement `parse_status` determination in `analyze`: check `finish_reason == "length"` + JSON parse failure → `truncated`; JSON parse failure alone → `invalid_json`; valid JSON but schema validation failure → `schema_mismatch`; success → `ok`. Verify each path produces the correct `parse_status` value.

## 5. Truncation with per-call limits

- [x] 5.1 Make `truncate_tokens` a per-call parameter on `analyze_raw` and `analyze`, falling back to the constructor value when `None`. Verify that passing an explicit `truncate_tokens` overrides the constructor default.
- [x] 5.2 Track whether truncation occurred (`truncated: bool`) during `_truncate` and propagate it to the log entry. Verify `truncated=true` appears in the log when input exceeds the limit, and `truncated=false` when it does not.

## 6. Call-type wrappers

- [x] 6.1 Implement `classify(session, text, model_cls, system_prompt=None, max_tokens=128, truncate_tokens=300, **kwargs)` that calls `analyze` with `call_type="classify"`. Verify it uses the classification defaults when no explicit values are passed.
- [x] 6.2 Implement `extract_theses(session, text, model_cls, system_prompt=None, max_tokens=512, truncate_tokens=2000, **kwargs)` that calls `analyze` with `call_type="extract_theses"`. Verify it uses the extraction defaults when no explicit values are passed.
- [x] 6.3 Implement `classify_many` and `extract_theses_many` batch variants that call `analyze_many` with the appropriate `call_type` and defaults. Verify the batch variants pass `call_type` into each log entry.
- [x] 6.4 Verify that an explicit `system_prompt` passed to a wrapper overrides the constructor's `system_prompt`.

## 7. jsonl logging

- [x] 7.1 Implement a `_LogWriter` helper class with `asyncio.Lock` that appends one jsonl line per call to `write(entry: dict)`. Verify concurrent writes do not interleave by checking line integrity in the output file.
- [x] 7.2 Add `log_path: Optional[str] = None` and `run_id: str = "smoke"` to `AsyncTask.__init__`. Create a `_LogWriter` instance when `log_path` is provided. Verify `run_id` defaults to `"smoke"`.
- [x] 7.3 Write one log entry per logical request (not per retry attempt) with fields: `timestamp`, `run_id`, `call_type`, `system_prompt_hash`, `text_hash`, `model_name`, `params`, `raw_content`, `latency_ms`, `finish_reason`, `usage`, `status`, `parse_status`, `attempts`, `http_status`, `error`, `truncated`. Verify a completed request produces exactly one jsonl line with all required fields present.

## 8. Smoke test

- [x] 8.1 Create `smoke_test.py` with placeholder Pydantic models for the classifier shape (`decision: int`, `theses: list[str]`, `confidence: float`). Verify the models instantiate and validate a sample JSON.
- [x] 8.2 Add a structured-mode probe that sends a minimal request with `response_format={"type": "json_schema", ...}`, falls back to `json_object` on 400, and prints which mode was accepted. Verify the probe correctly reports the server's supported mode.
- [x] 8.3 Add `classify` and `extract_theses` calls against the running server, an `analyze_raw` composition check (verify `analyze` result equals parsing `analyze_raw` content), a determinism check (same input twice → identical results), and a truncation check (long input → `truncated=true` in log). Verify all checks pass against a live server.
- [x] 8.4 Add jsonl log verification (entries exist, required fields present, `status=ok`, `parse_status=ok`) and exit code logic (0 on success, non-zero on any failure). Verify the script exits 0 on a healthy server and non-zero when the server is unreachable.
- [x] 8.5 Run `python smoke_test.py` against a live `llama-server` started via `python server_launcher.py --config config.toml`. Verify exit code 0, log file has entries with correct fields, and results are deterministic across two runs.
