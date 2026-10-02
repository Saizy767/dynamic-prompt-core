# Design

## Context

See proposal.md for motivation. The current `asyncTask.py` (455 lines) has a
single `analyze` method that does HTTP transport, JSON parsing, and retry in
one loop. `system_prompt` and `max_tokens` are constructor-level, which blocks
the optimization cycle (which varies the classification prompt per round). The
constructor defaults `backend=Backend.VLLM` and never reads
`config["llm"]["backend"]`. `_post_once` discards `finish_reason`.
`_is_retryable` returns `True` for `ValidationError` regardless of temperature.

The existing `server_launcher.py` and `config.toml` are unchanged by this
design. The `llm-server-launcher` spec already covers server lifecycle.

## Goals / Non-Goals

**Goals:**

- Separate transport (HTTP + retry) from parsing (JSON validation) so downstream
  stages can inspect raw responses and failure modes.
- Provide call-type wrappers (`classify` / `extract_theses`) with per-call
  `system_prompt`, `max_tokens`, and `truncate_tokens` — the API the
  optimization cycle needs.
- Add structured jsonl logging with enough fields to debug failures without
  re-running.
- Fix the backend-from-config bug and the wasteful retry-on-parse-failure at
  temp=0.
- Provide a smoke test that verifies the layer end-to-end.

**Non-Goals:**

- Real Pydantic schemas for classification or extraction (stage 2).
- Dataset runner, metrics, thesis bank, clustering, rule selection (stages 2+).
- The 5-round optimization cycle (stage 2+).
- Caching, MCP, external services, databases.
- Per-attempt logging (deferred to stage 2+ if retry storms appear).
- Pydantic validation of log records on write (deferred to stage 2+ when
  metrics read the log programmatically).

## Decisions

### D1: RawResponse as a plain dataclass, not a Pydantic model

`RawResponse` is a `@dataclass` with fields: `content`, `model`, `latency_ms`,
`status`, `attempts`, `finish_reason`, `usage`, `http_status`, `error`.

**Rationale**: Pydantic validation on every response adds overhead on the hot
path. `RawResponse` is internal transport metadata, not a user-facing schema.
A dataclass is sufficient and avoids coupling transport to Pydantic's
validation cost.

**Alternative**: Pydantic `BaseModel` with `model_config = ConfigDict(arbitrary_types_allowed=True)`. Rejected — unnecessary validation overhead for an internal type.

### D2: analyze_raw does transport + retry; analyze parses on top

```
analyze_raw(session, text, model_cls, call_type, **call_params)
    -> RawResponse
        |
        |  HTTP POST (with retry on transport errors)
        |  NO parsing here
        v
    RawResponse(status=ok|error, content=raw_str, ...)

analyze(session, text, model_cls, call_type, **call_params)
    -> Optional[model_cls]
        |
        |  raw = analyze_raw(...)
        |  if raw.status != ok: return None
        |  parse raw.content -> model_cls
        |  set parse_status on RawResponse/log
        v
    parsed instance or None
```

**Rationale**: The split lets the smoke test verify transport independently and
lets stages 2+ inspect `RawResponse` for failure mode without re-running. The
retry loop stays in `analyze_raw` because retry is a transport concern (retry
on parse failure is disabled at temp=0 — see D5).

### D3: Per-call parameters via wrappers, constructor as fallback

```
classify(session, text, model_cls,
         system_prompt=None,    # -> constructor default
         max_tokens=None,       # -> 128
         truncate_tokens=None,  # -> 300
         **kwargs) -> Optional[model_cls]

extract_theses(session, text, model_cls,
               system_prompt=None,    # -> constructor default
               max_tokens=None,       # -> 512
               truncate_tokens=None,  # -> 2000
               **kwargs) -> Optional[model_cls]
```

`None` means "use the wrapper's default for this call type." The constructor's
`system_prompt` is the fallback when the wrapper default is also `None`.

**Rationale**: The optimization cycle varies `system_prompt` per round and
needs different `max_tokens` / `truncate_tokens` for classify vs. extract.
Making these per-call avoids recreating `AsyncTask` (which reloads the
tokenizer) each round.

**Alternative**: Keep constructor-level only and recreate `AsyncTask` per
round. Rejected — tokenizer reload is wasteful and the API is awkward for the
cycle.

### D4: jsonl logging via a _LogWriter helper with asyncio.Lock

```
_LogWriter(log_path: str, run_id: str)
    ._lock: asyncio.Lock
    .write(entry: dict) -> None   # json.dumps + append, under lock
```

One line per logical request. The writer is created in `AsyncTask.__init__`
when `log_path` is provided. `run_id` defaults to `"smoke"`.

**Rationale**: `asyncio.Lock` prevents interleaved writes from concurrent
batch tasks. A dedicated helper keeps the logging concern out of the transport
methods. One file per run (not one global file) so parallel runs don't
interleave.

**Alternative**: `aiofiles` for non-blocking writes. Rejected for stage 0 —
adds a dependency, and with `asyncio.Lock` the write is sequential anyway.
For stage 2+ with 1000s of concurrent writes, revisit with a queue-based
writer.

### D5: Retry policy — no retry on ValidationError at temp=0

```python
def _is_retryable(exc, temperature):
    if isinstance(exc, ValidationError):
        return temperature > 0.0  # only retry parse failures if stochastic
    if isinstance(exc, (ClientConnectionError, ServerTimeoutError, TimeoutError)):
        return True
    if isinstance(exc, HttpError):
        return exc.status >= 500 or exc.status == 429
    return False
```

**Rationale**: At `temperature=0.0`, the model is deterministic — same input
yields the same output. Retrying a parse failure just wastes `max_retries`
requests. At `temperature>0`, retry makes sense because the model may emit
different content.

### D6: Backend resolution priority

```
1. Explicit constructor argument    (highest)
2. [llm].backend from config.toml
3. Auto-detect via GET /v1/models    (last resort, raises on failure)
```

The constructor default for `backend` changes from `Backend.VLLM` to `None`.
When `None`, read `self._config["llm"]["backend"]`. If config lacks it, raise
`ConfigError`.

`detect_backend` raises `ConfigError` on unrecognized `owned_by` instead of
returning `Backend.OPENAI`.

**Rationale**: This project only uses llama.cpp and vLLM. Silently picking the
OpenAI adapter changes `structured_mode` defaults (JSON_SCHEMA vs JSON_OBJECT)
and drops `chat_template_kwargs` — a real behavioral difference.

### D7: parse_status determined by finish_reason + JSON validation

```
if status != ok:
    parse_status = None  (no parsing attempted)
elif finish_reason == "length" and json.loads fails:
    parse_status = "truncated"
elif json.loads fails:
    parse_status = "invalid_json"
elif model_validate_json fails:
    parse_status = "schema_mismatch"
else:
    parse_status = "ok"
```

**Rationale**: `truncated` is the most actionable failure — it means
`max_tokens` is too small, not that the model or prompt is bad. Distinguishing
it from `invalid_json` prevents misdiagnosing a config problem as a prompt
quality problem in stages 2+.

### D8: Smoke test as a standalone script, not pytest

`smoke_test.py` is run directly (`python smoke_test.py`). It assumes a server
is already running (started via `server_launcher.py`). It uses placeholder
Pydantic models with the expected shape (decision, theses, confidence).

**Rationale**: The smoke test is a manual readiness gate, not a CI test. It
needs a running server with a loaded model. pytest tests with mock servers for
retry/failure behavior are a separate concern (stage 0 doesn't gate on them).

## Risks / Trade-offs

- **[Tokenizer/GGUF version mismatch]** `AutoTokenizer.from_pretrained(model_path)` and the GGUF served by llama-server may come from different model versions, causing silent truncation at wrong token boundaries. → Mitigation: document the invariant that `model_path` and `gguf_path` must come from the same model. No code-level check possible.

- **[max_tokens=512 too small for extraction]** If theses are long or numerous, 512 tokens may truncate the JSON. → Mitigation: `max_tokens` is per-call and can be increased without code changes. The `truncated` parse_status makes the failure visible.

- **[asyncio.Lock blocks event loop on slow disk]** Under high concurrency with slow disk, the log write lock could stall the event loop. → Mitigation: acceptable for stage 0 (2 calls). For stage 2+, switch to a queue-based async writer.

- **[Numerical nondeterminism with concurrency>1]** Server-side batching in llama.cpp can introduce floating-point differences at `temp=0`, affecting `confidence` at ~1e-4 level. → Mitigation: smoke test uses `concurrency=1`. Document that `concurrency>1` may cause tiny variations in float fields.

- **[config_path is relative]** `config_path` defaults to `"config.toml"` (relative to cwd). → Mitigation: resolve to absolute path in `__init__` using `os.path.abspath`. Low effort, prevents breakage when invoked from a different directory.
