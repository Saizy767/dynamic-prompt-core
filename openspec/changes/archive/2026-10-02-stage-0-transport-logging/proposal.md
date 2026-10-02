# Proposal

## Why

The project's next stages (parsers, runner, metrics, thesis bank) all depend on
a stable transport and logging layer for LLM requests. The current `asyncTask.py`
mixes transport and parsing in a single `analyze` method, has no structured
logging, ignores `[llm].backend` from config (silently defaults to vLLM),
discards `finish_reason`, and retries parse failures even at `temperature=0`
where retry is pointless. These gaps must be closed before any downstream stage
can rely on the client.

## What Changes

- **New `RawResponse` dataclass** separating transport from parsing:
  `analyze_raw` returns raw content + metadata without parsing; `analyze`
  composes parsing on top.
- **`classify` / `extract_theses` wrappers** (and batch variants
  `classify_many` / `extract_theses_many`) that pass `call_type` into logs and
  use per-call defaults for `system_prompt`, `max_tokens`, and
  `truncate_tokens`.
- **jsonl logging** via `log_path` in the constructor, one line per logical
  request, written under `asyncio.Lock`. Fields include `run_id` (default
  `"smoke"`), `call_type`, `parse_status`, `truncated`, `finish_reason`, and
  `usage`.
- **`parse_status` taxonomy**: `ok`, `invalid_json`, `schema_mismatch`,
  `truncated` — distinguishes failure modes that retry cannot fix.
- **Backend read from `[llm].backend`** in the constructor when not passed
  explicitly. **BREAKING** for callers that relied on the `vllm` default: the
  constructor now respects the config's backend.
- **Retry on `ValidationError` disabled at `temperature=0`** — same input
  produces the same output, so retry is pure waste.
- **Defaults changed**: `temperature=0.0`, `top_p=1.0` (was `0.1` / `0.9`).
- **`detect_backend` fallback** raises `ConfigError` instead of silently
  defaulting to `OPENAI` when `owned_by` is unrecognized.
- **`smoke_test.py`**: two calls (classify, extract_theses) with placeholder
  Pydantic schemas, `analyze_raw` composition check, determinism check,
  truncation check, structured-mode probe, log verification, exit code.

## Capabilities

### New Capabilities

- `llm-transport-client`: Structured transport, parsing, and logging for LLM
  requests via OpenAI-compatible servers (llama.cpp, vLLM). Covers the
  transport/parsing split, call-type wrappers, jsonl logging, retry policy,
  truncation, and the smoke test.

### Modified Capabilities

None. The existing `llm-server-launcher` spec covers server lifecycle, not the
client.

## Impact

- **`asyncTask.py`**: major changes — new `RawResponse` dataclass,
  `analyze_raw` / `analyze` split, `classify` / `extract_theses` wrappers, jsonl
  logging, backend-from-config, retry policy change, default changes.
- **`config.toml`**: no schema change (backend key already exists); the client
  now reads it.
- **New file `smoke_test.py`**: standalone readiness gate script.
- **Dependencies**: no new dependencies (aiohttp, pydantic v2, transformers
  already in use).
- **API compatibility**: `analyze` and `analyze_many` signatures change
  (`system_prompt` and `max_tokens` become per-call parameters on the new
  wrappers; the constructor retains them as defaults). `AsyncTask.__init__` no
  longer defaults to `Backend.VLLM` — it reads `[llm].backend`.
