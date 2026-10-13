# Design

## Context

The scoring infrastructure (`infrastructure/llm/scoring/`) uses two
infrastructure-internal seams — `BatchedCausalLanguageModel` (model) and
`BatchTokenizerAdapter` (tokenizer) — that `LLMLogitCandidateScorer` orchestrates.
The current GGUF path (`LlamaCppBatchedCausalLMAdapter` +
`LlamaCppBatchTokenizerAdapter`) loads the model in-process via
`llama-cpp-python`, runs `model.eval()` under a `threading.Lock` +
`BoundedSemaphore`, and offloads via `asyncio.to_thread`. A `llama-server`
subprocess is already managed by `ServerLauncher` for generative calls, exposing
an OpenAI-compatible HTTP API at `http://host:port`. The codebase uses `aiohttp`
exclusively for HTTP (no `httpx`/`requests`). See proposal.md for motivation.

## Goals / Non-Goals

**Goals:**
- Implement the model and tokenizer seams over HTTP, reusing the already-running
  `llama-server` endpoint for GPU-accelerated logit extraction.
- Preserve the `CandidateScorer` port, `LLMLogitCandidateScorer` orchestrator,
  `LogitScorer`, and `ScoringPromptBuilder` unchanged.
- Eliminate `threading.Lock` / `BoundedSemaphore` contention and
  `asyncio.to_thread` offloading from the scoring model seam.
- Add no new dependencies — reuse core `aiohttp` and `numpy`.

**Non-Goals:**
- Removing or deprecating the in-process `"gguf"` backend — it remains as a
  fallback for environments without `llama-server`.
- Modifying the `CandidateScorer` port, domain models, or application layer.
- Changing the `ServerLauncher` or server lifecycle management — the scorer
  connects to an already-launched server.
- Implementing a native batched-logit HTTP API — `llama-server`'s `/completion`
  endpoint processes one prompt per request; per-item concurrent requests are
  sufficient.
- Migrating the `"huggingface"` backend — only the GGUF scoring path is affected.

## Decisions

### Decision 1: New adapter modules instead of modifying `llama_cpp_adapter.py`

Create `llama_server_model_adapter.py` and `llama_server_tokenizer_adapter.py`
in `infrastructure/llm/scoring/`, rather than adding HTTP code to
`llama_cpp_adapter.py`.

**Rationale**: The in-process and HTTP adapters share no implementation code
(no shared model instance, no shared locks, no shared tokenization path).
Separate modules keep each adapter focused, simplify testing, and allow the
`[gguf]` optional dependency to be fully isolated.

**Alternative considered**: A single `unified_adapter.py` with backend
dispatch — rejected because it couples two unrelated implementations and
makes the lazy `llama_cpp` import harder to isolate.

### Decision 2: `/completion` with `n_predict: 0` + `logits_all: true` for logits

Call `POST /completion` with the prompt, `n_predict: 0` (no generation), and
logits exposure enabled. The server performs one forward pass and returns
logits for all input positions.

**Rationale**: This is the canonical `llama-server` way to get logits without
generation. `n_predict: 0` ensures no wasted computation. The response includes
a `logits` array (or `completion_probabilities` depending on server version)
covering all input tokens.

**Alternative considered**: A dedicated `/logits` or `/evaluate` endpoint —
not available in current `llama-server` builds. The `/completion` endpoint with
`n_predict: 0` is the stable, documented approach.

### Decision 3: `/tokenize` endpoint for tokenization

Call `POST /tokenize` with the text to encode. The server returns token IDs
using the same tokenizer/model the scorer uses for inference, guaranteeing
tokenization consistency.

**Rationale**: Using the server's tokenizer ensures the token IDs sent to
`/completion` match exactly. A local `transformers.AutoTokenizer` could
diverge from the GGUF model's tokenizer on edge cases (BPE merges, special
tokens).

**Alternative considered**: Reuse `transformers.AutoTokenizer` from the model
path — rejected because of potential tokenizer mismatch between the
HuggingFace tokenizer and the GGUF model's internal tokenizer.

### Decision 4: `aiohttp` for HTTP transport

Use `aiohttp.ClientSession` with `TCPConnector(limit=...)` and
`ClientTimeout(total=...)`, following the established `AsyncTask` / `client.py`
pattern.

**Rationale**: `aiohttp` is already a core dependency and is used everywhere
else for HTTP LLM calls. Introducing `httpx` or `requests` would add a
dependency and diverge from the established pattern. `aiohttp` is natively
async, so no `asyncio.to_thread` offloading is needed.

### Decision 5: `forward_batch` via `asyncio.gather` of per-item requests

Implement `forward_batch(items)` by issuing one `/completion` request per batch
item concurrently with `asyncio.gather`, then assembling the results into a
single `BatchedLogits` with shape `[B, S, V]`.

**Rationale**: `llama-server`'s `/completion` endpoint processes one prompt per
request — there is no native batched-logit tensor API. Concurrent requests
leverage the server's internal batching/thread pool. The seam-level contract
("one `forward_batch` call per scoring invocation") is preserved; only the
internal implementation changes. This is the permanent strategy for the HTTP
adapter, as permitted by the modified "Sequential compatibility adapter"
requirement.

**Alternative considered**: Sequential per-item requests — rejected because
concurrent requests better utilize the server and reduce wall-clock latency.

### Decision 6: Extend `ScorerBackendConfig` with `"llama_server"` backend

Add `"llama_server"` to `VALID_SCORER_BACKENDS`, add a `LlamaServerParams`
dataclass (`endpoint`, `timeout`) to `ScorerBackendConfig`, and extend
`scorer_backend_config_from_toml` to parse `[model.llama_server]` sub-table.

**Rationale**: Keeps all scorer backends behind one config type and one
factory. The existing `"gguf"` and `"huggingface"` paths are untouched.

**Alternative considered**: A separate config section `[scoring.llama_server]`
— rejected because `[model]` is the established scorer config section.

### Decision 7: Shared `aiohttp.ClientSession` lifecycle

The model adapter owns a lazy-initialized `aiohttp.ClientSession` (created on
first `forward_batch` call, reused across calls). The session is closed via an
explicit `aclose()` method called by the factory or composition root during
shutdown.

**Rationale**: Creating a session per request would defeat connection pooling.
Following `teacher_client.py`'s module-level `_get_session()` pattern ensures
reuse. The explicit `aclose()` avoids relying on garbage collection for
connection cleanup.

## Risks / Trade-offs

- **[Network latency per item]** Each batch item is a separate HTTP round-trip.
  → Mitigated by `asyncio.gather` concurrency and the server's internal
  thread pool. For typical candidate counts (2–10), the concurrent wall-clock
  cost is dominated by GPU forward time, not network overhead.

- **[Server must be running]** The scorer fails if `llama-server` is not
  started. → The readiness check (`wait_for_ready`) already gates server
  startup. The adapter's error message will name the endpoint and suggest
  the launch command with `--n-gpu-layers 99`.

- **[Logit response format varies by server version]** Different
  `llama-server` versions may use `logits`, `completion_probabilities`, or
  nested structures. → The response parser will check for known field names
  and raise a clear `LLMScoringError` if none are found, with the response
  body excerpt for debugging.

- **[Tokenization boundary mismatch]** The `/tokenize` endpoint may add BOS
  or EOS tokens that the in-process adapter did not, shifting token positions.
  → The adapter will explicitly request no special tokens (or account for
  them) and validate that `prefix_tokens + candidate_tokens` reconstructs the
  expected full sequence length. Unit tests with HTTP mocks will verify
  boundary alignment against known tokenizations.

- **[No `describe()` model checksum]** The HTTP adapter cannot compute a
  SHA-256 model checksum without reading the GGUF file. → `describe()` will
  report the endpoint URL and served model alias instead, which is sufficient
  for run identity when the server is the single model source.

## Migration Plan

1. Add the new adapter modules, config support, and factory branch behind the
   `"llama_server"` backend value — no existing behavior changes.
2. Add unit tests with `aiohttp` test server / mocks for the new adapters.
3. Update `config.toml` documentation (comments) to show the
   `"llama_server"` backend option.
4. Switch `[model].backend` to `"llama_server"` in `config.toml` when ready.
5. **Rollback**: Set `[model].backend` back to `"gguf"` — no code changes
   needed, the in-process adapter and `[gguf]` extra remain intact.

## Open Questions

- Does the target `llama-server` version return logits under the `logits` key
  or `completion_probabilities` key? This affects only the response parser
  field name, not the architecture. Can be resolved during implementation by
  inspecting a live server response.
- Should the scorer reuse the `[llm].host`/`[llm].port` config values (the
  generative server's endpoint) by default, or require an explicit
  `[model.llama_server].endpoint`? Defaulting to the `[llm]` endpoint reduces
  config duplication but couples the two sections. For now, an explicit
  `[model.llama_server].endpoint` with a sensible default
  (`http://127.0.0.1:8080`) is chosen; unification can be revisited later
  without changing the specs.
