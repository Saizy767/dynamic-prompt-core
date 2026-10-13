# Proposal

## Why

The GGUF candidate scorer loads the model in-process via `llama-cpp-python`,
which runs forward passes on CPU, blocks the event loop through
`asyncio.to_thread`, and holds a `threading.Lock` + `BoundedSemaphore` around
every `model.eval()` call. On Apple Silicon this wastes the Metal GPU and
causes the validate/convert stages to hang for hundreds of seconds per batch.
A `llama-server` subprocess with `--n-gpu-layers 99` is already launched and
managed by `ServerLauncher` for generative LLM calls — the scorer should reuse
that same HTTP endpoint to get GPU-accelerated logit extraction in
sub-second time.

## What Changes

- Add a new `LlamaServerCausalLMAdapter` (model seam) that calls
  `llama-server`'s `/completion` endpoint with `n_predict: 0` and
  `logits_all: true` to retrieve raw logits over HTTP, returning the same
  `BatchedLogits` shape the orchestrator expects.
- Add a new `LlamaServerBatchTokenizerAdapter` (tokenizer seam) that calls
  `llama-server`'s `/tokenize` endpoint to encode prefix and candidate text,
  preserving the separate-encoding boundary-determination strategy.
- Add a `"llama_server"` backend branch to `build_candidate_scorer` factory
  dispatch and `ScorerBackendConfig` / `scorer_backend_config_from_toml`,
  with config fields for endpoint URL and request timeout.
- Use `aiohttp` (already a core dependency) for HTTP calls, following the
  established `AsyncTask` / `client.py` pattern — no new dependencies.
- **BREAKING** for `[model]` config: the `"gguf"` in-process backend remains
  available, but `"llama_server"` becomes the recommended backend for
  Apple Silicon. The `[model.gguf]` in-process params (`n_ctx`, `n_batch`,
  `n_gpu_layers`, etc.) are not used by the new backend; connection params
  (`endpoint`, `timeout`) replace them.
- Remove the `threading.Lock` / `BoundedSemaphore` contention from the scoring
  path — HTTP I/O is natively async, no thread offloading needed for the
  model seam.
- The `CandidateScorer` port, `LLMLogitCandidateScorer` orchestrator,
  `LogitScorer`, `ScoringPromptBuilder`, and all application/domain code remain
  unchanged.

## Capabilities

### New Capabilities

- `llama-server-logit-scorer`: HTTP-based scoring adapter that implements the
  `BatchedCausalLanguageModel` and `BatchTokenizerAdapter` seams by calling
  `llama-server` REST endpoints (`/completion` for logits, `/tokenize` for
  tokenization), including endpoint configuration, request/response mapping,
  error translation, and async HTTP transport via `aiohttp`.

### Modified Capabilities

- `llm-logit-scorer`: Relax the "Sequential compatibility adapter is temporary"
  requirement to permanently permit remote/HTTP model adapters that delegate
  batch execution to sequential HTTP calls, since `llama-server` does not
  expose a native batched-logit tensor API. The "one `forward_batch` call per
  scoring invocation" contract is preserved at the seam level; the HTTP adapter
  may issue one HTTP request per batch item internally.

## Impact

- **Code**: `infrastructure/llm/scoring/` (new adapter modules, factory
  dispatch, config extension), `infrastructure/llm/scoring/factory.py`,
  `infrastructure/llm/scoring/config.py`. No changes to `application/`,
  `domain/`, or the `CandidateScorer` port.
- **Config**: `config.toml` `[model]` section gains `backend = "llama_server"`
  and connection fields; `[model.gguf]` remains for the in-process fallback.
- **Dependencies**: No new dependencies. `llama-cpp-python` becomes optional
  only for the legacy `"gguf"` backend; the new backend uses core `aiohttp`.
- **Tests**: New unit tests with HTTP mocks for the adapter; the
  `gguf`-marked integration tests remain for the in-process path.
- **Performance**: Forward-pass time expected to drop from hundreds of seconds
  (CPU) to sub-second / low single-digit seconds (Metal GPU). The
  `adapter: lock` metric is eliminated for the HTTP backend.
