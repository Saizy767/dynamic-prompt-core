# Tasks

## 1. Config support

- [x] 1.1 Add `LlamaServerParams` dataclass (`endpoint`, `timeout`) and `"llama_server"` to `VALID_SCORER_BACKENDS` in `infrastructure/llm/scoring/config.py`. Extend `ScorerBackendConfig` to hold `llama_server: LlamaServerParams | None` and extend `scorer_backend_config_from_toml` to parse the `[model.llama_server]` sub-table with defaults (`http://127.0.0.1:8080`, `30`). Verify: unit test that `scorer_backend_config_from_toml` parses a `[model.llama_server]` table and returns correct `endpoint` and `timeout` values, and that defaults are applied when the sub-table is absent.
- [x] 1.2 Add config validation unit tests for the `"llama_server"` backend: invalid endpoint URL rejected, missing `model_path` still validated, default endpoint/timeout applied, existing `"gguf"` and `"huggingface"` backends unaffected. Verify: `pytest tests/unit/ -k "scorer_backend_config"` passes.

## 2. HTTP model adapter

- [x] 2.1 Create `infrastructure/llm/scoring/llama_server_model_adapter.py` with `LlamaServerCausalLMAdapter` implementing the `BatchedCausalLanguageModel` seam. The constructor takes `endpoint`, `timeout`, and optional `served_model_name`. Implement `forward_batch(input_ids, attention_mask)` as an async method that issues one `POST /completion` request per batch item concurrently via `asyncio.gather` with a lazy-initialized `aiohttp.ClientSession` (`TCPConnector`, `ClientTimeout`). Each request sends the unpadded token IDs with `n_predict: 0` and logits enabled. Verify: unit test with an `aiohttp` test server stub returns a `BatchedLogits` with shape `[batch_size, seq_len, vocab_size]` matching the input batch.
- [x] 2.2 Implement the response logit parser: extract the logits array from the server response (checking known field names: `logits`, `completion_probabilities`), validate that the sequence-length dimension matches the input token count and the vocabulary dimension is consistent, convert to `numpy.ndarray` of floats, and raise `LLMScoringError` on missing/malformed logits or shape mismatch. Verify: unit tests with sample response bodies (valid, missing logits, wrong shape, NaN values) — valid parsed correctly, invalid raises `LLMScoringError` with descriptive message.
- [x] 2.3 Implement HTTP error translation: non-200 status → `LLMScoringError` with status code and body excerpt; `aiohttp.ClientConnectorError` / connection refused → `LLMScoringError` with endpoint URL; `asyncio.TimeoutError` → `LLMScoringError` with timeout and endpoint. No raw `aiohttp` exceptions escape the adapter. Verify: unit tests with error responses and connection failures raise `LLMScoringError` with correct detail fields.
- [x] 2.4 Implement `describe()` returning backend metadata (`backend: "llama_server"`, `endpoint`, `served_model_name`, `vocab_size` from first response) and `aclose()` for session cleanup. Verify: unit test that `describe()` returns expected metadata and `aclose()` closes the `aiohttp.ClientSession` without error.

## 3. HTTP tokenizer adapter

- [x] 3.1 Create `infrastructure/llm/scoring/llama_server_tokenizer_adapter.py` with `LlamaServerBatchTokenizerAdapter` implementing the `BatchTokenizerAdapter` seam. The constructor takes `endpoint`, `timeout`. Implement `encode_batch(prefixes, candidate_texts)` by calling `POST /tokenize` for each prefix and candidate separately, composing `prefix_tokens + candidate_tokens`, and recording `prefix_token_count` and `candidate_token_ids` per item. Produce `BatchTokenization` with `input_ids`, `attention_mask` (right-padded), and per-item boundary metadata. Verify: unit test with an `aiohttp` test server stub returns correct `BatchTokenization` with right padding and per-item `prefix_token_count` / `candidate_token_ids`.
- [x] 3.2 Add tokenizer adapter unit tests for boundary determination: candidate value sent verbatim (no strip/normalize), separate encoding of prefix and candidate, BOS/EOS token handling (request no special tokens or account for them), and padding isolation (padding tokens excluded from candidate positions). Verify: `pytest tests/unit/ -k "llama_server_tokenizer"` passes.

## 4. Factory dispatch

- [x] 4.1 Add a `_build_llama_server_adapters(config)` function and `"llama_server"` branch to `build_candidate_scorer` in `infrastructure/llm/scoring/factory.py`, constructing `LlamaServerCausalLMAdapter` and `LlamaServerBatchTokenizerAdapter` from `config.llama_server` and wiring them into `LLMLogitCandidateScorer`. Verify: unit test that `build_candidate_scorer` with `backend="llama_server"` returns a `LLMLogitCandidateScorer` instance, and that `"gguf"` / `"huggingface"` branches are unchanged.
- [x] 4.2 Add a factory unit test that the `"llama_server"` path does not import `llama_cpp` (verify `llama_cpp` is not in `sys.modules` after building the scorer). Verify: `pytest tests/unit/ -k "factory and llama_server"` passes.

## 5. Integration checks

- [x] 5.1 Update `config.toml` comments to document the `"llama_server"` backend option and `[model.llama_server]` sub-table with `endpoint` and `timeout` fields. Add a commented-out example configuration. Verify: `load_raw_config()` successfully parses the updated `config.toml`.
- [x] 5.2 Run `mypy` on all new and modified modules in `infrastructure/llm/scoring/`. Verify: no type errors (`mypy src/dynamic_prompt_core/infrastructure/llm/scoring/` exits 0).
- [x] 5.3 Run `ruff` on all new and modified modules. Verify: no lint violations (`ruff check src/dynamic_prompt_core/infrastructure/llm/scoring/` exits 0).
- [x] 5.4 Run `lint-imports` (import-linter) to verify hexagonal boundary contracts still hold with the new modules. Verify: `lint-imports` exits 0.
- [x] 5.5 Run the full test suite. Verify: all existing tests pass and all new unit tests pass (`pytest tests/` exits 0, excluding `gguf`-marked integration tests that require a live model).
