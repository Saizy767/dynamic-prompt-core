# Tasks

## 1. GGUF reference capture (before any deletion)

- [x] 1.1 Add `tools/capture_gguf_reference.py`: loads ~20 rows from the prepared dataset artifact, builds the classifier from the current `config.toml` (GGUF backend), classifies each row with a fixed rule set, and writes `data/results/gguf_reference_scores.json` (row id -> per-candidate scores). Verify: the artifact exists with 20 entries and non-empty `judgment_scores`.
- [x] 1.2 Run the capture and record wall time. Verify: `data/results/gguf_reference_scores.json` written; capture log shows per-row classify latency consistent with the known GGUF baseline.

## 2. Server model adapter rewrite

- [x] 2.1 Rewrite `LlamaServerCausalLMAdapter` in `src/dynamic_prompt_core/infrastructure/llm/scoring/llama_server_model_adapter.py`: per item, per candidate token `k`, one `POST /v1/completions` with `prompt = ids[:prefix_count+k]`, `max_tokens: 1`, `logprobs: 1`, `temperature: 0`, `logit_bias {<c_k>: 100}`; parse `choices[0].logprobs.content[0]`; verify returned `id == c_k` else `LLMScoringError`; build full-length `SequenceLogits` with `SparseLogitVector` at scored positions and floor vectors elsewhere. `prefix_token_counts` / `candidate_token_ids` are required kwargs (explicit error when missing). Keep the aiohttp session pattern, bounded retries with backoff on disconnect/connector errors, semaphore, `aclose()`, `describe()` (backend, endpoint, forcing window). Delete the `prompt_logprobs` parsers and native/v1 fallback chain. Verify: unit tests below pass.
- [x] 2.2 Rewrite `tests/unit/infrastructure/llm/scoring/test_llama_server_model_adapter.py` with a mocked aiohttp transport: forced-token happy path, identity-mismatch error, multi-token iterative requests (prompt grows per step), missing-kwargs error, transient disconnect retried, non-200 -> `LLMScoringError` with status and excerpt, `describe()`/`aclose()` behavior. Verify: `pytest tests/unit/infrastructure/llm/scoring/test_llama_server_model_adapter.py` passes.

## 3. Config: single backend

- [x] 3.1 In `src/dynamic_prompt_core/infrastructure/llm/scoring/config.py`: `VALID_SCORER_BACKENDS = ("llama_server",)`; delete `GgufParams`, the `gguf` field and its validation, and `model_path` from `ScorerBackendConfig`; absent `[model]` defaults to `llama_server` with default endpoint; keep endpoint/timeout validation. Update `scorer_backend_config_from_toml` (drop `default_model_path`, drop `[model.gguf]` parsing). Verify: `pytest tests/unit/infrastructure/config/test_scorer_backend_config.py` (rewritten in 3.2) passes.
- [x] 3.2 Rewrite `tests/unit/infrastructure/config/test_scorer_backend_config.py` for the single-backend config: valid `llama_server` parse, defaults without `[model]`, invalid endpoint rejected, non-positive timeout rejected, any other backend name rejected with a clear error. Delete GGUF-specific cases. Verify: `pytest tests/unit/infrastructure/config/test_scorer_backend_config.py` passes.
- [x] 3.3 Update the three `scorer_backend_config_from_toml` call sites (`application/use_cases/run_baseline/runner.py`, `application/use_cases/compare_versions/comparator.py`, `interfaces/cli/main.py`) to the new signature. Verify: `rg "scorer_backend_config_from_toml" src/` shows no `default_model_path` usage; mypy clean on the three files.
- [x] 3.4 Update `config.toml`: `[model] backend = "llama_server"`; remove `model_path` and the `[model.gguf]` table; uncomment `[model.llama_server]` with `endpoint`/`timeout`; replace the NOTE comment (llama_server now scores via `logit_bias` forcing; validated on llama.cpp 0.5.0). Verify: `python -c "import tomllib; tomllib.load(open('config.toml','rb'))"` parses and `[model.gguf]` is absent.

## 4. Factory: single path

- [x] 4.1 In `src/dynamic_prompt_core/infrastructure/llm/classifier_factory.py`: delete `_build_gguf_adapters` and `_build_huggingface_adapters` and their dispatch branches; `build_classifier` constructs only the llama_server adapters from `config.llama_server`. Verify: `pytest tests/unit/infrastructure/llm/test_classifier_factory.py` (updated in 4.2) passes.
- [x] 4.2 Update factory tests: single-path dispatch, llama_server adapters wired, no `llama_cpp`/`transformers` scoring imports. Verify: `pytest tests/unit/infrastructure/llm/ -k factory` passes.

## 5. Delete GGUF and HuggingFace scoring paths

- [x] 5.1 Delete `src/dynamic_prompt_core/infrastructure/llm/scoring/llama_cpp_adapter.py` and remove from `src/dynamic_prompt_core/infrastructure/llm/scoring/model_adapter.py`: `TorchCausalLMAdapter`, `TorchBatchedCausalLMAdapter`, `SequentialBatchCompatibilityAdapter` (keep protocols, `SequenceLogits`, `BatchedLogits`, `SparseLogitVector`). Verify: `rg "llama_cpp|TorchCausalLM|SequentialBatchCompatibility" src/` returns nothing.
- [x] 5.2 Delete GGUF/HF test modules and fixtures: `test_llama_cpp_model_adapter.py`, `test_llama_cpp_tokenizer_adapter.py`, `test_gguf_security.py`, `test_selective_logits.py`, `test_gguf_structural_equivalence.py`, `test_real_model_opt_in.py`, `test_prefix_reuse.py`, `test_concurrency.py`, `_gguf_mocks.py`, `tests/regression/bench_gguf.py`; trim GGUF cases from `test_runtime_config.py`, `test_instrumentation.py`, `test_benchmark.py`; fix any `test_architectural_invariants.py` references. Verify: `pytest tests/unit/infrastructure/llm/scoring/` passes.
- [x] 5.3 In `pyproject.toml`: remove the `[gguf]` optional dependency, the `llama_cpp.*` mypy override, and the `gguf` pytest marker. Verify: `rg "llama_cpp|gguf" pyproject.toml` returns nothing; `pytest --markers` has no `gguf` marker.

## 6. Equivalence verification and benchmark

- [x] 6.1 Add `tools/verify_server_equivalence.py`: loads `data/results/gguf_reference_scores.json`, classifies the same rows with the same rules via the new server backend, and compares per-candidate probabilities — decision match is the hard criterion, probability delta bounded at 0.1 (measured cross-backend float32 kernel drift: up to 0.064, decisions identical on all 20 rows). Verify: run against the live llama-server; prints max |Δ| and mismatch count. Result: 20/20 rows, 0 mismatches, mean latency 0.33s vs GGUF 8.90s.
- [x] 6.2 Add `tests/regression/bench_llama_server.py` (env `LLAMA_SERVER_ENDPOINT`): same shape as the old GGUF bench — init time, mean/max per-example latency, throughput. Verify: run it; record mean latency and compare against the GGUF baseline from task 1.2 (expect seconds -> sub-second per row).

## 7. Integration gates

- [x] 7.1 Run the full gate suite. Results: `pytest tests/` 421 passed; mypy clean on all touched modules (infrastructure/llm: 0 errors; full tree 4 pre-existing errors in loader.py/steps.py that also fail at HEAD — down from 11 at HEAD); ruff clean on all touched files (full tree 15 pre-existing errors in untouched files, down from 25 at HEAD); lint-imports no-cyclic-imports broken pre-existing at HEAD (verified via worktree: same cycle extractor->application.schemas + factory->ports), no new cross-layer imports introduced by this change.
- [x] 7.2 End-to-end smoke: BaselineRunner driven with a rules-bearing prompt artifact (v1 rule), 10 dev rows, skip_extraction. Artifact results_smoke-server_v1-smoke_dev_*.jsonl: 10/10 rows with populated judgment_scores, classify_latency_ms mean 771 / max 1871 (GGUF baseline 20000-47000).

## 8. OpenSpec resolution

- [x] 8.1 Removed the superseded unarchived change dirs `migrate-candidate-scorer-to-llama-server`, `add-gguf-logit-provider`, and `gguf-scoring-perf-optimization` (their behavior is replaced/deleted by this change; git history preserves them). Verify: `openspec validate server-only-candidate-scoring` passes; `openspec list` shows no superseded changes.
