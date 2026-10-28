# Proposal

## Why

The in-process GGUF scorer (current default) spends 20-47 s per row converting
full-vocabulary logits to Python floats (validated against run artifacts:
`classify_latency_ms` 19866-46974 ms in `results_..._v1_1_holdout`), keeps 20
of 36 Qwen2.5-3B layers on CPU, and pegs one CPU core while the GPU sits idle.
The HuggingFace backend is CPU-only (no device placement) with the same
pure-Python conversion. Meanwhile a llama-server instance is already running
with `-ngl -1` (all layers on Metal), and live probing validated a
server-side scoring mechanism that returns exact per-token log-probs with no
Python-side logit materialization at all.

## What Changes

- **Rewrite** `LlamaServerCausalLMAdapter` to a `logit_bias`-forced completion
  mechanism: per candidate token, one `POST /v1/completions` request with
  `prompt` as token IDs, `max_tokens: 1`, `logprobs: 1`, `temperature: 0`,
  `logit_bias: {<token_id>: 100}`. The server is forced to generate exactly
  that token and reports its true full-vocabulary log-prob (validated live on
  llama.cpp 0.5.0: forced `0` -> -14.6238, `1` -> -12.167518, byte-identical
  to grammar-forced values). Score semantics are identical to the current
  GGUF scorer (mean true log-prob of candidate tokens).
- **BREAKING**: remove the `"gguf"` and `"huggingface"` scorer backends
  entirely — code, config sections, factory branches, tests, the
  `llama-cpp-python` optional dependency, and the `gguf` pytest marker.
  `"llama_server"` becomes the only scorer backend.
- **BREAKING** for `[model]` config: `backend` and `model_path` keys reduce to
  a single valid backend; `[model.gguf]` is removed. `[model.llama_server]`
  (`endpoint`, `timeout`) becomes the active sub-table. Absent `[model]`
  defaults to `llama_server` with the default endpoint.
- `forward_batch` kwargs `prefix_token_counts` / `candidate_token_ids` become
  required for the server adapter (the classifier already passes them).
- Delete dead code: `SequentialBatchCompatibilityAdapter` (defined, never
  used).
- Add `tests/regression/bench_llama_server.py` (replaces `bench_gguf.py`).
- Verify score equivalence against a GGUF reference capture before deleting
  the GGUF backend.

## Capabilities

### New Capabilities
- `llama-server-logit-scorer`: HTTP-based candidate scoring against a running
  llama-server via `logit_bias`-forced `/v1/completions` requests — exact
  per-token log-probs without top-N windows, floors, or renormalization
  arithmetic; tokenization via `/tokenize`; async aiohttp transport; error
  translation; backend metadata. Supersedes the unarchived
  `migrate-candidate-scorer-to-llama-server` change (whose
  `prompt_logprobs` mechanism is unsupported by llama.cpp 0.5.0 — verified
  live: no `prompt_logprobs` on `/completion`, no prompt logprobs via `echo`).

### Modified Capabilities
- `llm-logit-scorer`: remove the "Sequential compatibility adapter is
  temporary" requirement — the adapter is dead code (defined in
  `model_adapter.py`, referenced nowhere) and is deleted.

## Impact

- `infrastructure/llm/scoring/`: `llama_server_model_adapter.py` (rewrite),
  `config.py` (single backend), `model_adapter.py` (delete Torch adapters +
  compat adapter), `llama_cpp_adapter.py` (delete).
- `infrastructure/llm/classifier_factory.py`: single llama_server branch.
- Call sites of `scorer_backend_config_from_toml`: `run_baseline/runner.py`,
  `compare_versions/comparator.py`, `interfaces/cli/main.py`.
- `config.toml`: `[model]` section rewrite; `[model.gguf]` removed.
- `pyproject.toml`: drop `[gguf]` extra, `llama_cpp.*` mypy override, `gguf`
  pytest marker.
- Tests: rewrite `test_llama_server_model_adapter.py` and
  `test_scorer_backend_config.py`; delete GGUF/HF-specific test modules and
  mocks; trim GGUF parts of `test_runtime_config.py`, `test_instrumentation.py`,
  `test_benchmark.py`.
- OpenSpec: superseded unarchived changes
  (`migrate-candidate-scorer-to-llama-server`, `add-gguf-logit-provider`,
  `gguf-scoring-perf-optimization`) are resolved at archive time.
- Requires a running llama-server with `logit_bias` support on
  `/v1/completions` (validated on Homebrew llama.cpp 0.5.0).
