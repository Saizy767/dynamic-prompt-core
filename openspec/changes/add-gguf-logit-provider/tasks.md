# Tasks

## 1. Optional dependency and type-check scaffolding

- [x] 1.1 Add `llama-cpp-python` (version-pinned) to
  `[project.optional-dependencies].gguf` in `pyproject.toml` and an optional
  section in `requirements.txt`. Verify `pip install -e .` (without extras)
  still succeeds and `python -c "import dynamic_prompt_core"` works without
  `llama_cpp` installed.
- [x] 1.2 Add a `[[tool.mypy.overrides]]` entry for `module = "llama_cpp.*"`
  with `ignore_missing_imports = true` in `pyproject.toml`. Verify
  `mypy` passes on the package with no `llama_cpp` installed.

## 2. GGUF error class and model adapter

- [x] 2.1 Add `GGUFProviderError(LLMScoringError)` to
  `infrastructure/llm/scoring/errors.py` with a docstring stating it wraps
  GGUF-specific failures and flows through the existing
  `CandidateScoringError` boundary. Verify `ruff check` and `mypy` pass on
  the module.
- [x] 2.2 Create `LlamaCppBatchedCausalLMAdapter` in a new module
  `infrastructure/llm/scoring/llama_cpp_adapter.py`. Lazy-import `llama_cpp`
  inside `__init__`. Load the model once from `model_path` via
  `Llama(model_path=..., n_ctx=..., n_batch=..., n_threads=...,
  n_gpu_layers=..., seed=..., verbose=..., logits_all=True)`. Store config
  and a `threading.Lock` for concurrency. Raise `GGUFProviderError` with an
  actionable message if `llama_cpp` import fails or the model file is
  missing/unreadable. Verify the module imports cleanly without `llama_cpp`
  installed (lazy import) and `mypy` passes.
- [x] 2.3 Implement `forward_batch(input_ids, attention_mask) ->
  BatchedLogits` on the GGUF adapter: for each batch item, strip padding via
  the attention mask, acquire the lock, `model.reset()`, `model.eval(tokens)`,
  read `model.eval_logits` once per evaluation and index into it for the
  required positions (do NOT access `eval_logits` inside a loop over
  positions), validate vocab dimension against `model.n_vocab`, reject
  NaN/inf, build `SequenceLogits` (plain Python floats, matching the existing
  contract) per item, return `BatchedLogits` in input order. Translate
  `llama_cpp` exceptions into `GGUFProviderError`. Verify `mypy` and `ruff`
  pass on the module.
- [x] 2.4 Add a `describe() -> dict` method returning backend id (`"gguf"`),
  `model_path`, file content checksum (not the path alone),
  quantization/metadata where available, `n_vocab`, and inference settings.
  Verify `mypy` passes and the method returns a JSON-serializable dict (unit
  test with a mock model).
- [x] 2.5 Write unit tests in `tests/unit/infrastructure/llm/scoring/` for the
  GGUF model adapter using a mock `llama_cpp.Llama` double: verify
  `forward_batch` returns `BatchedLogits` in input order with the exact
  existing contract (return type, `[B, S, V]` shape, plain-float elements,
  per-item ordering), padding tokens are stripped, vocab dimension mismatch
  raises `GGUFProviderError`, NaN/inf logits raise `GGUFProviderError`,
  missing `llama_cpp` raises `GGUFProviderError` with an install hint, and
  `eval_logits` is read once (not per position). Verify
  `pytest tests/unit/infrastructure/llm/scoring/ -x` passes without
  `llama_cpp` installed.
- [x] 2.6 Add a structural-equivalence test asserting `LogitScorer` receives
  identical structural semantics (shape, element type, position alignment)
  from the GGUF adapter and the existing Hugging Face adapter, using a mock
  that returns the same logits. Add a deterministic causal-alignment test
  with known token IDs verifying that the first candidate token is scored
  from the final prefix position and each subsequent token from its
  preceding position. Verify
  `pytest tests/unit/infrastructure/llm/scoring/ -x` passes.

## 3. GGUF tokenizer adapter

- [x] 3.1 Create `LlamaCppBatchTokenizerAdapter` in
  `infrastructure/llm/scoring/llama_cpp_adapter.py` (same module or a
  sibling). Lazy-import `llama_cpp`. Wrap a `Llama` instance's tokenizer.
  Implement `encode_batch(prefixes, candidates) -> BatchTokenization` using a
  full-encode prefix-prefix compatibility check: for each pair, encode the
  full `prefix + candidate` text with explicitly configured BOS/special-token
  behavior, encode the prefix separately with the same policy, check whether
  the prefix token sequence is an exact prefix of the full sequence. If it is,
  set `prefix_token_count = len(prefix_ids)` and
  `candidate_token_ids = full[len(prefix_ids):]`. If it is not, apply an
  explicit fallback or reject with a `GGUFProviderError` compatibility error
  (do not silently score a different sequence). Handle an empty candidate
  without adding BOS twice. Apply right padding to the max length with a pad
  token id. Verify `mypy` and `ruff` pass.
- [x] 3.2 Write unit tests for the GGUF tokenizer adapter using a mock
  tokenizer double: verify the full-encode prefix-prefix check correctly
  determines `prefix_token_count` and `candidate_token_ids` when the prefix
  is an exact prefix; verify a cross-boundary merge (prefix not an exact
  prefix) triggers the explicit fallback or a `GGUFProviderError` (not silent
  scoring of a different sequence); verify an empty candidate is handled
  without double BOS; verify right padding and attention masks are correct;
  verify unequal-length prefixes/candidates raise. Verify
  `pytest tests/unit/infrastructure/llm/scoring/ -x` passes without
  `llama_cpp` installed.

## 4. Backend configuration

- [x] 4.1 Create `ScorerBackendConfig` and `GgufParams` dataclasses in
  `infrastructure/config/` (or `infrastructure/llm/scoring/config.py`).
  `ScorerBackendConfig` has `backend: str` and `model_path: str` plus
  `gguf: GgufParams | None`. `GgufParams` has `n_ctx`, `n_batch`,
  `n_threads`, `n_gpu_layers`, `seed`, `verbose`, `logits_all`. Add a
  validation function that rejects invalid backend names, missing
  `model_path`, and invalid GGUF parameter values at construction time.
  Verify `mypy` and `ruff` pass.
- [x] 4.2 Extend the config loader (`infrastructure/config/loader.py`) to parse
  a `[model]` section with `backend` and `model_path`, and a `[model.gguf]`
  sub-table, into `ScorerBackendConfig`. Reuse the existing `ConfigError`
  pattern. Default to `backend = "huggingface"` when the section is absent.
  Verify `mypy` and `ruff` pass.
- [x] 4.3 Write unit tests for config validation: valid HF config, valid GGUF
  config, invalid backend name raises `ConfigError`, missing `model_path`
  raises, invalid GGUF params raise, absent `[model]` defaults to HF. Verify
  `pytest tests/unit/infrastructure/config/ -x` passes.

## 5. Factory backend selection and call-site wiring

- [x] 5.1 Extend `build_candidate_scorer` in
  `infrastructure/llm/scoring/factory.py` to accept
  `config: ScorerBackendConfig` and dispatch on `config.backend`:
  `"huggingface"` constructs `HuggingFaceBatchTokenizerAdapter` +
  `TorchBatchedCausalLMAdapter` (current path, unchanged); `"gguf"`
  lazy-imports the GGUF adapters and constructs them with `config.gguf`
  params. Both return an `LLMLogitCandidateScorer` with the shared
  `ScoringPromptBuilder` and `LogitScorer`. Verify `mypy` and `ruff` pass.
- [x] 5.2 Update the three call sites (`interfaces/cli/main.py`,
  `application/use_cases/run_baseline/runner.py`,
  `application/use_cases/compare_versions/comparator.py`) to build a
  `ScorerBackendConfig` from the loaded config and call
  `build_candidate_scorer(config)`. `build_candidate_scorer` is internal (not
  exported in any `__all__` or `__init__.py`), so this is an internal-only
  signature change with no public API break. Verify `mypy` passes on all
  three modules and `lint-imports` reports no new contract violations.
- [x] 5.3 Write unit tests for the factory: HF backend returns a scorer wired
  with HF adapters, GGUF backend returns a scorer wired with GGUF adapters
  (mock the lazy import), invalid backend raises at construction, missing
  `llama_cpp` with GGUF backend raises `GGUFProviderError` with an install
  hint. Verify `pytest tests/unit/infrastructure/llm/scoring/ -x` passes
  without `llama_cpp` installed (mock the GGUF path).

## 6. Experiment metadata and resume identity

- [x] 6.1 Integrate `describe()` output from the active model adapter into the
  experiment/run metadata recording path (where dataset fingerprint and
  prompt version are already recorded). Include backend id, model content
  checksum (not path alone), runtime/dependency versions, tokenizer and
  prompt-template config, and inference settings. Distinguish result-changing
  settings (model contents, tokenizer, prompt template, scoring method,
  dataset) from non-semantic settings (logging verbosity). Verify the recorded
  metadata is JSON-serializable and contains all required fields by reading a
  recorded run in a unit test.
- [x] 6.2 Implement resume identity verification: on resume, compare the
  recorded content checksum and semantic settings against the current
  provider's `describe()`. If a result-changing setting differs, either
  reject the resume or explicitly record the run as a new experiment — do not
  silently continue with a different configuration. Verify with unit tests
  that a matching resume succeeds, a mismatched-model-contents resume rejects
  or records a new experiment, and a non-semantic-only change does not block
  resume. Verify `pytest` passes for these tests.

## 7. Integration tests with a GGUF fixture

- [x] 7.1 Configure a small, explicitly licensed GGUF test fixture (or a
  separately configured local test model path via an env var). Mark these
  tests opt-in (e.g., `pytest.mark.gguf` / `--run-gguf`) so the default
  suite does not require the fixture. Verify the default `pytest` run skips
  them and `pytest --run-gguf` collects them.
- [x] 7.2 Add integration tests verifying: the GGUF model loads successfully
  (one load per provider instance); repeated evaluation of the same input
  produces consistent outputs; each example is evaluated independently (no
  state contamination); multiple examples in a batch return correct output
  ordering; class scores and predicted labels have the expected shape and
  semantics; baseline and candidate prompts run through the existing
  evaluator; the prefix/candidate boundary for the project's actual prompt
  format is an exact prefix (or triggers the explicit fallback/reject);
  single- and multi-token labels produce correct causal log-probability
  alignment (first candidate token scored from the final prefix position).
  Verify `pytest --run-gguf tests/integration/ -x` passes.
- [x] 7.3 Add integration tests verifying the optimization cycle accepts and
  rejects candidates according to its existing policy when using the GGUF
  backend, and that resume validates model and experiment identity. Verify
  `pytest --run-gguf tests/integration/ -x` passes.

## 8. Regression tests: Hugging Face vs GGUF

- [x] 8.1 Add a regression test that runs a fixed evaluation dataset through
  both the Hugging Face and GGUF backends using corresponding models.
  Record prediction agreement, macro F1, per-class F1, accuracy, latency,
  and peak memory. Assert no hard logit equality between backends; flag
  significant prediction disagreements for investigation. Verify
  `pytest --run-gguf tests/regression/ -x` passes and produces a recorded
  comparison artifact.

## 9. Security and reliability tests

- [x] 9.1 Add tests for: malformed model paths (raises actionable error),
  invalid configuration values (raises at init), oversized prompts exceeding
  `n_ctx` (raises context-overflow error; no partial successful batch),
  concurrent inference calls on one provider instance (correct results,
  serialized context access, no state corruption), interrupted evaluation and
  resume (subsequent evaluations correct), missing or corrupted checkpoint
  metadata (resume rejects or records), resume with a changed model checksum
  (rejected or explicitly treated as a new experiment), long-context
  evaluation (measured memory within the documented budget), and logging
  behavior with sensitive inputs (errors do not log input text or secrets by
  default). Verify `pytest tests/ -x` (unit portions) and `pytest --run-gguf`
  (integration portions) pass.

## 10. Performance benchmarks and documentation

- [x] 10.1 Add a benchmark script or pytest-benchmark configuration measuring
  initialization time, per-example inference latency, peak memory usage,
  and throughput under supported concurrency for the GGUF backend, with CPU
  and supported GPU offloading. Measure the `logits_all=True` memory cost
  for realistic context lengths and vocabulary sizes and confirm it stays
  within the documented budget. Do not claim partial-position logit reads
  are supported unless a working tested implementation proves it. Verify the
  benchmark runs and produces a recorded results artifact. Do not set a hard
  performance target.
- [x] 10.2 Document in `docs/` (or the project's established docs location):
  installation of the optional GGUF dependency, supported model and hardware
  configurations, GGUF model acquisition and local path setup, backend
  selection, context and memory considerations, label tokenization and
  scoring behavior, differences between Hugging Face and GGUF scores, and
  troubleshooting/benchmark procedures. Clearly distinguish raw logits,
  log-probabilities, and normalized class scores. Verify the documented
  install and backend-selection commands run as written.
- [x] 10.3 Run `mypy`, `ruff check`, and `lint-imports` across the full
  package. Verify no new type, lint, or import-contract violations. Run the
  default `pytest` suite and verify it passes without `llama_cpp` or the
  GGUF fixture installed.
