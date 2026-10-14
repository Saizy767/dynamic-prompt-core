# Tasks

## 1. Config: embedding device and batch size

- [x] 1.1 Add `embedding_device: str = "auto"` and `embedding_batch_size: int = 32` fields to `ThesisAnalyzerConfig` in `thesis_analyzer.py`, parse them from the `[thesis_analyzer]` section in `ThesisAnalyzerConfig.from_config`, and validate that `embedding_device` is one of `"auto"`, `"mps"`, `"cuda"`, `"cpu"` (raise `ThesisAnalyzerError` on invalid value). Verify by loading `config.toml` and checking the parsed values.
- [x] 1.2 Add `embedding_device: str = "auto"` field to `PromptComposerConfig` in `composer.py`, parse from `[prompt_composer]` in its `from_config` method with the same validation. Verify by loading `config.toml` and checking the parsed value.
- [x] 1.3 Add `embedding_device = "auto"` and `embedding_batch_size = 32` keys to the `[thesis_analyzer]` section and `embedding_device = "auto"` to the `[prompt_composer]` section in `config.toml`. Verify the file parses and the values are read back correctly.

## 2. Device resolution in `_get_embedding_model`

- [x] 2.1 Add a `resolve_embedding_device(device: str) -> str` function in `thesis_analyzer.py` that maps `"auto"` → first available of MPS/CUDA/CPU, validates explicit `"mps"`/`"cuda"` with fallback to `"cpu"` + WARNING log, and passes `"cpu"` through. Use `torch.backends.mps.is_available()` and `torch.cuda.is_available()` for detection. Verify with a unit test that mocks availability and checks each path.
- [x] 2.2 Modify `_get_embedding_model` to accept a `device: str = "auto"` parameter, call `resolve_embedding_device`, and pass the resolved device to `SentenceTransformer(model_name, device=resolved)`. Expand the global cache key from `_embedding_model_name` to `(_embedding_model_name, resolved_device)` so a device change triggers a model reload. Log the resolved device at INFO level. Verify with a unit test that the model is loaded with the expected device and that changing device reloads.

## 3. Batch-encoding pre-pass in `analyze()`

- [x] 3.1 Add a `precompute_embeddings(norms: set[str], model_name: str, device: str, batch_size: int) -> dict[str, np.ndarray]` function in `thesis_analyzer.py` that batch-encodes all unique normalized theses in a single `model.encode(sorted(norms), batch_size=batch_size, convert_to_numpy=True)` call and returns a `{norm: embedding}` dict. Verify with a unit test using a mock model that checks all norms are encoded in one call with the correct batch size.
- [x] 3.2 Modify `ThesisBank.add_or_update` to accept an optional `precomputed_embedding: np.ndarray | None = None` parameter. When provided, skip `compute_embedding` and use the passed embedding directly. When `None`, fall back to the existing `compute_embedding` call. Verify with a unit test that a pre-computed embedding is used and `compute_embedding` is not called.
- [x] 3.4 Modify `analyze()` to perform a two-pass flow: (1) collect unique `theses_norm` from all OK-records, (2) call `precompute_embeddings`, (3) iterate records again calling `add_or_update` with the pre-computed embedding from the lookup. Pass `embedding_device` and `embedding_batch_size` from the config. Verify with a unit test on a small fixture that embeddings are identical to the per-thesis path and clustering results are unchanged.

## 4. Composer distortion check device passthrough

- [x] 4.1 Modify `compute_rule_embedding` in `composer.py` to accept a `device: str = "auto"` parameter and pass it through to `compute_embedding`. Verify with a unit test that the device reaches `_get_embedding_model`.
- [x] 4.2 Modify `check_distortion` in `composer.py` to accept and forward the `device` parameter to `compute_rule_embedding`. Update all call sites of `check_distortion` in `composer.py` to pass `config.embedding_device`. Verify by tracing the device from `PromptComposerConfig` through to `_get_embedding_model`.

## 5. CLI adapter device passthrough

- [x] 5.1 Update the CLI embedding adapter in `interfaces/cli/main.py` to pass `device="auto"` (or a configurable value) to `compute_embedding`. Verify the CLI still produces embeddings correctly.

## 6. Tests

- [x] 6.1 Add a unit test for `resolve_embedding_device` covering all four input values and the fallback path (mock `torch.backends.mps.is_available` and `torch.cuda.is_available`). Verify each branch returns the expected device and that unavailable GPU triggers a WARNING + CPU fallback.
- [x] 6.2 Add a unit test for `precompute_embeddings` using a mock `SentenceTransformer` that records the `encode` call arguments. Verify a single batch call is made with the correct texts, batch size, and `convert_to_numpy=True`.
- [x] 6.3 Add a unit test for the two-pass `analyze()` flow on a small synthetic fixture (3 records, 5 unique theses). Verify the resulting `ThesisBank` has the same theses, embeddings (within `np.allclose(atol=1e-4)`), and cluster assignments as the original per-thesis path.
- [x] 6.4 Add a unit test for `ThesisBank.add_or_update` with a pre-computed embedding, verifying `compute_embedding` is not called and the embedding is stored correctly.
- [x] 6.5 Add a unit test for invalid `embedding_device` values raising a configuration error in both `ThesisAnalyzerConfig.from_config` and `PromptComposerConfig.from_config`.

## 7. Integration verification

- [x] 7.1 Run `python -m pytest tests/ -x -q` and verify all existing tests pass with no regressions from the config or signature changes.
- [x] 7.2 Run `ruff check src/` and `mypy src/` and verify no new lint or type errors.
- [x] 7.3 Run `python -m dynamic_prompt_core.interfaces.cli.main` (or the thesis analyzer CLI) on a small results artifact with `embedding_device = "auto"` and verify the resolved device is logged and the thesis bank dump is produced with correct embeddings.
