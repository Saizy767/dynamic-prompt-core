# Design

## Context

Thesis embeddings are computed by `compute_embedding` in
`thesis_analyzer.py:142`, which calls `SentenceTransformer.encode` one thesis at
a time on CPU. The model is lazy-loaded by `_get_embedding_model` (`:123`), a
module-level function with global cache (`_embedding_model`,
`_embedding_model_name`). Three callers share this cache: thesis analysis
(`add_or_update` at `:244`), the composer distortion check (`composer.py:318`),
and the CLI adapter (`cli/main.py:140`). The platform is macOS with Apple
Silicon, so MPS (Metal Performance Shaders) is the available GPU backend. The
clustering algorithm (`assign_cluster` at `:266`) is an online incremental
greedy method with ≤20 clusters — distance computations are negligible relative
to embedding cost.

## Goals / Non-Goals

**Goals:**
- Move embedding inference from CPU to GPU (MPS) when available
- Batch-encode all unique theses in one call instead of per-thesis in a loop
- Make the device configurable and auto-detected with graceful CPU fallback
- Apply the GPU device to all three consumers of `_get_embedding_model`
- Preserve identical embedding vectors and clustering results

**Non-Goals:**
- Moving distance computations (cosine/Manhattan) to GPU — negligible with ≤20
  clusters
- Changing the online clustering algorithm or its thresholds
- Changing the thesis-bank dump/load format
- Adding GPU support to the LLM inference backends (llama-server, GGUF, HF) —
  those are separate concerns
- Batch-encoding in the composer distortion check — rules are formulated
  sequentially with LLM calls between them, so batching does not apply

## Decisions

### D1: Device resolution — auto-detect with explicit override

`_get_embedding_model` gains a `device` parameter. Resolution logic:

```
"auto" → torch.backends.mps.is_available() → "mps"
       → torch.cuda.is_available()         → "cuda"
       → else                               → "cpu"
"mps"  → check available, else WARNING + "cpu"
"cuda" → check available, else WARNING + "cpu"
"cpu"  → always "cpu"
```

Rationale: `"auto"` as default means existing deployments get GPU acceleration
with no config change. Explicit `"cpu"` preserves the current behavior for
reproducibility or CI. The fallback ensures the code never crashes on a machine
without the requested GPU.

Alternative considered: hardcode `"mps"` — rejected because the project may run
on Linux (CUDA) or CI (CPU-only).

### D2: Batch-encoding pre-pass in `analyze()`

Currently `analyze()` (`:492`) iterates OK-records and calls
`add_or_update` per thesis, which calls `compute_embedding` for each new thesis.
Restructured flow:

1. **Collect**: iterate all OK-records, gather unique `theses_norm` into a set
2. **Batch-encode**: call `model.encode(sorted(unique_norms), batch_size=N,
   convert_to_numpy=True)` once — returns a matrix `[U, 384]`
3. **Build lookup**: `{norm: embedding}` dict from the results
4. **Cluster**: iterate OK-records again, `add_or_update` receives the
   pre-computed embedding from the lookup instead of calling
   `compute_embedding`

`ThesisBank.add_or_update` gains an optional `precomputed_embedding` parameter.
When provided, it skips `compute_embedding`. When `None` (e.g., composer
distortion check, cross-task transfer), it falls back to the existing
per-thesis `compute_embedding` path.

Rationale: two passes over the data is cheap (I/O-bound, records already in
memory). The single batch encode call eliminates Python-loop overhead and lets
the GPU parallelize the forward passes.

Alternative considered: batch-encode inside `ThesisBank.__init__` — rejected
because the bank doesn't know all theses upfront; it's built incrementally.

### D3: Device and batch size in config

Add to `ThesisAnalyzerConfig`:
- `embedding_device: str = "auto"`
- `embedding_batch_size: int = 32`

Parsed from `[thesis_analyzer]` in `config.toml`. `PromptComposerConfig` gains
`embedding_device` only (no batch size — distortion check is per-rule).

The device is passed through `_get_embedding_model(model_name, device=...)`.
Since the model is globally cached by `(model_name, device)`, the cache key
expands to include device so a config change triggers a reload.

### D4: Global cache key includes device

Currently `_embedding_model_name` is the cache key. Change to
`(_embedding_model_name, _embedding_device)` so switching device reloads the
model on the new device. This matters for the composer, which may use a
different device than thesis analysis in principle (though in practice they
share the same config section's model).

## Risks / Trade-offs

- **MPS numerical differences**: MPS may produce slightly different float
  results than CPU due to Metal kernel implementations. Embeddings should be
  numerically close but may not be bit-identical. → Mitigation: document that
  clustering results may differ at the margin; the cosine/Manhattan thresholds
  have grey-zone tolerance built in. Add a test asserting embeddings are close
  within `np.allclose(atol=1e-4)`.

- **Memory pressure on GPU**: batch-encoding a large set of theses on MPS could
  exceed available GPU memory. → Mitigation: `batch_size` is configurable
  (default 32); `sentence-transformers` handles internal batching and falls
  back gracefully. For very large thesis sets, the user can lower
  `embedding_batch_size`.

- **Two-pass memory**: collecting all unique theses before encoding holds them
  in memory alongside the records. → Mitigation: theses are short strings
  (normalized phrases); even 10K unique theses is trivial memory.

- **First-call latency**: loading the model on MPS has a one-time cost (Metal
  shader compilation). → Mitigation: this is amortized over all theses in the
  batch; subsequent calls reuse the cached model.
