# Proposal

## Why

Thesis clustering (`stage1-thesis-analyzer`) computes sentence embeddings via
`SentenceTransformer` on CPU, one thesis at a time, inside a Python loop. On
macOS with Apple Silicon the MPS (Metal Performance Shaders) GPU is available
but unused because no `device` is passed to the model. The embedding forward
pass is the dominant cost of thesis analysis — distance computations against
≤20 cluster centroids are negligible by comparison. The same CPU-bound
embedding path is reused by the prompt composer's semantic distortion check.
Batch-encoding all unique theses in a single GPU call would eliminate both
the per-thesis Python-loop overhead and the CPU inference bottleneck.

## What Changes

- Add a configurable `embedding_device` field to `ThesisAnalyzerConfig` and
  `PromptComposerConfig` (values: `"auto"`, `"mps"`, `"cuda"`, `"cpu"`).
  `"auto"` detects MPS → CUDA → CPU in that order. Default: `"auto"`.
- Pass the resolved device to `SentenceTransformer(model_name, device=...)`
  in `_get_embedding_model`, which is the shared lazy-load entry point for
  thesis analysis, the composer distortion check, and the CLI embedding
  adapter.
- Restructure `analyze()` to **batch-encode** all unique normalized theses
  in a single `model.encode(list, batch_size=N, convert_to_numpy=True)` call
  before the clustering loop, then feed pre-computed embeddings to
  `ThesisBank.add_or_update` so no per-thesis `encode` happens inside the
  loop.
- Log the resolved device and embedding batch size at startup so the
  runtime configuration is observable.
- Fall back gracefully to CPU with a WARNING when the requested device is
  unavailable, preserving identical results.

## Capabilities

### New Capabilities

<!-- None. -->

### Modified Capabilities

- `stage1-thesis-analyzer`: The embedding computation requirement changes
  to support configurable GPU device selection and batch encoding of all
  unique theses in a single call, instead of per-thesis CPU encoding inside
  the clustering loop.

## Impact

- **Code**: `application/use_cases/analyze_theses/thesis_analyzer.py`
  (device config, batch-encode pre-pass, `add_or_update` accepts
  pre-computed embedding), `application/use_cases/compose_prompt/composer.py`
  (device config for distortion check), `application/services/clustering/__init__.py`
  (device passthrough), `interfaces/cli/main.py` (device passthrough).
- **Config**: `config.toml` `[thesis_analyzer]` and `[prompt_composer]`
  sections gain `embedding_device` and `embedding_batch_size` keys.
- **APIs**: No change to any public port or domain contract. The
  `ThesisBank` dump/load format is unchanged — embeddings are the same
  vectors, computed faster.
- **Dependencies**: No new required dependencies. `torch` is already
  available transitively via `sentence-transformers` and `transformers`.
- **Tests**: Unit tests for device resolution logic (auto-detect, fallback),
  batch-encode pre-pass, and numerical equivalence of batch vs. per-thesis
  embeddings.
