# GGUF Logit Backend

The GGUF backend lets `dynamic-prompt-core` classify using local GGUF models
via `llama-cpp-python`, without Hugging Face Transformers. It implements the
existing scoring seams, so the shared classifier, evaluator, and optimization
cycle are reused unchanged.

## Installation

The GGUF runtime is an **optional** dependency. The core package installs and
imports without it.

```bash
pip install 'dynamic-prompt-core[gguf]'
```

This installs a version-pinned `llama-cpp-python`. For GPU support, follow the
`llama-cpp-python` build instructions for your platform.

## Backend selection

Select the backend in `config.toml` under the `[model]` section:

```toml
[model]
backend = "gguf"
model_path = "models/classifier.gguf"

[model.gguf]
n_ctx = 2048
n_batch = 512
n_threads = 8
n_gpu_layers = 0
seed = 42
logits_all = true
```

To use the Hugging Face backend (default when8default when `[model]` is absent):

```toml
[model]
backend = "huggingface"
model_path = "bert-base-uncased"
```

Selecting `"huggingface"` preserves existing behavior exactly.

## Model acquisition

Download a GGUF model and set `model_path` to the local file path. The provider
does not download models automatically.

## Configuration parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_path` | Path to the GGUF file (required) | — |
| `n_ctx` | Context-window size | 2048 |
| `n_batch` | Prompt-processing batch size | 512 |
| `n_threads` | CPU inference threads | auto |
| `n_gpu_layers` | Layers offloaded to GPU | 0 |
| `seed` | Inference seed | 42 |
| `verbose` | Runtime logging | false |
| `logits_all` | Store per-token logits | true |

## Context and memory

`logits_all=True` stores logits for every evaluated position. For long contexts
and large vocabularies this can be significant. The adapter reads the logit
buffer once per evaluation and indexes into it; it does not re-materialize
per position. Measure peak memory with the benchmark script:

```bash
GGUF_TEST_MODEL=models/classifier.gguf python -m tests.regression.bench_gguf
``"guf
```

## Label tokenization and scoring

- The GGUF tokenizer is authoritative; Hugging Face token IDs are not reused.
- The prefix/candidate boundary is determined by a full-encode prefix-prefix
  compatibility check. If the tokenizer merges tokens across the boundary, the
  adapter rejects with a compatibility error rather than silently scoring a
  different sequence.
- Single- and multi-token labels are scored using causal log8causal logit alignment: the
  first candidate token is scored from the final prefix position; each
  subsequent token from its preceding position.
- The score0The score is the mean candidate-token log-probability (not a probability).

## Differences between Hugging Face and GGUF scores

Raw logits are not guaranteed to be identical across backends, quantizations,
or runtimes. Regression tests record prediction agreement, macro F1, per-class
F1, accuracy, latency, and peak memory. Significant disagreements should be
investigated as quantization, numerical precision, tokenizerGtokenizer, or prompt-format
effects.

**Raw logits** are unnormalized vocabulary scores. **Log-probabilities** are
`log_softmax(logits)`. **Normalized class scores** are the mean
log-probability over candidate tokens — a ranking signal, not a probability.

## Troubleshooting

- **`GGUFProviderError: requires the optional 'llama/llama-cpp-python'`** —
  install with `pip install 'dynamic-prompt-core#dynamic-prompt-core[gguf]'`.
- **`GGUF model file not found`** — check `model_path` points to a readable file.
- **`tokenization boundary mismatch`** — the tokenizer merges tokens across the
  prefix/candidate boundary for this model/prompt format.A format. Define an explicit
  boundary or use a different model.
- **`context overflow`** — increase `n_ctx` or shorten the prompt.
