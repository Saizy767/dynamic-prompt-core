# Proposal

## Why

The candidate scorer currently loads models exclusively through Hugging Face
`transformers` (`AutoModelForCausalLM.from_pretrained`). Users who keep
quantized GGUF models locally cannot classify through `dynamic-prompt-core`
without first converting or re-loading the model in Transformers, which
doubles disk usage and prevents using `llama.cpp`-quantized weights directly.
A `llama-cpp-python`-backed logit provider lets the existing scoring
pipeline consume GGUF models natively while preserving the current
Hugging Face path and all classification semantics.

## What Changes

- Add a GGUF model adapter (`LlamaCppBatchedCausalLMAdapter`) implementing
  the existing `BatchedCausalLanguageModel` seam using `llama-cpp-python`,
  loading a local GGUF file once per provider instance.
- Add a GGUF tokenizer adapter (`LlamaCppBatchTokenizerAdapter`)
  implementing the existing `BatchTokenizerAdapter` seam, using the
  `llama-cpp-python` tokenizer as authoritative for the GGUF model.
- Add backend selection to the scorer factory (`build_candidate_scorer`)
  so a configured backend (`"huggingface"` or `"gguf"`) selects the
  corresponding adapters. Selecting `"huggingface"` preserves current
  behavior exactly.
- Pin `llama-cpp-python` as an **optional** dependency group
  (`[project.optional-dependencies].gguf`). Importing or installing the
  core package MUST NOT require the GGUF runtime. The GGUF adapter MUST
  fail with a clear, actionable error if the optional dependency is
  missing.
- Extend the configuration mechanism (the existing TOML config / loader)
  to carry backend selection and GGUF parameters (`model_path`, `n_ctx`,
  `n_batch`, `n_threads`, `n_gpu_layers`, `seed`, `verbose`, `logits_all`)
  rather than introducing a separate config file.
- Define context isolation so repeated and concurrent evaluations do not
  contaminate inference state, and define resource release at shutdown.
- Record backend identity, model metadata, runtime versions, and
  inference settings in experiment metadata so resume operations can
  verify model/dataset/scoring identity.

## Capabilities

### New Capabilities

- `gguf-logit-provider`: GGUF-backed logit provider — model initialization
  from a local GGUF file, configuration validation, context isolation for
  repeated and concurrent evaluation, resource management, optional
  dependency handling, and experiment metadata reporting. The provider
  implements the existing `BatchedCausalLanguageModel` and
  `BatchTokenizerAdapter` seams defined by `llm-logit-scorer`.

### Modified Capabilities

- `llm-logit-scorer`: Adds backend selection through configuration. The
  scorer factory SHALL construct either the Hugging Face or GGUF adapter
  set from a configured backend identifier. The GGUF adapters SHALL
  satisfy the existing seam contracts (`BatchedCausalLanguageModel`,
  `BatchTokenizerAdapter`) so the shared `LLMLogitCandidateScorer`,
  `LogitScorer`, and prompt-construction logic remain unchanged. Invalid
  backend names, missing model files, and missing optional dependencies
  SHALL fail during initialization.

## Impact

- **Code**: `infrastructure/llm/scoring/` gains GGUF model and tokenizer
  adapters; `factory.py` gains backend selection; `infrastructure/config/`
  gains GGUF backend parameters and validation.
- **Dependencies**: `llama-cpp-python` added as an optional dependency
  group in `pyproject.toml` and `requirements.txt` (optional section).
  Core install footprint unchanged.
- **APIs**: No change to the `CandidateScorer` port or any
  domain/application contract. Backend selection is infrastructure-internal.
- **Tests**: Unit tests with mocked inference; opt-in integration tests
  with a small GGUF fixture; regression tests comparing HF and GGUF
  backends on a fixed dataset.
- **Docs**: Installation of the optional GGUF dependency, backend
  selection, model acquisition, and score-difference guidance.
