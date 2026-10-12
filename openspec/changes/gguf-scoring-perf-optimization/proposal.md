# Proposal

## Why

The GGUF-backed candidate scorer (`LlamaCppBatchedCausalLMAdapter` +
`LogitScorer` + `LLMLogitCandidateScorer`) spends ~99.8% of a scoring
request in `forward_batch`, but the current logs cannot attribute that
time to its real causes: per-item context reset, `eval()`, logit
materialization, Python float conversion, and NaN/inf validation all run
inside one un-instrumented critical section. The adapter also
materializes the full `[T x V]` logit matrix (vocab ~151 936) into
Python lists, builds a complete `log_softmax` vector per candidate token
when only one log-probability is needed, and re-evaluates the shared
prompt prefix once per candidate. Without measurements we cannot promise
a speedup; without an optimized path we cannot scale beyond the current
per-request latency on the 6 091-example dataset.

## What Changes

- Add per-stage **instrumentation** to the GGUF adapter and scorer that
  records monotonic timings (`lock_wait`, `reset`, `eval`, `logits_read`,
  `logits_convert`, `logits_validate`, `scoring`) plus token/vocab/candidate
  counters, without logging user text or full logit arrays. One benchmark
  run MUST localize the dominant bottleneck.
- Replace the full-matrix logit materialization with **selective logit
  extraction**: read logits only at the positions required for candidate
  scoring, so the volume crossing the C++→Python boundary is proportional
  to the number of scored positions, not the full sequence length. Provide
  a fallback when the installed `llama-cpp-python` API cannot restrict
  saved positions.
- Replace the per-position full `log_softmax` Python list with a direct
  **stable log-sum-exp** computation of the single target-token
  log-probability (`z_i - LSE(z)`), preserving the existing NaN/Inf
  validation policy and the mean-log-probability score.
- Add **shared-prefix KV-cache reuse** for candidates that share the
  prompt prefix within one `score()` invocation: prefill the common
  prefix once, branch per candidate, and isolate per-candidate state.
  Fall back to the current per-candidate reset+eval when reuse is unsafe
  or unsupported.
- Make **CPU/GPU runtime configuration** explicit and measured: log
  `n_threads`, sweep thread counts, validate `n_batch`/`n_ctx`
  compatibility, and test GPU offload when a compatible backend is
  available, without changing `n_ctx`/`n_batch` without a benchmark.
- Define **concurrency semantics** for the shared model: measure lock
  contention, keep the lock until thread-safety is proven, and bound
  queued work (queue or dedicated worker) so parallel runs do not
  contaminate KV-cache state or accumulate unbounded load.
- Add a standalone **benchmark** (independent of the full 6 091-example
  run) covering short/long prompt, candidate-count, shared-prefix,
  sequential latency, and parallel throughput scenarios, emitting
  machine-readable JSON/CSV with mean/median/p95, tokens/s, examples/s,
  peak RAM, and score deviation from baseline.
- Add **regression tests** asserting numerical equivalence to the
  baseline scorer within a fixed tolerance, plus unit tests for the new
  log-sum-exp, selective-logit, and prefix-reuse paths.

## Capabilities

### New Capabilities

- `gguf-scoring-performance`: Performance instrumentation, optimized
  logit extraction/scoring, shared-prefix KV-cache reuse, runtime
  configuration, concurrency semantics, and benchmark tooling for the
  GGUF-backed candidate scoring path — all while preserving the existing
  `CandidateScorer.score()` contract and the mean-log-probability score.

### Modified Capabilities

<!-- None. The scoring algorithm, causal alignment, error translation,
     port contract, and information boundary defined by llm-logit-scorer
     are preserved exactly. Optimizations are infrastructure-internal and
     additive. -->

## Impact

- **Code**: `infrastructure/llm/scoring/llama_cpp_adapter.py` (eval,
  logit extraction, batching, prefix reuse, instrumentation, config),
  `infrastructure/llm/scoring/logit_scorer.py` (log-sum-exp),
  `infrastructure/llm/scoring/candidate_scorer.py` (stage timing,
  adapter orchestration); `infrastructure/llm/scoring/config.py` may gain
  measured runtime parameters.
- **APIs**: No change to the `CandidateScorer` port or any
  domain/application contract. New diagnostics and benchmark hooks are
  infrastructure-internal.
- **Dependencies**: No new required dependencies. NumPy may be used
  opportunistically for vectorized log-sum-exp if already available;
  otherwise a pure-Python stable implementation is used.
- **Tests**: Unit tests with mocked inference for new paths; regression
  tests comparing optimized vs. baseline scores on a fixed set; opt-in
  benchmark with a small GGUF fixture emitting JSON/CSV.
- **Docs**: Benchmark usage, instrumentation fields, runtime
  configuration recommendations, and backend limitations/hardware notes.
