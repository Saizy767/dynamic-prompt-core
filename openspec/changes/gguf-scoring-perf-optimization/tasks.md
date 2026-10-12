# Tasks

## 1. Instrumentation and baseline profiling (P0)

- [x] 1.1 Implement a typed timing collector in `infrastructure/llm/scoring/`
  using `time.perf_counter()` to record `lock_wait_ms`, `reset_ms`,
  `eval_ms`, `logits_read_ms`, `logits_convert_ms`, `logits_validate_ms`,
  and `scoring_ms`, together with counters `input_tokens`, `scored_tokens`,
  `vocab_size`, and `candidate_count`. The collector MUST NOT store prompt
  text, candidate text, raw logit arrays, or other sensitive user content.
  All timing values MUST use milliseconds, be non-negative, and follow a
  documented schema. Verify `ruff check` and `mypy` pass on the new module.
- [x] 1.2 Integrate the collector into
  `LlamaCppBatchedCausalLMAdapter.forward_batch()` without changing the
  scoring algorithm. Measure lock acquisition wait, model reset,
  `model.eval(...)`, access to `eval_logits`, logit conversion, and logit
  validation as separate stages. The `eval_ms` interval MUST measure only
  the `model.eval(...)` call. The `logits_read_ms` interval MUST include
  access to `eval_logits` and any lazy materialization triggered by that
  access. Ensure instrumentation does not double-count nested intervals or
  change exception propagation. Verify a unit test using a mock Llama whose
  `eval_logits` simulates lazy materialization emits exactly one complete
  profile record with all seven timing fields and four counters populated.
- [x] 1.3 Integrate adapter sub-stage timings into `LLMLogitCandidateScorer.score()`
  so the existing prompt/tokenization/forward/logit breakdown is extended
  with the adapter's detailed profile. Define the relationship between
  aggregate and nested timings explicitly to prevent double-counting. Verify
  the info-level log record contains the sub-stage breakdown without user
  text or logit values, and `pytest tests/unit/infrastructure/llm/scoring/ -x`
  passes without `llama_cpp` installed.
- [x] 1.4 Add unit tests verifying all timings are non-negative and that the
  collector records correct stage boundaries using a monotonic clock. Test
  the absence of prompt text, candidate text, and raw logit arrays in
  emitted records. Verify that completed stage timings remain available
  when a later stage raises an exception and that instrumentation does not
  alter the underlying scoring error behavior.
- [x] 1.5 Produce a baseline performance profile on the target hardware using
  one short-prompt and one long-prompt scenario, two candidates per
  scenario. Save machine-readable JSON under `logs/` or `tools/`. Record
  per-stage latency, stage time shares, token counts, candidate counts,
  model configuration, backend version, and hardware configuration. The
  profile MUST distinguish model evaluation time from lazy logit
  materialization and Python post-processing. Use repeated measurements for
  steady-state statistics and record cold-start/first-request latency
  separately. Verify the profile file is present and parseable.

## 2. Logit post-processing optimization (P1)

- [x] 2.1 Replace `LogitScorer._log_softmax()` with a direct single-token
  log-probability function computing `z_i - LSE(z)` using numerically
  stable max-shifted log-sum-exp. Preserve the existing NaN/Inf rejection
  policy and raise `LLMScoringError` for empty logit vectors. Preserve the
  scoring contract, including vocabulary normalization and candidate-token
  alignment. Verify `ruff check` and `mypy` pass.
- [x] 2.2 Implement an optional NumPy-vectorized path when NumPy is available,
  with a pure-Python fallback that does not allocate the full normalized
  log-probability vector. NumPy MUST NOT become a mandatory dependency.
  Verify the package remains importable and tests pass without NumPy.
- [x] 2.3 Add unit tests comparing the new function against the existing
  `_log_softmax()` on deterministic random and extreme-magnitude finite
  vectors: numerical equivalence within a documented tolerance, NaN/Inf
  rejection per existing policy, explicit empty-vector handling, and
  absence of a full normalized-vector allocation. Verify
  `pytest tests/unit/infrastructure/llm/scoring/ -x` passes.
- [x] 2.4 Investigate the installed `llama-cpp-python` API and implement a
  capability probe determining whether the backend supports restricting
  saved logit positions to those required for candidate scoring. The probe
  MUST be based on verified API behavior, not merely the presence of an
  arbitrary attribute. Document the token-position convention and verify
  causal alignment (logits at position `i` predict token `i+1`). Define the
  scored position range precisely in terms of the actual token sequence
  and candidate boundaries; preserve any required EOS/special-token
  scoring behavior. The probe MUST return a boolean and fall back to the
  existing full `eval_logits` read + indexing path when selective
  extraction is unavailable or unverified. Verify with a unit test using a
  mock model exposing/omitting the verified capability.
- [x] 2.5 Implement selective logit extraction only when the capability probe
  succeeds. The optimized path MUST reduce logit data volume read into
  Python in proportion to scored positions while preserving full-vocabulary
  logits at each scored position for correct log-sum-exp normalization. It
  MUST NOT claim to eliminate full-vocabulary model computation unless
  backend measurements establish that result. Verify unit tests cover
  selected positions, read-volume scaling, causal alignment, fallback
  behavior, and equivalence of token log-probabilities and candidate
  scores within a documented tolerance.
- [x] 2.6 Add regression tests comparing optimized and baseline scoring on a
  fixed input set. Compare per-token log-probabilities, final candidate
  scores, and ranking consistency, with an initial absolute tolerance of
  `1e-4`. Cover different candidate lengths, shared and non-shared
  prefixes, special tokens, EOS handling where applicable, non-finite
  logits, empty vectors, and selective-extraction fallback. Any tolerance
  that cannot be met MUST be investigated and documented rather than
  silently relaxed. Verify the regression tests pass with a deterministic
  mock model.

## 3. Shared-prefix KV-cache reuse (P1)

- [x] 3.1 Implement token-level shared-prefix detection across candidates
  within one `score()` invocation. Detection MUST compare the actual
  tokenized sequences passed to the model, not textual equality. The
  implementation MUST account for tokenizer boundary behavior and identify
  the exact common token prefix. Verify a unit test confirms identical text
  is not assumed to produce identical token sequences when tokenization
  context differs.
- [x] 3.2 Investigate and verify a safe KV-cache save/restore or equivalent
  context-branching mechanism supported by the installed `llama-cpp-python`.
  Implement a capability probe only after validating the mechanism against
  the actual backend API. Prefix reuse MUST remain disabled if the
  mechanism is unavailable, fails validation, or cannot guarantee
  candidate-state isolation. The probe MUST return a boolean and the
  fallback MUST use per-candidate reset-and-evaluate producing
  baseline-equivalent scores. If no verified mechanism is available,
  retain the baseline implementation and document the limitation instead
  of introducing unsafe reuse. Verify with a unit test using a mock model.
- [x] 3.3 Implement prefix prefill-once with isolated per-candidate
  continuation state. The shared token prefix MUST be evaluated only once
  when reuse is enabled and supported, and each candidate MUST branch from
  an identical, valid prefix state. No candidate's continuation may
  contaminate another's KV cache, logits, or score. Verify a unit test
  uses a mock model counting evaluation calls and tracking candidate
  state to confirm shared-prefix evaluation and isolation.
- [x] 3.4 Add regression tests comparing prefix-reuse results against the
  per-candidate reset-and-evaluate baseline on fixed input sets. Compare
  per-token log-probabilities and final candidate scores within a
  documented tolerance, verify candidate-state isolation and ranking
  consistency, and cover the fallback path when reuse is unavailable.
  Verify `pytest tests/unit/infrastructure/llm/scoring/ -x` passes.

## 4. Runtime configuration (P2)

- [x] 4.1 Resolve and record the effective `n_threads` at adapter
  initialization when reliably determinable. Distinguish the explicitly
  configured value, the backend-selected default, and the actual observed
  CPU thread usage where measurable. Include `n_threads`, `n_gpu_layers`,
  `n_batch`, `n_ctx`, and `logits_all` in `describe()` metadata. Do not
  infer actual thread usage solely from a configured value. Verify a unit
  test asserts all parameters are represented and the source/resolution
  status of the thread setting is documented.
- [x] 4.2 Add a benchmark sweep over candidate thread counts to the
  standalone benchmark harness. The output MUST include the tested
  configuration and corresponding latency, throughput, and memory
  measurements. Document the selected configuration based on target-hardware
  measurements. Do not change `n_ctx`, `n_batch`, or `n_gpu_layers` without
  benchmark evidence, an explicit memory check, and a review checkpoint
  recording previous and proposed values. Verify the sweep results are
  emitted in the benchmark output.
- [x] 4.3 Add an opt-in GPU-offload test path using `n_gpu_layers > 0`. The
  test MUST execute only when a compatible backend and suitable hardware
  are available; otherwise it MUST be skipped rather than failed. Document
  the CPU-only configuration separately as the tested baseline when GPU
  acceleration is unavailable. Verify the test is skipped (not failed)
  when prerequisites are missing.

## 5. Concurrency semantics (P2)

- [x] 5.1 Use the instrumentation from Section 1 to measure `lock_wait_ms`
  under concurrent scoring calls. Add a deterministic unit test or
  test-double scenario that introduces lock contention and verifies
  lock-wait time is recorded separately from model evaluation and
  post-processing time.
- [x] 5.2 Bound queued concurrent scoring work using a bounded queue or
  dedicated worker while retaining the serialization lock around shared
  model state. Define queue capacity, overload behavior, cancellation
  behavior, and worker shutdown semantics. Do not remove the lock unless
  the backend's thread-safety guarantees have been independently verified.
  Verify unit tests confirm bounded queue behavior, score integrity, and
  absence of KV-cache cross-contamination under concurrent invocation.
- [x] 5.3 Add concurrency regression tests asserting concurrent invocation
  produces scores equivalent to sequential invocation within a documented
  tolerance. Benchmark request latency and throughput separately and
  report both. Tests MUST NOT assume concurrency improves throughput or
  latency; any claimed benefit MUST be demonstrated by measurements using
  the same workload and hardware. Verify with a deterministic mock model.

## 6. Benchmarking, regression, and validation

- [x] 6.1 Create a standalone benchmark module under `tools/` or
  `tests/benchmark/` that executes independently of the full 6,091-example
  dataset experiment. Running the benchmark MUST NOT launch the complete
  dataset run. Support a deterministic mock-model mode for CI and an
  opt-in real-model mode for hardware profiling. Verify a mock-model
  invocation runs without the full dataset.
- [x] 6.2 Implement six benchmark scenarios: short prompt + 2 candidates,
  long prompt + 2 candidates, 1 prompt + 8 candidates, candidates sharing
  a common token prefix, sequential scoring latency, and parallel-request
  throughput. Every scenario MUST perform an explicit warm-up phase where
  applicable and record cold-start/first-request latency separately from
  steady-state. Verify a mock-model dry run exercises all six scenarios
  without downloading or loading a large model.
- [x] 6.3 Emit machine-readable JSON or CSV results per scenario. Include
  mean, median, p95 latency, tokens/s, examples/s, peak Python memory, peak
  process RAM where measurable, evaluated token positions, candidate count,
  token counts, and score deviation from baseline. Record model and backend
  versions, runtime parameters, hardware configuration, warm-up policy,
  sample count, and measurement units. Distinguish Python memory
  (`tracemalloc`) from process RSS and accelerator memory. Verify unit
  tests parse the emitted files and assert all required fields are present
  and correctly typed.
- [x] 6.4 Add opt-in real-model regression and benchmark tests isolated from
  the default unit suite, explicitly documenting model, hardware, and
  backend requirements. The default `pytest` run MUST NOT require a large
  model, network access, or a compatible GPU. Real-model tests MUST be
  gated behind an explicit opt-in flag and report a clear skip reason when
  prerequisites are missing. Verify the default suite is independent and
  real-model tests skip cleanly.
- [x] 6.5 Benchmark each optimization independently against the same
  baseline: direct log-probability calculation, selective logit extraction,
  shared-prefix KV-cache reuse, runtime configuration changes, and
  concurrency changes. Each optimization MUST have a separately gated
  implementation and a verified fallback to the baseline path. Benchmark
  results MUST report individual changes in latency, throughput, memory,
  and score deviation, not only cumulative speedup. Combined optimizations
  MAY be evaluated after their individual effects are measured. Verify the
  benchmark output records per-optimization deltas.
- [x] 6.6 Expand regression coverage for special-token handling, EOS handling
  where applicable, NaN and infinite logits, empty logit vectors,
  candidate-length differences, causal alignment, and all optimized-path
  fallback behaviors. Assert equivalence with the baseline scoring
  contract and preserve established error behavior. Verify
  `pytest tests/unit/infrastructure/llm/scoring/ -x` passes.
- [x] 6.7 Produce a performance report comparing baseline and optimized
  measurements on the target hardware. Identify the measured dominant
  bottleneck before and after optimization, document backend limitations
  and capability-probe results (selective logit extraction, KV-cache reuse,
  thread safety), and record hardware and software configuration. State
  measured speedup, or explicitly report the absence of improvement, based
  on reproducible benchmark data. Commit both a machine-readable result
  file and a human-readable report under `docs/` or `logs/`. Do not claim
  an improvement not supported by measurements. Verify both artifacts are
  present.
- [x] 6.8 Run `ruff check`, `mypy`, and `pytest tests/unit/ -x` on the
  complete change. All checks MUST pass, existing scoring tests MUST remain
  green, and the default test suite MUST remain independent of optional
  `llama_cpp`, NumPy, large-model, and GPU dependencies where those are not
  required by the baseline package contract. Verify all pass.

## 7. Acceptance criteria and implementation order

- [x] 7.1 Begin with the standalone benchmark and baseline instrumentation,
  then produce a reproducible baseline profile on the target hardware.
  Establish the scoring contract and a fixed golden test set before
  changing logit processing or cache behavior. Verify the golden set is
  committed and the baseline profile is reproducible before any
  optimization is enabled.
- [x] 7.2 Implement the direct log-probability optimization before
  higher-risk backend-specific optimizations. Keep selective extraction
  and KV-cache reuse disabled until their capabilities and causal
  correctness are verified. Justify runtime and concurrency changes by
  profiling and benchmark evidence. Verify each optimization is gated off
  by default until its verification tasks pass.
- [x] 7.3 Confirm completion: static analysis and unit tests pass,
  reproducible benchmark artifacts are produced, backend limitations are
  documented, and measured performance changes are reported without
  assuming any proposed optimization necessarily improves latency or
  throughput. Verify the final `ruff`/`mypy`/`pytest` gate and the
  performance report are complete.
