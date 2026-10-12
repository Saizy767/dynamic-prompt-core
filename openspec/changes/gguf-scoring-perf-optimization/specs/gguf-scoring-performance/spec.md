# Spec Delta

## Purpose

Define the performance, instrumentation, concurrency, and benchmarking
behavior of the GGUF-backed candidate scoring path, and the invariant
that optimizations preserve the existing mean-log-probability score and
`CandidateScorer` contract.

## ADDED Requirements

### Requirement: Scoring correctness preserved by optimization
Every optimization in the GGUF scoring path SHALL preserve the candidate
score produced by the baseline implementation. For a fixed model, fixed
backend settings, and fixed inputs, the optimized scorer SHALL return
scores equivalent to the baseline within a documented floating-point
tolerance. The mean-log-probability scoring formula, causal logit
alignment, EOS/special-token handling, empty-candidate and invalid-input
handling, and error-translation policy SHALL remain unchanged. No
optimization SHALL change the scoring formula, normalization, or
ranking policy.

#### Scenario: Optimized score matches baseline
- **WHEN** the same prompt and candidates are scored by the baseline and optimized scorers with identical model settings
- **THEN** each candidate's optimized score equals its baseline score within the documented tolerance

#### Scenario: Scoring formula unchanged
- **WHEN** the optimized scorer computes a candidate score
- **THEN** it is the arithmetic mean of the candidate-token log-probabilities using causal alignment, identical to the baseline formula

#### Scenario: Error policy unchanged
- **WHEN** the optimized scorer receives empty candidates, out-of-range token IDs, or NaN/Inf logits
- **THEN** it raises the same `CandidateScoringError` boundary as the baseline, with no partial results

### Requirement: Scoring reproducibility under fixed settings
For a fixed model file, fixed backend configuration, and fixed inputs,
the optimized scorer SHALL produce identical scores across repeated runs
absent non-determinism in the underlying backend. No optimization SHALL
introduce inter-request or inter-candidate state leakage that changes
scores.

#### Scenario: Repeated runs are reproducible
- **WHEN** the same inputs are scored twice with the same model and settings
- **THEN** the resulting scores are equal

#### Scenario: No state leakage between requests
- **WHEN** two scoring requests are evaluated consecutively
- **THEN** the second request's scores are unaffected by the first request's KV-cache or context state

### Requirement: Per-stage scoring instrumentation
The GGUF scoring path SHALL emit per-stage timing for each scoring
request so a single benchmark run can localize the dominant bottleneck.
The instrumentation SHALL record monotonic timings for lock wait,
context reset, model evaluation, logit read, logit type conversion,
logit NaN/Inf validation, and score computation, plus the counts of
input tokens, scored candidate tokens, vocabulary size, and candidate
count. Timings SHALL be based on a monotonic clock (`time.perf_counter`).
The instrumentation SHALL NOT log user prompt text, candidate text, or
full logit arrays. The instrumentation SHALL distinguish the cost of
`model.eval(...)` from the cost of accessing and processing logits,
including any lazy computation triggered by accessing `eval_logits`:
`eval_ms` SHALL measure only the `eval()` call, while `logits_read_ms`
SHALL measure the access to `eval_logits` (which may materialize logits
lazily on access) so the two costs are separable in the profile.

#### Scenario: Each stage timing is recorded
- **WHEN** a scoring request is evaluated with instrumentation enabled
- **THEN** the emitted record contains a timing for lock wait, reset, eval, logit read, logit convert, logit validate, and scoring stages

#### Scenario: Token and vocab counters are recorded
- **WHEN** a scoring request is evaluated with instrumentation enabled
- **THEN** the emitted record contains input token count, scored token count, vocabulary size, and candidate count

#### Scenario: Timings are monotonic
- **WHEN** stage timings are recorded
- **THEN** each timing is derived from a monotonic clock and is non-negative

#### Scenario: User content is not logged
- **WHEN** instrumentation is emitted
- **THEN** it contains no prompt text, candidate text, or full logit arrays

#### Scenario: Bottleneck is localizable
- **WHEN** a single benchmark run completes
- **THEN** the per-stage timings allow the stage with the largest time share to be identified

#### Scenario: Eval cost separated from logit-access cost
- **WHEN** accessing `eval_logits` triggers lazy materialization of the logit buffer
- **THEN** `eval_ms` measures only the `model.eval(...)` call and `logits_read_ms` measures the `eval_logits` access including that lazy computation, so the two costs are separable

#### Scenario: No double-counting of nested intervals
- **WHEN** adapter sub-stage timings are aggregated into the scorer's forward breakdown
- **THEN** the aggregate and nested timings are related by an explicitly documented rule so no interval is counted twice

#### Scenario: Exception propagation unchanged
- **WHEN** a stage raises an exception during an instrumented scoring call
- **THEN** the exception propagates identically to the un-instrumented path, and completed stage timings remain available in the emitted record

#### Scenario: Documented schema with millisecond units
- **WHEN** a timing record is emitted
- **THEN** all timing fields are non-negative milliseconds following a documented schema, and the counters are integers

### Requirement: Logit extraction scoped to scored positions
The scorer SHALL extract logits only at the positions required to score
the candidate tokens — specifically the positions `m` through `m+n-2`
where `m` is the prefix token count and `n` is the candidate token count
— so the volume of logit data crossing from the backend into Python is
proportional to the number of scored positions rather than the full
sequence length, when the backend API permits. The exact causal
alignment (logits at position `i` predict token `i+1`) SHALL be verified
by tests before selective extraction is enabled. The scorer SHALL
inspect the capabilities of the installed `llama-cpp-python` version and
use selective access where supported; otherwise it SHALL retain a
compatible fallback that evaluates the existing full logits output and
selects the required positions. The scorer SHALL preserve the
normalizing log-sum-exp needed for a mathematically correct
log-probability at each scored position. Selective extraction SHALL NOT
be represented as eliminating full-vocabulary computation inside the
backend unless measurements demonstrate that result; it is a reduction
in saved and transferred volume, not a claimed reduction in compute,
unless proven otherwise.

#### Scenario: Data volume proportional to scored positions
- **WHEN** candidates are scored and the backend supports selective logit retrieval
- **THEN** the number of logit values read into Python is proportional to the number of scored candidate-token positions, not the full sequence length times vocabulary size

#### Scenario: Causal alignment verified before selective extraction
- **WHEN** selective logit extraction is enabled
- **THEN** the positions `m` through `m+n-2` have been verified against the causal alignment by tests before any selective access is used

#### Scenario: Probe based on verified API behavior
- **WHEN** the capability probe decides whether selective extraction is supported
- **THEN** it is based on verified API behavior, not merely the presence of an arbitrary attribute

#### Scenario: EOS and special-token scoring preserved
- **WHEN** selective extraction is enabled and the candidate sequence includes EOS or special-token scoring
- **THEN** any required EOS/special-token scoring behavior is preserved relative to the baseline

#### Scenario: Scores preserved with selective extraction
- **WHEN** selective logit extraction is used
- **THEN** the resulting candidate scores equal the baseline scores within the documented tolerance

#### Scenario: Fallback when selective extraction unavailable
- **WHEN** the installed backend API does not support restricting saved logit positions
- **THEN** the scorer falls back to evaluating the full logits output, selects the required positions, and produces baseline-equivalent scores

#### Scenario: No unmeasured full-vocabulary-compute claim
- **WHEN** selective extraction is described
- **THEN** it is not claimed to eliminate full-vocabulary computation unless a measurement demonstrates that the backend no longer computes the full distribution

### Requirement: Single-token log-probability via stable log-sum-exp
The scorer SHALL compute each candidate token's log-probability directly
as `z_i - LSE(z)` using a numerically stable log-sum-exp (`a = max(z)`,
`LSE = a + log(sum(exp(z_j - a))))`) without materializing the full
normalized log-probability vector when only one token's score is needed.
The implementation SHALL use vectorized NumPy operations when available
and SHALL provide a pure-Python fallback that does not require NumPy.
The implementation SHALL reject non-finite values consistent with the
existing NaN/Inf validation policy and SHALL raise `LLMScoringError` for
empty logit vectors.

#### Scenario: Single-token log-probability is correct
- **WHEN** a candidate token's logit vector and target token ID are given
- **THEN** the computed log-probability equals `z_i - LSE(z)` within floating-point tolerance

#### Scenario: Full normalized vector not materialized
- **WHEN** only one target-token log-probability is required
- **THEN** the full normalized log-probability vector is not constructed

#### Scenario: Stable log-sum-exp on extreme values
- **WHEN** the logit vector contains large-magnitude values
- **THEN** the computation avoids overflow and underflow via the max-shifted log-sum-exp

#### Scenario: Vectorized path with pure-Python fallback
- **WHEN** NumPy is available
- **THEN** the vectorized path is used; and **WHEN** NumPy is unavailable
- **THEN** the pure-Python fallback produces identical results without requiring NumPy

#### Scenario: Non-finite logits rejected
- **WHEN** the logit vector contains NaN or Inf values
- **THEN** the scorer raises the same error as the baseline validation policy

#### Scenario: Empty logit vector raises
- **WHEN** the logit vector is empty
- **THEN** the scorer raises `LLMScoringError` rather than producing a silent invalid score

### Requirement: Shared prompt-prefix KV-cache reuse
When multiple candidates in one `score()` invocation share the same
prompt prefix at the token level, the scorer SHALL tokenize the prefix
once, pre-fill the model context once, and evaluate each candidate's
specific continuation from the shared state, so the number of evaluated
prefix token positions decreases. The scorer SHALL determine shared
prefix at the token level, not the text level, and SHALL NOT treat
textually equal strings as equivalent when their tokenizations differ.
Cache reuse SHALL use a verified `llama-cpp-python` mechanism for saving
and restoring the KV cache, or an equivalent safe context-branching
approach; an unverified reuse mechanism SHALL NOT be used. Because the
model context is mutable, per-candidate state SHALL be isolated so no
candidate's KV-cache contaminates another's logits. The scorer SHALL
fall back to the existing reset-and-evaluate approach when safe reuse is
unavailable or fails validation. Candidate scores SHALL remain
equivalent to the baseline within the documented tolerance.

#### Scenario: Shared prefix evaluated once
- **WHEN** multiple candidates in one invocation share a token-level prompt prefix
- **THEN** the shared prefix token positions are evaluated once rather than once per candidate

#### Scenario: Prefix sharing decided at token level
- **WHEN** two candidates share prompt text but their tokenizations differ
- **THEN** they are not treated as sharing a prefix

#### Scenario: Reuse uses a verified cache mechanism
- **WHEN** prefix reuse is applied
- **THEN** it uses a verified KV-cache save/restore or equivalent safe context-branching mechanism, not an unverified reuse

#### Scenario: Per-candidate state isolated
- **WHEN** candidates share a prefix and are branched from shared state
- **THEN** no candidate's KV-cache state affects another candidate's logits or scores

#### Scenario: Prefix-reuse scores match baseline
- **WHEN** prefix reuse is applied
- **THEN** the candidate scores equal the baseline scores within the documented tolerance

#### Scenario: Fallback when reuse unsupported or fails validation
- **WHEN** the backend cannot safely reuse KV-cache state, or the reuse mechanism fails validation
- **THEN** the scorer falls back to per-candidate reset and eval and produces baseline-equivalent scores

#### Scenario: No unsafe reuse when no verified mechanism exists
- **WHEN** no verified KV-cache save/restore or safe context-branching mechanism is available
- **THEN** the implementation retains the baseline per-candidate reset-and-evaluate path and documents the limitation rather than introducing unsafe reuse

### Requirement: Explicit runtime configuration and measurement
The scorer SHALL expose and log the effective thread count and SHALL
make runtime parameters (`n_threads`, `n_gpu_layers`, `n_batch`,
`n_ctx`, `logits_all`) observable in experiment metadata. Changes to
context size, batch size, GPU configuration (`n_gpu_layers`), or other
runtime settings SHALL be supported by benchmark evidence and memory
checks rather than assumptions. When a compatible GPU-offload backend is
available, GPU offload SHALL be tested; otherwise the CPU configuration
SHALL be documented as the tested path. The selected configuration SHALL
be recorded in the performance report.

#### Scenario: Thread count explicit and logged
- **WHEN** the GGUF adapter is initialized
- **THEN** the effective thread count is explicitly set and recorded in metadata and logs

#### Scenario: Thread setting source and resolution documented
- **WHEN** the effective thread count is recorded
- **THEN** the explicitly configured value, the backend-selected default, and the actual observed CPU thread usage (where measurable) are distinguished, and actual usage is not inferred solely from a configured value

#### Scenario: Runtime parameters recorded
- **WHEN** a scoring run is executed
- **THEN** `n_threads`, `n_gpu_layers`, `n_batch`, `n_ctx`, and `logits_all` are present in run metadata

#### Scenario: Runtime changes backed by benchmark evidence
- **WHEN** `n_ctx`, `n_batch`, or `n_gpu_layers` would be changed
- **THEN** the change is backed by a benchmark and memory-constraint check, and the prior and new values are documented

#### Scenario: GPU offload tested when available
- **WHEN** a compatible GPU-offload backend is available on the target hardware
- **THEN** GPU offload is benchmarked and its result recorded; otherwise the CPU path is documented as the tested configuration

### Requirement: Concurrency semantics of the shared model
The scorer SHALL define the concurrency semantics of the shared GGUF
model object. Concurrent scoring calls SHALL NOT corrupt scores or
cross-contaminate KV-cache state between requests or candidates. The
serialization lock SHALL NOT be removed without proven
thread-safety of the underlying backend. When concurrent access is not
safe, the scorer SHALL bound queued work (via a queue or dedicated
worker) so load cannot accumulate unbounded. The instrumentation SHALL
measure lock-wait time so contention is observable. Concurrency changes
SHALL be evaluated using both request latency and throughput. A bounded
queue or increased parallelism SHALL NOT be assumed to inherently
improve inference speed; any throughput benefit SHALL be demonstrated by
measurement, not presumed.

#### Scenario: Concurrent calls do not corrupt scores
- **WHEN** multiple scoring requests run concurrently against the shared model
- **THEN** each request's scores are correct and unaffected by concurrent requests

#### Scenario: No KV-cache cross-contamination
- **WHEN** concurrent or consecutive candidates are evaluated
- **THEN** no candidate's KV-cache state leaks into another candidate's logits

#### Scenario: Lock retained until thread-safety proven
- **WHEN** the underlying backend's thread-safety is not proven
- **THEN** the serialization lock remains in place

#### Scenario: Queued work is bounded
- **WHEN** concurrent scoring load exceeds the model's safe concurrency
- **THEN** queued work is bounded by a queue or dedicated worker rather than accumulating unbounded

#### Scenario: Queue semantics defined
- **WHEN** a bounded queue or dedicated worker is introduced
- **THEN** its capacity, overload behavior, cancellation behavior, and worker shutdown semantics are explicitly defined

#### Scenario: Lock contention observable
- **WHEN** concurrent scoring calls contend on the model lock
- **THEN** lock-wait time is measured and reported in the instrumentation

#### Scenario: No presumed parallelism benefit
- **WHEN** a bounded queue or increased parallelism is introduced
- **THEN** any throughput improvement is demonstrated by measurement rather than assumed, and latency is evaluated alongside throughput

### Requirement: Standalone scoring benchmark
A benchmark SHALL be provided that runs independently of the full
dataset experiment. The benchmark SHALL cover, at minimum: a short
prompt with two candidates, a long prompt with two candidates, one
prompt with eight candidates, candidates sharing a common prefix,
sequential latency, and parallel throughput. The benchmark SHALL perform
model warmup before measurement and SHALL record first-request latency
separately from steady-state performance. Results SHALL be emitted in a
machine-readable format (JSON or CSV) containing, per scenario: mean,
median, p95 latency, tokens per second, examples per second, peak RAM,
count of evaluated token positions, and score deviation from baseline.

#### Scenario: Benchmark runs independently
- **WHEN** the benchmark is invoked
- **THEN** it runs without executing the full dataset experiment

#### Scenario: Mock-model and real-model modes
- **WHEN** the benchmark is run in CI
- **THEN** a deterministic mock-model mode is available; and **WHEN** hardware profiling is requested
- **THEN** an opt-in real-model mode is available

#### Scenario: Memory sources distinguished
- **WHEN** benchmark results report memory
- **THEN** Python memory (`tracemalloc`), process RSS, and accelerator memory are distinguished rather than reported as a single number

#### Scenario: All scenarios covered
- **WHEN** the benchmark is invoked
- **THEN** it exercises short-prompt, long-prompt, many-candidate, shared-prefix, sequential, and parallel scenarios

#### Scenario: Warmup before measurement
- **WHEN** a benchmark scenario is measured
- **THEN** model warmup runs before the recorded measurements

#### Scenario: First-request latency separated
- **WHEN** benchmark results are emitted
- **THEN** first-request latency is reported separately from steady-state latency

#### Scenario: Machine-readable results with required fields
- **WHEN** the benchmark completes
- **THEN** it emits JSON or CSV with mean, median, p95, tokens/s, examples/s, peak RAM, evaluated token positions, and score deviation from baseline per scenario

### Requirement: Scoring regression tests
The change SHALL include regression tests asserting that the optimized
scorer produces scores equivalent to the baseline within a documented
tolerance (initially targeting an absolute error below `1e-4`) on a
fixed input set, and unit tests covering the new log-sum-exp,
selective-logit, and prefix-reuse paths. Regression SHALL verify both
per-token log-probabilities and final candidate scores against the
baseline, and SHALL check ranking consistency, causal alignment,
non-finite logits, empty logit vectors, special-token handling, fallback
behavior, and candidate isolation. The standard test suite SHALL NOT
require downloading or loading a large model; real-model benchmark and
regression runs SHALL be opt-in and explicit about their model
dependency.

#### Scenario: Optimized candidate scores match baseline in regression
- **WHEN** the regression test suite runs against a fixed input set
- **THEN** optimized candidate scores equal baseline candidate scores within the documented tolerance

#### Scenario: Tolerance not silently relaxed
- **WHEN** a regression comparison cannot meet the documented tolerance
- **THEN** the failure is investigated and documented rather than the tolerance being silently relaxed

#### Scenario: Per-token log-probabilities match baseline
- **WHEN** the regression test suite compares per-token log-probabilities
- **THEN** each optimized per-token log-probability equals the baseline value within the documented tolerance

#### Scenario: Ranking consistency preserved
- **WHEN** the regression test suite compares candidate rankings
- **THEN** the optimized candidate ranking matches the baseline ranking on the fixed input set

#### Scenario: Causal alignment verified
- **WHEN** the regression test suite checks alignment
- **THEN** each candidate token is scored from the preceding position's logit distribution, matching the baseline

#### Scenario: Non-finite, empty, and special-token cases covered
- **WHEN** the regression test suite exercises non-finite logits, empty logit vectors, and special tokens
- **THEN** the optimized scorer behaves identically to the baseline (same errors, same handling)

#### Scenario: Fallback and candidate isolation covered
- **WHEN** the regression test suite exercises fallback paths and shared-prefix candidate isolation
- **THEN** fallback produces baseline-equivalent scores and no candidate's state contaminates another

#### Scenario: New paths unit-tested with doubles
- **WHEN** the unit test suite runs
- **THEN** log-sum-exp, selective logit extraction, and prefix reuse are exercised with deterministic doubles and no large model

#### Scenario: Real-model runs are opt-in
- **WHEN** a real-model benchmark or regression run exists
- **THEN** it is opt-in, isolated from the default suite, and explicit about its model dependency

### Requirement: Incremental optimization with independent benchmarking and fallback
The implementation SHALL proceed incrementally: first establish the
standalone benchmark, instrumentation, and a reproducible baseline
profile, then implement direct log-probability calculation and selective
logit extraction, then investigate shared-prefix KV-cache reuse, then
evaluate runtime configuration and concurrency. The scoring contract
and a fixed golden test set SHALL be established before changing logit
processing or cache behavior. The direct log-probability optimization
SHALL be implemented before higher-risk backend-specific optimizations.
Selective extraction and KV-cache reuse SHALL remain disabled until their
capabilities and causal correctness are verified. Each optimization
SHALL be benchmarked independently of the others so its individual effect
is measurable. Each optimization SHALL retain a safe fallback to the
existing implementation whenever correctness, compatibility, or a
performance improvement cannot be demonstrated. No optimization SHALL be
enabled by default until its regression equivalence and fallback are
verified. Completion requires passing static analysis and unit tests,
reproducible benchmark artifacts, documented backend limitations, and
reported measured performance changes without assuming any proposed
optimization necessarily improves latency or throughput.

#### Scenario: Golden test set established before changes
- **WHEN** logit processing or cache behavior is about to change
- **THEN** the scoring contract and a fixed golden test set have been established and committed beforehand

#### Scenario: Incremental implementation order
- **WHEN** the implementation is sequenced
- **THEN** benchmark and instrumentation precede log-probability and selective extraction, which precede prefix reuse, which precede runtime and concurrency evaluation

#### Scenario: Low-risk optimization before backend-specific
- **WHEN** optimizations are sequenced
- **THEN** the direct log-probability optimization is implemented before selective extraction and KV-cache reuse

#### Scenario: Each optimization benchmarked independently
- **WHEN** an optimization is introduced
- **THEN** it is benchmarked in isolation so its individual effect on latency, throughput, and memory is measurable apart from the others

#### Scenario: Safe fallback retained per optimization
- **WHEN** an optimization cannot demonstrate correctness, compatibility, or a performance improvement
- **THEN** the implementation falls back to the existing path for that optimization without disabling the others

#### Scenario: Optimizations not enabled by default until verified
- **WHEN** an optimization has not yet verified regression equivalence and fallback
- **THEN** it is not enabled by default

#### Scenario: Completion does not assume improvement
- **WHEN** the change is reported complete
- **THEN** measured performance changes are reported, including the absence of improvement where applicable, rather than assuming any optimization necessarily improves latency or throughput

### Requirement: Performance report and backend limitations documented
The change SHALL produce a performance report comparing baseline and
optimized measurements on the target hardware, including the measured
speedup (or its absence) and the dominant bottleneck before and after
optimization. The report SHALL document backend limitations (selective
logit support, KV-cache reuse support, thread-safety) and the hardware
and software configuration used. A target speedup ratio SHALL NOT be
committed before the baseline profile is obtained.

#### Scenario: Baseline and optimized compared
- **WHEN** the performance report is produced
- **THEN** it compares baseline and optimized latency/throughput/RAM on the target hardware

#### Scenario: Bottleneck reported before and after
- **WHEN** the performance report is produced
- **THEN** it identifies the dominant bottleneck before and after optimization

#### Scenario: Backend limitations documented
- **WHEN** the performance report is produced
- **THEN** it documents selective-logit, KV-cache-reuse, and thread-safety support of the used backend

#### Scenario: No unsupported speedup commitment
- **WHEN** a target speedup is stated
- **THEN** it is stated after, and grounded in, the baseline profile rather than promised a priori
