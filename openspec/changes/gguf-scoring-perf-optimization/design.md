# Design

## Context

The GGUF scoring path lives in `infrastructure/llm/scoring/` and is
wired through `LLMLogitCandidateScorer` → `LlamaCppBatchTokenizerAdapter`
→ `LlamaCppBatchedCausalLMAdapter` → `LogitScorer`. The adapter
(`llama_cpp_adapter.py:153`) evaluates each batch item sequentially
inside a `threading.Lock` critical section: `reset()` → `eval()` →
read `eval_logits` once, then converts every position's full-vocabulary
vector to Python floats and validates NaN/Inf element-wise.
`LogitScorer._log_softmax` (`logit_scorer.py:30`) builds the complete
normalized log-probability list per candidate token even though only one
entry is read. All candidates in one `score()` call share the same
prompt prefix but each triggers a full `reset()` + `eval()`, so the
prefix is recomputed per candidate.

Backend settings come from `GgufParams` (`config.py`):
`n_ctx=2048`, `n_batch=512`, `n_threads=None`, `n_gpu_layers=0`,
`logits_all=True`. The model is `Qwen2.5-3B-Instruct Q4_K_M` with
`n_vocab=151 936`. Observed forward time dominates (~99.8% of a scoring
request), but current logs cannot split `eval()` from logit read and
post-processing.

See `proposal.md` for motivation and `specs/gguf-scoring-performance/spec.md`
for the behavior contract. The existing `llm-logit-scorer` contracts
(port, causal alignment, mean-log-probability, atomic failure, information
boundary) are preserved and are NOT restated here.

## Goals / Non-Goals

**Goals:**

- Localize the real bottleneck with per-stage instrumentation before any
  algorithmic change.
- Reduce C++→Python logit volume to the scored positions only.
- Compute each target-token log-probability directly via stable
  log-sum-exp without the full normalized vector.
- Reuse the shared prompt-prefix KV-cache across candidates in one
  invocation with per-candidate isolation.
- Make runtime config explicit, measured, and recorded.
- Define and measure concurrency semantics; bound queued work.
- Provide a standalone benchmark and regression tests proving score
  equivalence to the baseline.

**Non-Goals:**

- Changing the scoring formula, normalization, ranking, or the
  `CandidateScorer` port contract.
- A hard speedup target before the baseline profile is measured.
- Native tensor batching for `llama-cpp-python` (single-context API).
- Replacing `llama-cpp-python` with a different backend runtime.
- Changing `n_ctx` or `n_batch` without a benchmark and memory check.
- Removing the serialization lock without proven backend thread-safety.

## Decisions

### D1: Instrumentation first, behind a monotonic-timer collector

Add a small timing collector (using `time.perf_counter()`) that records
`lock_wait_ms`, `reset_ms`, `eval_ms`, `logits_read_ms`,
`logits_convert_ms`, `logits_validate_ms`, and `scoring_ms` per scoring
request, plus counters `input_tokens`, `scored_tokens`, `vocab_size`,
`candidate_count`. The collector is infrastructure-internal and emits a
structured record (logged at info/debug or returned from a benchmark
hook); it SHALL NOT embed prompt text, candidate text, or logit arrays.
Instrumentation is added to the existing critical section without
changing the scoring algorithm, so the first measured run is a faithful
baseline profile.

**Eval vs. lazy logit materialization:** `llama-cpp-python`'s
`eval_logits` may materialize the logit buffer lazily on access, so the
cost of *reading* logits can include computation that is logically
distinct from `eval()`. The collector SHALL place the timing boundary so
`eval_ms` measures only the `model.eval(...)` call, and
`logits_read_ms` measures the subsequent access to `eval_logits`
(including any lazy materialization it triggers). This separation is the
point of the instrumentation: it lets the profile distinguish model
evaluation from logit access/processing rather than lumping them into
one "forward" number.

**Rationale:** The user's logs show forward dominates but cannot
attribute it, and cannot even separate `eval` from `eval_logits` access.
Measuring first — with the eval/logit-access split — avoids optimizing
the wrong stage and grounds any later speedup claim.

**Alternative:** External profilers (`cProfile`, `py-spy`). Rejected as
the primary mechanism — they do not split `eval` from `eval_logits`
access inside one C-critical section and are not part of the runnable
benchmark.

### D2: Selective logit extraction with a proven-API gate

The scorer needs logits only at positions `m .. m+n-2` (the prefix tail
and candidate positions that predict candidate tokens). Before enabling
selective extraction, the exact causal alignment (logits at position `i`
predict token `i+1`) SHALL be verified by tests so the position set
`m .. m+n-2` is proven correct, and any required EOS/special-token
scoring behavior is preserved. The adapter SHALL then probe the
installed `llama-cpp-python` version for the capability to restrict
saved positions; the probe MUST be based on verified API behavior, not
merely the presence of an arbitrary attribute. If unavailable or
unreliable, it SHALL fall back to the current `eval_logits` read and
index into the result for the required positions. Selective extraction
reduces the volume crossing into Python but does NOT eliminate computing
the full distribution inside the backend — it controls what is *saved
and transferred*. This design SHALL NOT represent selective extraction
as eliminating full-vocabulary computation unless a measurement
demonstrates that the backend no longer computes the full distribution.
The normalizing log-sum-exp at each scored position still uses the full
vocabulary vector from that position.

**Rationale:** The `[T x V]` materialization (vocab 151 936) is the
most likely Python-side cost after `eval`. Restricting saved positions
is the highest-leverage reduction that preserves correctness — but only
the saved/transferred volume is reduced; compute reduction is a separate
claim that requires its own evidence.

**Alternative:** A native C++ shim reading only target-token logits.
Deferred — isolated behind the adapter only if the public API is
insufficient and proven with upgrade-compatibility tests.

### D3: Direct single-token log-probability via stable log-sum-exp

Replace `LogitScorer._log_softmax` (which builds the full normalized
list) with a function computing the target token's log-probability
directly: `z_i - LSE(z)` where `LSE(z) = m + log(sum(exp(z_j - m)))`,
`m = max(z)`. Use a vectorized path (NumPy) when available and a
pure-Python fallback otherwise. Preserve the existing NaN/Inf policy:
reject non-finite logits; define explicit behavior for empty vectors
(raise `LLMScoringError`). The mean over candidate-token
log-probabilities is unchanged.

**Rationale:** The current code allocates a full log-prob list per
candidate token and reads one entry. Direct computation removes the
allocation and the per-element Python loop.

**Alternative:** Keep `_log_softmax` and index it. Rejected — it is the
allocation we are removing.

### D4: Shared-prefix KV-cache reuse with token-level detection and isolation

Within one `score(text, candidates)` call, all candidates share the same
prompt prefix. The adapter SHALL tokenize the prefix once, pre-fill the
model context once, and branch each candidate's continuation from the
shared state. Because `llama-cpp-python` uses a single mutable context,
branching SHALL use a *verified* mechanism for saving and restoring the
KV cache (or an equivalent safe context-branching approach); an
unverified reuse mechanism SHALL NOT be used. Per-candidate state SHALL
be isolated so no candidate's continuation contaminates another's
logits. Prefix sharing is decided at the token level: textually equal
strings with different tokenizations are NOT shared. A fallback to the
current per-candidate reset+eval path SHALL be used when safe reuse is
unavailable or fails validation.

**Rationale:** Re-evaluating the shared prefix per candidate is
redundant; prefill-once is the main compute win after logit extraction.
Requiring a verified save/restore mechanism prevents silent state
corruption from an unverified reuse.

**Risk:** KV-cache save/restore may not be exposed by the installed
version, or the exposed mechanism may not be safe under the project's
usage. The fallback preserves correctness; reuse is enabled only when a
working, tested save/restore path is proven and validation passes.

### D5: Explicit, measured runtime configuration

Log the effective `n_threads` at adapter init, distinguishing three
sources: the explicitly configured value, the backend-selected default
(when `None`), and the actual observed CPU thread usage where measurable.
Do not infer actual usage solely from a configured value; record the
source/resolution status alongside the value. Sweep thread counts in the
benchmark. Do not change `n_ctx`, `n_batch`, or `n_gpu_layers` without
benchmark evidence, an explicit memory check, and a review checkpoint
documenting prior and new values. Test GPU offload (`n_gpu_layers > 0`)
only when a compatible backend is present; otherwise document the CPU
path as tested. Record all runtime parameters (`n_threads`, `n_gpu_layers`,
`n_batch`, `n_ctx`, `logits_all`) in `describe()` metadata so the
performance report is reproducible.

### D6: Concurrency — keep the lock, bound the queue, measure contention

The shared mutable context requires serialization. The lock SHALL remain
until backend thread-safety is proven. The instrumentation (D1) measures
`lock_wait_ms` so contention is observable. For concurrent scoring, bound
queued work with a bounded queue or dedicated worker so load cannot
accumulate unbounded; explicitly define queue capacity, overload behavior,
cancellation behavior, and worker shutdown semantics. Evaluate concurrency
with both request latency and throughput (examples/s). A bounded queue or
increased parallelism SHALL NOT be assumed to inherently improve inference
speed — under a single mutable context, parallelism may only increase
contention. Any throughput benefit SHALL be demonstrated by measurement
using the same workload and hardware, not presumed. If a future backend
version proves safe concurrent contexts, the lock can be relaxed — that
decision is deferred pending measurement.

### D7: Standalone benchmark and machine-readable output

A benchmark module (under `tools/` or `tests/benchmark/`) runs
independently of the full 6 091-example experiment. It supports a
deterministic mock-model mode for CI and an opt-in real-model mode for
hardware profiling. Scenarios: short prompt + 2 candidates, long prompt +
2 candidates, 1 prompt + 8 candidates, candidates with a shared prefix,
sequential latency, parallel throughput. Each scenario performs an
explicit warm-up phase and records cold-start/first-request latency
separately from steady-state. Output is JSON/CSV with mean, median, p95,
tokens/s, examples/s, evaluated token positions, candidate/token counts,
and score deviation from the baseline scorer. Memory is reported with
sources distinguished: Python memory via `tracemalloc`, process RSS via
`resource`, and accelerator memory where measurable — not a single
number. Results also record model/backend versions, runtime parameters,
hardware configuration, warm-up policy, sample count, and measurement
units. Real-model runs are opt-in, gated behind an explicit flag with a
clear skip reason, and isolated from the default unit suite; the default
`pytest` run MUST NOT require a large model, network access, or a GPU.

### D8: Regression equivalence on a fixed set

A regression test scores a fixed input set with both the baseline path
(preserved behind a flag or a kept baseline adapter) and the optimized
path, asserting equivalence within a documented tolerance (initially
`abs(a - b) < 1e-4`). Regression SHALL verify both per-token
log-probabilities and final candidate scores against the baseline, and
SHALL check ranking consistency, causal alignment, non-finite logits,
empty logit vectors, special-token handling, fallback behavior, and
candidate isolation. A tolerance that cannot be met SHALL be
investigated and documented rather than silently relaxed. A fixed golden
test set SHALL be established and committed before any logit-processing
or cache-behavior change so regression has a stable reference. Unit tests
with deterministic doubles cover the new log-sum-exp, selective-extraction,
and prefix-reuse paths without a large model.

### D9: Incremental, independently benchmarked optimizations with fallback

Each optimization (D2, D3, D4, D5/D6) SHALL be introduced incrementally
and benchmarked in isolation so its individual effect on latency,
throughput, and memory is measurable apart from the others. The direct
log-probability optimization (D3, low-risk, pure-Python/NumPy) SHALL be
implemented before the higher-risk backend-specific optimizations (D2
selective extraction, D4 KV-cache reuse). Selective extraction and
KV-cache reuse SHALL remain disabled until their capability probes and
causal correctness are verified. Each SHALL retain a safe fallback to
the existing path, gated so it is not enabled by default until its
regression equivalence and fallback are verified. Combined optimizations
MAY be evaluated after their individual effects are measured. This makes
it possible to attribute any measured speedup to a specific change and to
roll back a single optimization without disabling the others.

## Risks / Trade-offs

- **[Selective logit API absent]** → Probe-and-fallback (D2); correctness
  preserved via full-read fallback. Selective reads are claimed only when
  proven on the installed version. Full-vocabulary-compute reduction is
  not claimed unless separately measured.
- **[KV-cache save/restore absent or unsafe]** → Fallback to per-candidate
  reset+eval (D4); prefix reuse is enabled only when a verified, tested
  save/restore path exists and validation passes.
- **[Log-sum-exp numerical drift]** → Stable max-shifted formulation (D3)
  and per-token regression tolerance (D8) guard correctness; NaN/Inf and
  empty-vector policies unchanged.
- **[Lock contention under concurrency]** → Measured (D1/D6); queue bounds
  load. Removing the lock is deferred until backend thread-safety is proven.
- **[Assumed parallelism benefit]** → A bounded queue or extra parallelism
  is not presumed to improve speed under a single mutable context; any
  throughput gain is demonstrated by measurement (D6).
- **[Backend version drift]** → Capability probes (D2/D4) gate optional
  paths; fallbacks keep behavior correct across versions.
- **[No a-priori speedup]** → The baseline profile (D1) is produced before
  any speedup is claimed; each optimization is benchmarked independently
  (D9) and the report states the measured ratio or its absence.

## Migration Plan

1. Add instrumentation (D1) only — no algorithm change; produce the
   baseline profile with the eval/logit-access split.
2. Add direct log-sum-exp (D3) and selective logit extraction (D2) behind
   capability probes; verify causal alignment by tests first; validate
   each in isolation against the baseline (D9).
3. Add prefix reuse (D4) behind a verified save/restore probe with
   fallback; benchmark independently (D9).
4. Add runtime config logging (D5) and concurrency bounding (D6);
   evaluate with both latency and throughput.
5. Add the benchmark (D7) and regression tests (D8), including per-token,
   ranking, special-token, fallback, and isolation checks.
6. Rollback: each optimization is independently gated and falls back to
   the baseline path; reverting one flag restores that optimization's
   baseline behavior without disabling the others.

## Open Questions

- Whether the installed `llama-cpp-python` exposes partial-position logit
  reads and KV-cache save/restore. Probed at runtime; fallbacks preserve
  correctness. Does not change the specs or task breakdown.
- The exact floating-point tolerance for regression equivalence. Set
  conservatively (`1e-4` absolute) and tightened after the baseline
  profile; a tolerance change does not alter the spec contract.
- The small GGUF fixture for opt-in benchmark/regression runs (model,
  size, license). A test-asset choice that does not affect the design.
