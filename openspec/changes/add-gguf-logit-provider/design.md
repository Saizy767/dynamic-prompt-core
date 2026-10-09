# Design

## Context

The candidate scoring infrastructure lives in
`infrastructure/llm/scoring/`. The `LLMLogitCandidateScorer` orchestrates
prompt construction (`ScoringPromptBuilder`), batch tokenization
(`BatchTokenizerAdapter`), batched model inference
(`BatchedCausalLanguageModel`), and logit-to-score calculation
(`LogitScorer`). The current production adapters are
`HuggingFaceBatchTokenizerAdapter` and `TorchBatchedCausalLMAdapter`,
loaded via `AutoModelForCausalLM.from_pretrained` in
`factory.py:build_candidate_scorer(model_path)`. The seams are
`typing.Protocol` classes, so new adapters can be added without changing
the scorer, logit scorer, or prompt builder.

The factory is called from three sites: `interfaces/cli/main.py`,
`application/use_cases/run_baseline/runner.py`, and
`application/use_cases/compare_versions/comparator.py` (each imports it
lazily inside a function). Configuration today is a TOML `[llm]` section
for the inference server (`loader.py`); the scorer has no backend
selection — it always assumes Hugging Face.

See `proposal.md` for motivation and `specs/` for the behavior contract.

## Goals / Non-Goals

**Goals:**

- GGUF model and tokenizer adapters that satisfy the existing seams so all
  shared scoring logic is reused unchanged.
- Backend selection at the factory, preserving Hugging Face behavior exactly.
- Optional `llama-cpp-python` dependency that never blocks core import/install.
- Context isolation so repeated and concurrent evaluations are independent.
- Experiment metadata sufficient for resume identity verification.

**Non-Goals:**

- Native tensor batching for GGUF (llama.cpp is single-context; see Decisions).
- Identical logits across HF and GGUF backends.
- A second scoring algorithm, metric implementation, or acceptance policy.
- Automatic model download or an HTTP inference server for GGUF.
- A hard performance target before a baseline workload is defined.

## Decisions

### D1: Adapter-pair approach, not a second scorer

Implement `LlamaCppBatchedCausalLMAdapter` (model seam) and
`LlamaCppBatchTokenizerAdapter` (tokenizer seam) that satisfy the existing
`BatchedCausalLanguageModel` and `BatchTokenizerAdapter` protocols. Wire them
into the same `LLMLogitCandidateScorer` + `LogitScorer` + `ScoringPromptBuilder`
used by Hugging Face.

**Output contract (explicit):** The existing seam contract is plain Python
floats — `SequenceLogits.logits: tuple[Sequence[float], ...]` and
`BatchedLogits.items: tuple[SequenceLogits, ...]`; the Torch adapter already
converts tensors via `.tolist()`. The GGUF adapter SHALL match this exactly:
return `BatchedLogits`/`SequenceLogits` (not a raw NumPy/PyTorch array),
semantic shape `[B, S, V]`, plain-float elements, per-item ordering, causal
alignment, and error behavior. There is no tensor/device ambiguity because the
contract is already plain floats. A structural-equivalence test SHALL assert
`LogitScorer` receives identical structural semantics (shape, element type,
position alignment) from both providers.

**Rationale:** The seams were designed exactly for this. Reusing the shared
scorer preserves causal logit alignment, mean-log-probability scoring,
numerical validation, atomic failure, and ordering guarantees without
duplication.

**Alternative considered:** A separate `GGUFLogitCandidateScorer`. Rejected —
it would duplicate the scoring algorithm and violate the spec's "no parallel
abstraction" rule.

### D2: GGUF adapter is a sequential compatibility adapter

`llama-cpp-python` evaluates one token sequence per context; it has no native
batched tensor forward. The GGUF model adapter implements `forward_batch` by
evaluating each batch item sequentially (reset context → eval → read logits,
per item), returning `BatchedLogits` in input order.

**Rationale:** The existing spec permits sequential delegation for
"explicitly supported compatibility adapters." llama.cpp's API is inherently
single-context, so this is the only faithful option. The scoring contract is
preserved: one `forward_batch` call per invocation, correct item order, atomic
failure, no cross-item state leakage.

**Trade-off:** No batched-forward speedup for GGUF. Prefix reuse across
candidates sharing the same `text` is a future optimization (eval the shared
prefix once, then eval each candidate continuation) — noted as a follow-up,
not in scope here.

### D3: Context isolation and memory-aware logit access

Load the `Llama` model once per provider instance. For each batch item,
reset the KV cache (`model.reset()`) before evaluating, so tokens from one
example cannot leak into another. Evaluate with `logits_all=True`.

**Logit access:** `llama-cpp-python` exposes `eval_logits` as a `deque` of
lists materialized from the stored score array; accessing it can copy and
materialize logits for the entire evaluated sequence × vocabulary. The
adapter SHALL read `eval_logits` once per evaluation and index into it for
the positions the `LogitScorer` needs (prefix tail + candidate positions);
it SHALL NOT repeatedly access `eval_logits` inside a loop over candidate
positions. Use the public API where it meets the memory requirement. If
full-vocabulary copies prove too expensive for realistic contexts, investigate
a version-pinned, tested access strategy that reads only the required
positions; if that requires private internals or low-level bindings, isolate
it behind the adapter and add compatibility tests for runtime upgrades. Do
not claim partial-position reads are supported until a working implementation
proves it.

**Rationale:** A single mutable context is unsafe across examples; resetting
per item is the simplest correct isolation. The model load (expensive)
happens once; context reset (cheap) happens per item. Reading the logit
buffer once avoids repeated full-vocabulary materialization.

**Alternative considered:** A fresh `Llama` per evaluation. Rejected —
reloads the model, violating "loaded once per instance."

**Concurrency:** Serialize concurrent calls with a `threading.Lock` around
the eval+read critical section (the context is mutable). Document this as the
worker model. The async `score()` already offloads to a thread via
`asyncio.to_thread`, so the lock makes concurrent thread calls safe.

**Causal alignment (explicit):** For tokens `x_0..x_n`, logits at position
`i` predict token `x_{i+1}`. The first candidate token is scored from the
final prefix position, not the first candidate position. A small deterministic
test with known token IDs SHALL verify every position the `LogitScorer` reads
matches this alignment.

### D4: Full-encode tokenization boundary with prefix-prefix compatibility check

`llama-cpp-python` exposes no offset mapping, so the HuggingFace adapter's
offset-based boundary detection is unavailable. The GGUF tokenizer adapter
SHALL determine the boundary as follows:

1. Encode the full prefix-plus-candidate text with the GGUF tokenizer, with
   special-token (BOS) behavior explicitly configured.
2. Encode the prefix separately using the same BOS and special-token policy.
3. Check whether the prefix token sequence is an exact prefix of the full
   token sequence.
4. If it is, the candidate continuation is the remaining full-sequence tokens;
   set `prefix_token_count = len(prefix_ids)` and
   `candidate_token_ids = full[len(prefix_ids):]`.
5. If it is not, use an explicitly defined fallback or reject the example with
   a clear compatibility error. Do NOT silently score a different token
   sequence than `tokenize(prefix + candidate)`.

Handle an empty candidate without adding BOS twice. The prefix-is-prefix check
is a compatibility check, not a universal solution to token-boundary
ambiguity: if the tokenizer merges across the boundary so no unique
token-level division reproduces the original scoring semantics, the project
SHALL explicitly define the intended boundary or reject the configuration.

**Rationale:** This is a safer default than assuming separate encoding
(`encode(prefix) + encode(candidate)`) is equivalent to `encode(prefix +
candidate)`. The model evaluates the full-text token sequence, so candidate
scoring reflects the tokenization the model would actually see. The existing
`BatchTokenizerAdapter` spec permits a proven-safe strategy and warns that
separate encoding may not equal full encoding; this approach respects that
warning.

**Risk:** When the prefix is not an exact prefix of the full sequence
(cross-boundary merge), there is no unique correct division. The fallback or
rejection must be explicit and tested against the project's actual prompt
format. Integration tests (D7) SHALL verify the test model's boundary
behavior for the project's prompt format and exercise the mismatch path.

### D5: Optional dependency via lazy import

Import `llama_cpp` lazily inside the GGUF adapter constructor, not at module
top level. The factory imports the GGUF adapter module lazily only when
`backend == "gguf"`. Add `llama-cpp-python` to
`[project.optional-dependencies].gguf` in `pyproject.toml`, version-pinned.
Add a `mypy` override for `llama_cpp.*` with `ignore_missing_imports = true`.

**Rationale:** Keeps `import dynamic_prompt_core` and core install working
without the GGUF runtime. A missing dependency surfaces as an actionable
`ConfigError`/`GGUFProviderError` at initialization, not an `ImportError`
deep in scoring.

### D6: Backend selection via a config object

Introduce a `ScorerBackendConfig` dataclass (infrastructure config) with
fields `backend` (`"huggingface"` | `"gguf"`), `model_path`, and an optional
`GgufParams` (`n_ctx`, `n_batch`, `n_threads`, `n_gpu_layers`, `seed`,
`verbose`, `logits_all`). Extend `build_candidate_scorer` to accept this
config and dispatch on `backend`. Update the three call sites to build the
config from the existing TOML (new `[model]` section with `[model.gguf]`
sub-table, validated by the existing `loader.py` pattern).

**Public-API check (verified):** `build_candidate_scorer` is NOT exported in
any `__all__` or package `__init__.py`; it is imported lazily inside three
internal functions (`main.py`, `runner.py`, `comparator.py`). It is therefore
an internal function, and the signature change (`model_path: str` →
`config: ScorerBackendConfig`) is internal-only — no external public API
breaks, so no compatibility shim or version bump is required. The
`CandidateScorer` port is unchanged.

**Rationale:** Extends the existing config mechanism rather than adding a
file. Selecting `"huggingface"` constructs the current adapters with the
same `model_path`, preserving behavior.

**Alternative considered:** Keep `build_candidate_scorer(model_path)` and add
a sibling `build_gguf_candidate_scorer(...)`. Rejected — splits backend
selection across two functions and forces callers to branch.

### D7: Error hierarchy

Add `GGUFProviderError(LLMScoringError)` in
`infrastructure/llm/scoring/errors.py` for GGUF-specific failures (missing
dependency, invalid model, context overflow, invalid logits shape). The
existing `LLMLogitCandidateScorer` already translates `LLMScoringError` and
raw exceptions into `CandidateScoringError`, so GGUF errors flow through the
existing boundary with no application-layer change. Errors carry debugging
context (parameter names, lengths, vocab size) but never log input text or
secrets by default.

### D8: Experiment metadata via a provider describe hook

Add a `describe() -> dict` method on the GGUF model adapter returning backend
id, model path, file content checksum (a reliable content identity, not the
path alone), quantization/metadata (where `llama-cpp-python` exposes it),
`n_vocab`, and inference settings. The run recorder includes this in
experiment metadata alongside the existing dataset fingerprint and prompt
version.

**Resume identity (precise):** Resume compares the recorded content checksum
and semantic settings (model contents, tokenizer, prompt template, scoring
method, dataset) against the current provider's `describe()`. Result-changing
settings are distinguished from non-semantic settings (logging verbosity). A
mismatch in a result-changing setting either rejects the resume or explicitly
records the run as a new experiment — it does not silently continue with a
different configuration. A model path alone is NOT sufficient identity because
file contents can change without the path changing.

**Rationale:** Keeps metadata production in infrastructure (the provider knows
its own settings) without leaking model internals through the port —
`describe()` is infrastructure-internal, called by the recorder, not by the
application port.

## Risks / Trade-offs

- **[No native batching for GGUF]** → Accepted; sequential per-item eval
  inside `forward_batch` preserves the contract. Prefix-reuse optimization
  deferred. Documented as the worker model.
- **[Token-boundary merge across prefix/candidate]** → The full-encode
  prefix-prefix check detects mismatches; on mismatch the adapter rejects or
  applies an explicit fallback rather than silently scoring a different
  sequence. Integration tests exercise both the match and mismatch paths for
  the project's prompt format. This is a compatibility check, not a universal
  solution — a model with no clean division requires an explicitly defined
  boundary or config rejection.
- **[Logit memory with `logits_all=True`]** → `eval_logits` materializes a
  deque of lists for the full sequence × vocabulary. The adapter reads it once
  per evaluation and indexes into it, never inside a position loop. Peak
  memory is measured for realistic context lengths and vocabulary sizes. If
  full-vocabulary copies are too expensive, a version-pinned partial-read
  strategy is investigated and isolated behind the adapter with upgrade
  compatibility tests; partial reads are not claimed until proven.
- **[HF vs GGUF score disagreement]** → Regression tests record agreement,
  macro-F1, per-class F1, accuracy, latency, and peak memory. Significant
  disagreements are investigated and explained (quantization, precision,
  tokenizer, prompt format) — not forced to equality.
- **[Factory signature change]** → `build_candidate_scorer` is internal (not
  exported); the three call sites are updated in the same change. No external
  public API changes (the `CandidateScorer` port is unchanged).
- **[Concurrent context mutation]** → Mitigated by a per-provider lock around
  eval+read; documented as serialized execution.
- **[Resume identity on changed model contents]** → Content checksum, not
  path, is the identity; a result-changing mismatch rejects resume or records
  a new experiment rather than silently continuing.

## Migration Plan

1. Add the optional dependency group and `mypy` override — no runtime impact.
2. Add the GGUF adapters and `GGUFProviderError` behind lazy imports — no
   impact unless `backend == "gguf"` is selected.
3. Extend the factory signature and update the three call sites — default
   backend remains `"huggingface"`, so existing behavior is unchanged.
4. Add the `[model]` config section — optional; absent config defaults to
   Hugging Face.
5. Rollback: revert the factory signature and call sites; the GGUF adapters
   and optional dependency are inert without `backend == "gguf"`.

## Implementation Order

1. Confirm the exact existing protocols and causal alignment
   (`BatchTokenizerAdapter`, `BatchedCausalLanguageModel`, `BatchedLogits`,
   `SequenceLogits`, `LogitScorer`) — those interfaces, not the proposed
   names, determine the final adapter implementation.
2. Build a minimal GGUF adapter and verify its output structure against the
   shared scorer (D1 structural-equivalence test).
3. Fix the prefix/candidate boundary strategy (D4) and test it with the
   actual prompt format, including the mismatch path.
4. Verify memory behavior (read `eval_logits` once, no per-position loop)
   and context isolation (D3).
5. Wire configuration, factory selection (D6), and experiment metadata (D8).
6. Run regression tests against the existing Hugging Face backend.

## Open Questions

- Whether `llama-cpp-python` exposes partial-position logit reads (to avoid
  full-vocabulary copies for long contexts). The adapter uses the public API
  (read once, index) until a working partial-read implementation is proven;
  if private internals are needed, they are isolated behind the adapter with
  runtime-upgrade compatibility tests. This does not change the specs,
  approach, or task breakdown.
- The exact small GGUF fixture for integration/regression tests (model, size,
  license). This is a test-asset choice that does not affect the design.
- The explicit fallback behavior when the prefix is not an exact prefix of
  the full token sequence (reject vs. a model-specific re-division). The
  default is reject with a compatibility error; a fallback, if needed, is a
  model-specific addition that does not change the seam.
