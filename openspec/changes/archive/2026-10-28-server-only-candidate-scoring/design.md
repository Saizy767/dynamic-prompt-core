# Design

## Context

The classifier scores candidates by mean log-probability of candidate tokens
(`LogitScorer` with causal alignment). Three backends exist today: GGUF
in-process (default, 20-47 s/row in Python logit conversion), HuggingFace
(CPU-only, same conversion pattern), and llama_server HTTP (built on
`prompt_logprobs`, which the running llama.cpp 0.5.0 does not expose — the
adapter fails at runtime with "switch the scorer backend to 'gguf'").

Live probing of the running llama-server established the API surface this
design relies on:

- `/tokenize` returns token IDs (`add_special: false`).
- `/v1/completions` accepts `prompt` as a list of token IDs.
- With `logit_bias {<id>: 100}`, `max_tokens: 1`, `logprobs: 1`,
  `temperature: 0`, the server generates exactly the forced token and reports
  its **raw, pre-bias, full-vocabulary** log-prob (validated: forced `0` ->
  -14.6238, `1` -> -12.167518; identical to grammar-forced values; the
  combined `root ::= "0" | "1"` grammar picks the same winner).
- `top_logprobs` in responses are raw and unmasked; there is no
  renormalization anywhere in the reported values.
- No `prompt_logprobs` on `/completion`; `/v1/completions` `echo` returns
  logprobs only for generated tokens.
- The server caches prompt prefixes across requests (`cached_tokens` in
  usage), so per-candidate requests sharing a prefix are cheap.

## Goals / Non-Goals

**Goals:**
- Exact mean-log-prob scoring semantics, byte-identical to the current GGUF
  scorer, with zero Python-side logit materialization.
- Single scoring backend: the already-running llama-server (all-GPU).
- Delete GGUF and HuggingFace scoring paths and their optional dependency.

**Non-Goals:**
- Extraction (`AsyncTask`), embeddings (MPS), teacher, launcher argv, vLLM.
- Server-side slot count tuning (`-np`).
- Any change to `LogitScorer`, the `Classifier` port, or domain/application
  layers.

## Decisions

### D1: `logit_bias` forcing over top-N lookup and over grammar masking
Alternatives, all validated live:
- **top_logprobs window (top-20)**: works, but needs a floor policy for
  tokens outside the window and depends on the window size.
- **GBNF grammar masking** (`root ::= "0" | "1"`): works, but for multi-char
  candidates the grammar admits any token that is a *prefix* of the remaining
  string (a bare space token is a legal first token of `" that"`), so the
  generated token can diverge from the candidate's tokenization; also needs
  text escaping and would require passing candidate *texts* through the seam.
- **Mask-to-pair + derive the loser from the winner's top-1**: invalid — the
  reported log-probs are raw, not renormalized over the masked set, so
  `P(0) + P(1) != 1` in the response and the derived score would be silently
  wrong.
- **`logit_bias {token: +100}`**: forces the exact token *by ID* — no text
  escaping, no seam extension (the adapter already receives
  `candidate_token_ids`), exact for multi-token candidates (bias forces the
  precise ID at each step), and the reported log-prob is the true raw value.

Chosen: `logit_bias`. The identity check (returned ID == forced ID) stays as a
paranoia guard.

### D2: One request per candidate token, sequential per candidate
Requests for the tokens of one candidate are sequential by construction
(token `k` needs tokens `0..k-1` in the prompt). Requests across candidates
run concurrently (shared prefix is server-cached). For `"0"`/`"1"` this is 2
requests per row; the second hits the prompt cache.

### D3: Sparse logit vectors at the seam, `LogitScorer` untouched
The adapter builds `SequenceLogits` of full sequence length with
`SparseLogitVector(actual_token=c_k, actual_logit=lp)` at scored positions
and floor vectors elsewhere. `LogitScorer`'s `log_softmax` path returns the
exact `lp` for the candidate token. No new scoring math anywhere.

### D4: Required kwargs on `forward_batch`
`prefix_token_counts` and `candidate_token_ids` become required (explicit
error when missing). The classifier already passes them; without them the
mechanism cannot know which tokens to force. No fallback path.

### D5: Config collapses to one backend
`VALID_SCORER_BACKENDS = ("llama_server",)`; `GgufParams`, the `gguf` field,
and `model_path` are removed from `ScorerBackendConfig` (the server owns the
model; the classifier needs no path). Absent `[model]` defaults to
`llama_server` with the default endpoint. Call sites drop the
`default_model_path` argument.

### D6: Delete, don't deprecate
The GGUF/HF backends are removed outright: the GGUF path is a performance
trap (validated 20-47 s/row), the HF path is CPU-only, and both carry the
pure-Python conversion this change exists to eliminate. A deprecation
shim would keep dead code alive. Score equivalence is verified against a
GGUF reference capture **before** the deletion lands (see tasks).

### D7: Superseded unarchived changes are removed at archive time
`migrate-candidate-scorer-to-llama-server`, `add-gguf-logit-provider`, and
`gguf-scoring-perf-optimization` are complete but unarchived changes whose
implemented behavior this change replaces or deletes. Archiving them would
sync superseded requirement texts into main specs (e.g. the
`prompt_logprobs` mechanism) that this change immediately contradicts. They
are removed as superseded planning artifacts when this change archives; git
history preserves them.

## Risks / Trade-offs

- [llama.cpp extension coupling] `logit_bias` on `/v1/completions` is a
  llama.cpp extension of the OpenAI surface → validated live on 0.5.0; the
  equivalence check re-validates on real data.
- [Float32 CPU-vs-Metal drift] GGUF reference scores come from a 16-layer
  GPU / 20-layer CPU split; server scores from all-Metal → different
  llama.cpp compute kernels accumulate float32 differently through 36
  layers. Measured on 20 real rows: decisions identical on all rows,
  probability drift up to 0.064. Equivalence criterion: decision match is
  hard, probability delta bounded at 0.1.
- [Single server slot] Scoring and extraction share one llama-server slot;
  requests queue → acceptable at `[runner] concurrency = 2`; future lever:
  `-np` in launcher argv (non-goal here).
- [Large deletion surface] ~15 test files touched → full gate suite
  (pytest, mypy strict, ruff, lint-imports) must pass before the change is
  considered done.

## Migration Plan

1. Capture GGUF reference scores (artifact) while the GGUF backend still
   works.
2. Implement the server adapter + config + factory; verify equivalence
   against the artifact.
3. Delete GGUF/HF code and tests; run gates.
4. Rollback: git revert of the change; the GGUF backend returns with its
   config section (no data migrations involved — scoring artifacts record
   backend metadata, and old runs remain interpretable).

## Open Questions

None — all mechanism questions were resolved by live probing during spec
formation.
