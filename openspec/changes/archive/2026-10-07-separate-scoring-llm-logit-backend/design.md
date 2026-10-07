# Design

## Context

The candidate-scoring contracts are implemented: `Candidate` and `Judgment`
(`domain/models/`), the `CandidateScorer` port
(`application/ports/outbound/candidate_scorer.py`), and
`validate_unique_candidates`. The layered architecture with import-linter
contracts is in place. Classification still flows through the generative path:
`BaselineRunner` → `AsyncTask.classify_detailed` → `ClassificationResult`
(Pydantic `decision` + `confidence`). The existing `LLMClient` port is built
around generative structured output. `transformers.AutoTokenizer` is already
used in `infrastructure/llm/client.py`; the model is served over an
OpenAI-compatible HTTP endpoint.

This change introduces the application scoring-classification flow and the
infrastructure logit-backed scorer on top of the existing contracts, without
modifying the generative path or wiring the new flow into any existing caller.

## Goals / Non-Goals

**Goals:**
- Implement the application flow: `Classification` domain model,
  `ClassificationPolicy` application service + concrete argmax policy,
  `ClassifyInput` use case.
- Implement the infrastructure `LLMLogitCandidateScorer` with testable seams
  (tokenizer, model, prompt builder, logit scorer).
- Define the initial logit-to-score algorithm with formal causal logit
  alignment, explicit multi-token candidate handling, and mean length
  normalization.
- Translate infrastructure failures into an application-facing
  `CandidateScoringError` that lives in the application layer.
- Keep all model internals behind the port boundary; preserve the generative
  path; do not wire the new flow into existing callers.

**Non-Goals:**
- Migrate any existing caller off the generative path or wire the new flow into
  production.
- Calibrate scores, normalize to `[0, 1]`, or introduce probability semantics.
- Introduce a `Score` value object or `CandidateSet` type.
- Implement prefix reuse, KV-cache optimization, or provider batch APIs.
- Define production classification policy beyond argmax.
- Remove `ClassificationResult`, `LLMClient.classify`, or `BaselineRunner`.
- Prescribe the concrete async execution mechanism (thread offloading,
  executor, etc.) for synchronous model inference.

## Decisions

### Decision: `Classification` is a frozen dataclass with `tuple[Judgment, ...]` in `domain/models/classification.py`
Follows the existing `Candidate`/`Judgment` pattern: `@dataclass(frozen=True)`,
stdlib-only, no Pydantic. Fields: `selected: Candidate` and
`judgments: tuple[Judgment, ...]`. A tuple (not `list`) ensures deep
immutability: `frozen=True` prevents reassigning the field, and a tuple prevents
`append`/`__setitem__` on the judgment sequence. The constructor accepts the
judgments as an iterable and stores `tuple(judgments)`. Invariant: `judgments`
is non-empty and `selected` is among the candidates in `judgments`.
- **Alternative considered**: `list[Judgment]` with `frozen=True`. Rejected:
  `frozen=True` only prevents field reassignment; the list itself remains
  mutable (`classification.judgments.append(...)` would succeed), violating the
  immutability the spec promises.
- **Alternative considered**: Pydantic `BaseModel` in `application/schemas/`.
  Rejected: `Classification` is a domain decision object, not an I/O schema; the
  architecture spec defaults `domain` to stdlib-only and Pydantic-free.
- **Alternative considered**: store only the selected candidate, not the
  judgments. Rejected: retaining the judgments permits offline re-ranking,
  threshold experiments, and score inspection without re-scoring.

### Decision: `ClassificationPolicy` is a `typing.Protocol` in `application/services/classification_policy.py`; concrete `ArgmaxClassificationPolicy` in the same module
The policy is an application-internal strategy abstraction, NOT an outbound
port and NOT an external dependency. It is a pure decision rule with no
infrastructure coupling. Placing the protocol and concrete implementation
together in `application/services/classification_policy.py` keeps it clearly an
application service, avoids falsely implying it is an infrastructure-implemented
port, and follows the existing `application/services/` convention (`metrics`,
`clustering`, `prompt_render`). The `ClassifyInput` use case depends on the
`ClassificationPolicy` protocol (enabling test doubles and strategy
substitution), not the concrete implementation. Because the policy is not under
`application/ports/`, the `application-ports-isolated` contract does not cover
it; the policy module simply does not import infrastructure (it is pure
application logic).
- **Alternative considered**: place the protocol in `application/ports/outbound/`.
  Rejected: ports are contracts for external dependencies implemented by
  infrastructure; the classification policy is not infrastructure-implemented,
  so calling it an outbound port creates a false impression that classification
  is delegated to an external adapter.
- **Alternative considered**: make the policy an infrastructure-implemented port.
  Rejected: the classification decision is a semantic rule owned by the
  application; making it infrastructure would make it untestable without LLM
  dependencies and violate the evaluation-vs-decision separation.

### Decision: Initial policy is argmax with first-in-order tie-breaking; empty judgments raise `ValueError`
`ArgmaxClassificationPolicy.classify(judgments)` selects the candidate with the
highest score. When two or more candidates share the maximum score, the one
appearing first in the input judgment list is selected (deterministic, no
randomization). Empty `judgments` raises `ValueError` — no silent default
classification. The policy does not calibrate, normalize, or rescale. The
policy preserves the input judgment order in `Classification.judgments` so
tie-breaking and downstream inspection have no hidden dependency on sort order.
- **Alternative considered**: raise on ties. Rejected: ties are legitimate at
  low score resolution; a deterministic first-in-order rule keeps the flow
  usable while remaining explicit and testable.
- **Alternative considered**: return a "no candidate" result on empty judgments.
  Rejected: adds a representation the specs do not require yet; raising is the
  minimal explicit behavior.

### Decision: `ClassifyInput` use case in `application/use_cases/classify_input/` with a typed `ClassifyInputDeps`
Follows the existing use case pattern (`run_cycle/`, `select_candidates/`):
directory with `__init__.py`, a use case function/module, and a typed frozen
dependency dataclass. `ClassifyInputDeps(scorer: CandidateScorer, policy:
ClassificationPolicy)` is passed from outside. The use case calls
`validate_unique_candidates` (already in the port module), rejects empty
candidate lists, awaits `scorer.score(text, candidates)`, then calls
`policy.classify(judgments)`. It imports only the `CandidateScorer` outbound
port, the `ClassificationPolicy` application service protocol, and domain types.
- **Alternative considered**: pass scorer and policy as loose function
  arguments. Rejected: the architecture spec requires a typed dependency object
  when a use case has multiple dependencies.

### Decision: `LLMLogitCandidateScorer` and its seams live in `infrastructure/llm/scoring/`
A new subpackage under the existing `infrastructure/llm/` keeps scoring code
cohesive and separate from the generative `client.py`/`adapters.py`. Modules:
`candidate_scorer.py` (the `CandidateScorer` implementation), `prompt_builder.py`
(prompt construction), `tokenizer_adapter.py` (tokenizer seam),
`model_adapter.py` (`CausalLanguageModel` protocol + concrete adapter),
`logit_scorer.py` (logit-to-score algorithm), `errors.py` (infrastructure-internal
`LLMScoringError`). This resolves the open question from
`candidate-scoring-architecture-prep` (infrastructure/llm/ vs.
infrastructure/scoring/) in favor of keeping LLM-related code together.
- **Alternative considered**: a top-level `infrastructure/scoring/`. Rejected:
  the scorer is LLM-specific; co-locating with `infrastructure/llm/` keeps
  related adapters and error conventions together.
- **Alternative considered**: flat modules in `infrastructure/llm/`. Rejected: a
  subpackage separates scoring from generative transport and groups the seams
  without polluting the top-level LLM directory.

### Decision: Tokenization boundary by separate prefix/candidate encoding, not substring search
The prompt builder produces `prefix = "{input}\nCandidate: "` and
`candidate = candidate.value`. The tokenizer encodes `prefix` and `candidate`
as separate inputs. The backend records `prefix_token_count = len(encode(prefix))`
and `candidate_token_ids = encode(candidate)`, then composes
`full_sequence = encode(prefix) + candidate_token_ids`. The candidate token
positions are known structurally from `prefix_token_count`, not discovered by
searching the full tokenization for the candidate substring or token subsequence.
This is reliable because tokenizers may merge or split tokens across the
boundary (e.g., the space after "Candidate:" may merge with the first candidate
token), making post-hoc substring search unreliable.
- **Alternative considered**: encode the full prompt and search for candidate
  token positions. Rejected: tokenizer boundary effects (BPE merges, leading
  space tokens) make substring/subsequence search fragile and
  model-dependent.

### Decision: Candidate value used verbatim for tokenization
The scorer tokenizes `candidate.value` exactly as supplied — no stripping,
lowercasing, casefolding, or whitespace normalization. This is consistent with
the `Candidate` domain model's no-auto-normalization rule. The prompt format
`"{input}\nCandidate: {candidate}"` fixes the tokenization context; any leading
or trailing whitespace in the candidate value is part of the semantic string
and is tokenized as supplied. This is an intentional decision: the caller or a
future candidate-generation layer owns normalization, not the scorer.
- **Alternative considered**: strip the candidate value before tokenization.
  Rejected: it would silently contradict the `Candidate` no-normalization
  contract and hide candidate-value differences from the caller.

### Decision: Causal logit alignment — `log_softmax(logits[position - 1])[token_id]`
For a causal LM, logits at position `i` predict token `i + 1`. Given prefix
tokens `p_0 ... p_m` and candidate tokens `c_0 ... c_{n-1}`, the backend
extracts:
```
candidate_logprob_0 = log_softmax(logits[m])[c_0_id]
candidate_logprob_1 = log_softmax(logits[m+1])[c_1_id]
...
candidate_logprob_{n-1} = log_softmax(logits[m+n-2])[c_{n-1}_id]
```
The score is the arithmetic mean of these log-probabilities. The alignment is
documented in `logit_scorer.py`. This is the most important implementation
detail: reading logits at candidate token positions as if they predict the
same-position token would be a fundamental causal-LM error.
- **Alternative considered**: read logits at candidate token positions directly.
  Rejected: for a causal LM, `logits[i]` predicts token `i + 1`, not token `i`;
  same-position reading would extract the wrong distribution.

### Decision: Logit-to-score algorithm is mean candidate-token log-probability
The score is `mean(candidate_logprob_0, ..., candidate_logprob_{n-1})` using the
causal alignment above. Mean (not sum) normalizes by candidate token count,
avoiding systematic length bias against longer candidates. The score is a ranking
signal; no softmax-to-probability claim is made. The strategy is documented in
`logit_scorer.py`.
- **Alternative considered**: sum of token log-probabilities. Rejected: sums
  systematically penalize longer candidates (more negative terms), creating
  length bias.
- **Alternative considered**: single verbalizer-token scoring. Rejected: it
  restricts candidates to single tokens and breaks on multi-word candidates
  ("financial regulation").
- **Alternative considered**: length-normalized log-probability with a
  temperature/softmax over candidates. Rejected: introduces calibration
  semantics explicitly deferred by the specs.

### Decision: Application-facing error in application layer; infrastructure error is internal
`application/errors/scoring.py` defines `CandidateScoringError(Exception)` — the
only error type the application sees from scoring.
`infrastructure/llm/scoring/errors.py` defines `LLMScoringError(Exception)` —
infrastructure-internal, for wrapping library failures and invalid scores.
`LLMLogitCandidateScorer.score()` catches `LLMScoringError` and raw library
exceptions (torch, transformers, provider SDK, tokenizer) and raises
`CandidateScoringError` with the original as `__cause__`. The logit scorer
raises `LLMScoringError` on NaN/inf. This preserves dependency direction:
infrastructure imports `application.errors` (allowed by
`infrastructure-isolated`), but the application never imports infrastructure
errors. The application can catch `CandidateScoringError` without knowing about
`LLMScoringError` or any library.
- **Alternative considered**: define `ScoringError` in infrastructure and call
  it "application-facing". Rejected: the application would need to import an
  infrastructure module to catch the error, inverting the dependency direction.
- **Alternative considered**: a broad error taxonomy. Rejected: the specs
  require the smallest necessary abstraction; two error types (one per layer)
  suffice.

### Decision: Async contract requires no event-loop blocking; concrete mechanism is infrastructure-specific
`CandidateScorer.score()` is async (port contract). The implementation SHALL NOT
block the event loop. The concrete mechanism — `asyncio.to_thread`, a dedicated
executor, an async-native model, or any other strategy — is an
infrastructure-internal choice, NOT prescribed by the port or the design. The
initial implementation MAY use `asyncio.to_thread` for a synchronous local
model, but this is an implementation detail, not an architectural decision. The
model seam is sync (`CausalLanguageModel.forward`); the scorer adapts it to
async. The model adapter puts the model in `eval()` and uses `torch.no_grad()`
(or the framework equivalent). The scorer does not own the model lifecycle.
- **Alternative considered**: fix `asyncio.to_thread` as the mandated mechanism.
  Rejected: GPU/CUDA concurrency has its own constraints; mandating a specific
  thread-offload strategy overcommits the architecture and may not suit all
  model stacks or deployment scenarios.
- **Alternative considered**: make the model seam async. Rejected: local model
  `forward` is sync; an async seam would add ceremony without benefit and
  complicate test doubles.

### Decision: Initial implementation evaluates candidates sequentially; multi-candidate API is an application contract, not a model batch
The scorer loops over candidates, building a prompt and running the model per
candidate. The multi-candidate `score(text, candidates)` API means one
application invocation evaluates a set of candidates — it does NOT require a
single batched model forward. Sequential evaluation satisfies the contract. The
seams are shaped so a batch method can be added later without changing the port.
No prefix reuse or KV-cache in this change.
- **Alternative considered**: batch all candidate prompts in one model call now.
  Rejected: batching mechanics depend on the concrete model stack and add
  complexity; the specs require only that the design not prevent batching, which
  the seam structure satisfies.

### Decision: Prompt builder owns the scoring prompt format
`ScoringPromptBuilder.build(text, candidate) -> str` produces the model-facing
prompt. The exact wording is an infrastructure implementation detail; the domain
and application contain no prompt text. The builder is a seam so prompt
experimentation does not require touching the scorer algorithm.
- **Alternative considered**: inline prompt formatting in the scorer. Rejected:
  a separate builder keeps the scorer focused on the algorithm and makes prompt
  variations testable in isolation.

## Risks / Trade-offs

- **[Mean log-prob may not be optimal]** Mean token log-probability is a
  reasonable length-normalized ranking signal but is not empirically validated
  for this model/task yet.
  → Mitigation: the score is an opaque ranking signal, not a probability;
  calibration/normalization is deferred per the specs. Empirical evaluation is
  an explicit follow-up.
- **[Sequential evaluation is slow]** Per-candidate model calls are not
  throughput-optimal.
  → Mitigation: the seam structure permits batching and prefix reuse later
  without port changes; performance optimization is explicitly behind
  `CandidateScorer`.
- **[Tie-breaking is arbitrary]** First-in-order tie-breaking is deterministic
  but not semantically motivated.
  → Mitigation: the behavior is explicit and documented; a domain-informed
  tie policy can replace it without changing the port or use case.
- **[Tokenization boundary may shift with prompt format]** The
  prefix/candidate split depends on the prompt format; changing the prompt
  wording could change where the boundary falls.
  → Mitigation: the prompt builder and tokenizer adapter are co-located in
  `infrastructure/llm/scoring/`; boundary changes are tested with deterministic
  tokenizer doubles.
- **[Model lifecycle assumption]** The design assumes the model is constructed
  externally and passed to the adapter. If the model must be loaded lazily, the
  adapter boundary may need adjustment.
  → Mitigation: the adapter is infrastructure-internal; lifecycle changes do
  not affect the port or application.

## Migration Plan

This change is additive: new domain, application, and infrastructure modules are
introduced; no existing production code is modified and no existing caller is
wired to the new flow. Rollback is removing the new modules and tests. The
generative path continues to operate unchanged. Subsequent migration (out of
scope) happens caller-by-caller: a caller using `LLMClient.classify` is moved to
`ClassifyInput` once candidate construction for that caller is defined. The
generative path is removed only after all required callers have migrated.

## Open Questions

- **Device placement**: should the model adapter default to CUDA when available,
  or require an explicit device argument? Deferrable — the adapter is
  infrastructure-internal; the default can be settled in implementation without
  affecting the port or specs.
- **Real-model integration test fixture**: which model/tokenizer pair to use for
  the opt-in real-model test. Deferrable — the opt-in test is isolated from the
  default suite; the fixture can be chosen in implementation.
- **Async execution mechanism**: `asyncio.to_thread` vs. a dedicated executor vs.
  an async-native model. Deferrable — the specs prescribe only that the event
  loop is not blocked; the concrete mechanism is an implementation choice.
