# Tasks

## 1. Domain: Classification model

- [x] 1.1 Create `src/dynamic_prompt_core/domain/models/classification.py` with a
  `@dataclass(frozen=True)` `Classification` holding `selected: Candidate` and
  `judgments: tuple[Judgment, ...]`. The constructor SHALL accept an iterable
  and store `tuple(judgments)`. Enforce in `__post_init__` that `judgments` is
  non-empty and that `selected` is among the candidates in `judgments`. Verify
  by importing the module and constructing a valid instance in a REPL.
- [x] 1.2 Export `Classification` from `domain/models/__init__.py` `__all__` and
  verify `from dynamic_prompt_core.domain.models import Classification` succeeds.
- [x] 1.3 Add `tests/unit/candidate_scoring/test_classification.py` covering:
  valid creation; deep immutability (frozen fields AND `judgments` is a tuple
  that rejects `append`/`__setitem__` with `TypeError`); empty judgments
  rejected; selected not in judgments rejected; selected preserved; judgments
  preserved as tuple; free of model internals (only domain imports). Verify with
  `pytest tests/unit/candidate_scoring/test_classification.py`.

## 2. Application: ClassificationPolicy service and argmax policy

- [x] 2.1 Create
  `src/dynamic_prompt_core/application/services/classification_policy.py` with a
  `@runtime_checkable` `ClassificationPolicy(Protocol)` declaring
  `classify(self, judgments: list[Judgment]) -> Classification` and a concrete
  `ArgmaxClassificationPolicy` implementing it: selects the candidate with the
  maximum score; ties broken by first-in-input order; empty judgments raise
  `ValueError`; preserves input judgment order in `Classification.judgments` as
  a tuple. Import only domain types. Verify `typing.is_protocol(ClassificationPolicy)`
  is True and a stub instance satisfies `isinstance(..., ClassificationPolicy)`.
- [x] 2.2 Add `tests/unit/candidate_scoring/test_classification_policy.py`
  covering: protocol is a `typing.Protocol`; argmax selects highest score
  (A=0.2, B=0.8, C=0.5 → B); single candidate; tie returns first-in-order
  deterministically (A=0.8, B=0.8 → A); empty judgments raise `ValueError`;
  `Classification.judgments` preserves input order as a tuple; policy performs no
  inference (no infrastructure imports in the module). Verify with `pytest
  tests/unit/candidate_scoring/test_classification_policy.py`.

## 3. Application: ClassifyInput use case and scoring error

- [x] 3.1 Create `src/dynamic_prompt_core/application/errors/__init__.py` and
  `src/dynamic_prompt_core/application/errors/scoring.py` with
  `CandidateScoringError(Exception)` — the application-facing scoring error.
  Verify `from dynamic_prompt_core.application.errors import CandidateScoringError`
  succeeds and the module imports no infrastructure.
- [x] 3.2 Create
  `src/dynamic_prompt_core/application/use_cases/classify_input/__init__.py` and
  `deps.py` with a `@dataclass(frozen=True)` `ClassifyInputDeps(scorer:
  CandidateScorer, policy: ClassificationPolicy)`. Import the `CandidateScorer`
  outbound port and the `ClassificationPolicy` from `application/services`.
  Verify the module imports without error.
- [x] 3.3 Create
  `src/dynamic_prompt_core/application/use_cases/classify_input/classify_input.py`
  with `async def classify_input(deps: ClassifyInputDeps, text: str, candidates:
  list[Candidate]) -> Classification` that: rejects empty `candidates`; calls
  `validate_unique_candidates`; awaits `deps.scorer.score(text, candidates)`;
  calls `deps.policy.classify(judgments)`; returns the `Classification`. Verify
  the module imports only ports, the policy service protocol, and domain types.
- [x] 3.4 Add `tests/unit/candidate_scoring/test_classify_input.py` using
  deterministic stubs for `CandidateScorer` and `ClassificationPolicy`. Cover:
  happy path (score → classify); empty candidates rejected before scoring;
  duplicate candidates rejected before scoring; use case does not inspect model
  outputs (stubs receive only `text`/`candidates`/`judgments`); scorer-mechanism
  agnostic (swap stub scorer, orchestration identical). Verify with `pytest
  tests/unit/candidate_scoring/test_classify_input.py`.

## 4. Infrastructure: scoring seams and errors

- [x] 4.1 Create `src/dynamic_prompt_core/infrastructure/llm/scoring/__init__.py`
  and `errors.py` with `LLMScoringError(Exception)` — the infrastructure-internal
  scoring error. Verify the module imports and does not export anything the
  application should import.
- [x] 4.2 Create `infrastructure/llm/scoring/prompt_builder.py` with
  `ScoringPromptBuilder.build(text: str, candidate: Candidate) -> str` that
  renders `"{input}\nCandidate: {candidate}"`. The builder SHALL expose the
  prefix and candidate portions separately (e.g., `build_prefix(text)` and
  `build_candidate(candidate)`) so the scorer can determine the tokenization
  boundary without substring search. Verify the prefix ends with `"Candidate: "`
  and the candidate portion is `candidate.value` verbatim.
- [x] 4.3 Create `infrastructure/llm/scoring/tokenizer_adapter.py` with a
  `TokenizerAdapter(Protocol)` seam (`encode(text) -> list[int]`,
  `decode(token_ids) -> str`) and a `HuggingFaceTokenizerAdapter` concrete
  adapter wrapping `transformers.AutoTokenizer`. Verify the protocol is
  `typing.is_protocol` and the concrete adapter satisfies `isinstance`.
- [x] 4.4 Create `infrastructure/llm/scoring/model_adapter.py` with a sync
  `CausalLanguageModel(Protocol)` seam (`forward(input_ids: list[int]) ->
  SequenceLogits` where `SequenceLogits` exposes per-position logits) and a
  `TorchCausalLMAdapter` concrete adapter that calls the model under
  `torch.no_grad()` / `model.eval()`. Verify the protocol is `typing.is_protocol`.
- [x] 4.5 Add `tests/unit/candidate_scoring/test_scoring_seams.py` with
  deterministic test doubles for the tokenizer and model seams. Cover: prompt
  builder prefix/candidate split is correct; candidate value is not transformed
  (verbatim, no strip/lower); tokenizer encode/decode round-trips on the double;
  model double returns deterministic logits; `LLMScoringError` is
  infrastructure-internal. Verify with `pytest
  tests/unit/candidate_scoring/test_scoring_seams.py`.

## 5. Infrastructure: logit-to-score algorithm

- [x] 5.1 Create `infrastructure/llm/scoring/logit_scorer.py` with a
  `LogitScorer` that, given prefix token IDs, candidate token IDs, and the
  model's per-position logits, computes candidate-token log-probabilities using
  causal alignment: `candidate_logprob_k = log_softmax(logits[prefix_len + k -
  1])[candidate_token_ids[k]]`. The score is the arithmetic mean of these
  log-probabilities. Document the causal alignment and mean strategy in the
  module docstring. Verify on a hand-computed fixture: prefix `[p0, p1, p2]`,
  candidate `[c0, c1]` → `score = mean(log_softmax(logits[2])[c0],
  log_softmax(logits[3])[c1])`.
- [x] 5.2 Add NaN/inf guards in `LogitScorer`: if the computed score is NaN,
  +inf, or -inf, raise `LLMScoringError`. Verify with a fixture that produces
  NaN and confirms `LLMScoringError` is raised.
- [x] 5.3 Add `tests/unit/candidate_scoring/test_logit_scorer.py` with small
  deterministic logits fixtures. Cover: single-token candidate; multi-token
  candidate (mean, not sum); causal alignment (first candidate token read from
  last prefix position, not same position); semantic ordering (A=-0.2 > B=-1.4);
  NaN rejected; +inf rejected; -inf rejected; no silent clamping; longer
  candidate not systematically rewarded (compare 1-token mean -0.2 vs 3-token
  mean -0.2). Verify with `pytest
  tests/unit/candidate_scoring/test_logit_scorer.py`.

## 6. Infrastructure: LLMLogitCandidateScorer

- [x] 6.1 Create `infrastructure/llm/scoring/candidate_scorer.py` with
  `LLMLogitCandidateScorer` implementing `CandidateScorer`. Constructor takes
  `prompt_builder`, `tokenizer`, `model`, `logit_scorer`. `async def score(text,
  candidates)` loops over candidates: builds prefix and candidate prompt parts,
  encodes prefix and candidate separately (recording `prefix_token_count` and
  `candidate_token_ids`), composes the full sequence, runs the model via a
  non-blocking mechanism (e.g., `asyncio.to_thread` — implementation-specific),
  computes the score via `logit_scorer`, constructs `Judgment`. Returns judgments
  in input order. Verify a stub instance satisfies
  `isinstance(..., CandidateScorer)`.
- [x] 6.2 Add error translation: catch `LLMScoringError` and raw library
  exceptions (torch, transformers, tokenizer) and re-raise as
  `CandidateScoringError` (from `application.errors`) with the original as
  `__cause__`. Verify a model double that raises `RuntimeError` results in
  `CandidateScoringError`, not the raw `RuntimeError`, and that the application
  module does not import `LLMScoringError`.
- [x] 6.3 Add atomic failure semantics: if any candidate fails, the whole
  `score` call raises `CandidateScoringError`; no partial judgment list is
  returned. Verify a double that fails on the second of three candidates raises
  rather than returning two judgments.
- [x] 6.4 Add `tests/unit/candidate_scoring/test_llm_logit_candidate_scorer.py`
  with deterministic tokenizer and model doubles. Cover: one judgment per
  candidate; judgments in input order; every judgment references its candidate;
  multi-token candidate scored via causal-aligned mean log-prob; tokenization
  boundary determined by separate prefix/candidate encoding (not substring
  search); candidate value used verbatim (no normalization); multiple candidates
  in one call (sequential is valid); model failure translated to
  `CandidateScoringError`; partial failure raises (atomic); NaN score raises
  `CandidateScoringError` (via `LLMScoringError`); no logits/token IDs exposed in
  returned `Judgment` objects; event loop not blocked (async contract). Verify
  with `pytest tests/unit/candidate_scoring/test_llm_logit_candidate_scorer.py`.

## 7. Architecture validation and integration

- [x] 7.1 Run `openspec validate --changes separate-scoring-llm-logit-backend`
  and verify all spec deltas (`candidate-scoring-flow`, `llm-logit-scorer`,
  `candidate-scoring-contracts`) are valid: every requirement has at least one
  scenario, new capabilities have `## Purpose`, the modified
  `candidate-scoring-contracts` delta uses `## REMOVED Requirements` with Reason
  and Migration.
- [x] 7.2 Run `lint-imports` and verify all contracts pass, including
  `application-ports-isolated` (the `ClassificationPolicy` service is in
  `application/services`, not `application/ports`; it does not import
  infrastructure), `domain-isolated`, `domain-stdlib-only-allowlist`
  (`Classification` imports only stdlib/domain), `infrastructure-isolated`
  (infrastructure imports `application.ports` and `application.errors` but not
  `application.use_cases`), and `no-cyclic-imports`.
- [x] 7.3 Run `mypy` and `ruff check` and verify they pass on the new modules.
  Confirm no new type or lint regressions.
- [x] 7.4 Run the full test suite `pytest` and verify all tests pass, including
  existing classification tests (generative path unchanged) and the new
  `tests/unit/candidate_scoring/` tests. Inspect `git diff --stat` and confirm
  no existing production source file is modified — only new files are added.
  Confirm the new flow is not wired into any existing caller.
- [x] 7.5 (Optional) Add an opt-in real-model integration test under
  `tests/integration/` that loads a small model/tokenizer, scores a few
  candidates, and asserts semantic ordering. Gate it behind an environment
  variable or marker so the default suite does not require a model download.
  Verify the default `pytest` run skips it and it is isolated from the unit
  suite.
