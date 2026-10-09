# Spec

## Purpose

Define the executable test and architectural-validation requirements that enforce
the candidate-scoring migration's cross-layer invariants, ensuring the
generative classification architecture cannot silently reappear and the
dependency boundaries between domain, application, and infrastructure remain
intact.

## Requirements

### Requirement: Candidate scoring is the classification evidence source
The test suite SHALL verify that classification receives `Judgment` objects from
`CandidateScorer` rather than from generated classification output. Application
classification tests SHALL depend on the `CandidateScorer` port and SHALL NOT
import infrastructure scoring implementations, model adapters, tokenizer
adapters, or logit adapters.

#### Scenario: Application test uses the port
- **WHEN** an application classification test is executed
- **THEN** it depends on `CandidateScorer` and does not import
  `LLMLogitCandidateScorer`, `transformers`, PyTorch model classes, tokenizer
  adapters, or logit adapters

#### Scenario: End-to-end classification path uses scoring
- **WHEN** the classification path is exercised with a deterministic fake scorer
- **THEN** the selected candidate is derived from candidate scores rather than
  from generated text, and no generated JSON response is involved

### Requirement: Policy owns candidate selection
The test suite SHALL verify that `ClassificationPolicy` selects the final
classification and that `CandidateScorer` does not. Application classification
tests SHALL use the real `ArgmaxClassificationPolicy` unless the behavior under
test specifically requires a predetermined result.

#### Scenario: Highest score selection
- **WHEN** judgments with distinct scores are classified
- **THEN** the candidate with the highest score is selected

#### Scenario: Tie breaks by original order
- **WHEN** two candidates have equal scores and the first candidate appears
  earlier in the input
- **THEN** the first candidate is selected

#### Scenario: Empty judgments rejected
- **WHEN** an empty judgment list is classified
- **THEN** a `ValueError` is raised

#### Scenario: Candidate identity preserved
- **WHEN** judgments are classified
- **THEN** the selected candidate is one of the input candidates and candidate
  identity is preserved through the classification result

### Requirement: Candidate ordering is preserved
The test suite SHALL verify that the ordering supplied to `CandidateScorer` is
preserved through judgments and policy evaluation. The correspondence
`candidates[i] == judgments[i].candidate` SHALL hold for all candidates.

#### Scenario: Judgment order matches candidate order
- **WHEN** a scorer evaluates candidates `[A, B, C]`
- **THEN** `judgments[0].candidate == A`, `judgments[1].candidate == B`, and
  `judgments[2].candidate == C`

#### Scenario: Tie resolution depends on input order
- **WHEN** candidates `A` and `B` both receive score `0.5` and candidate `C`
  receives `0.2`
- **THEN** the classification selects `A` because it appears first in the input
  ordering

### Requirement: Optimization does not consume raw scores as metrics
The test suite SHALL verify that `Judgment.score` is a ranking signal and is not
used directly as an optimization metric. Optimization metrics SHALL be computed
from the predicted decision and ground truth. The `BaselineRunner` unit test
SHALL verify that `predicted_decision` is derived from
`Classification.selected.value`. A separate integration test SHALL verify that
changing `Judgment.score` without changing the selected candidate does not
change the optimization metric.

#### Scenario: Predicted decision derives from classification
- **WHEN** the baseline runner processes an example
- **THEN** `predicted_decision` is derived from `Classification.selected.value`
  and not from raw model logits, `Judgment.score`, generated text, a confidence
  field, or a legacy classification result

#### Scenario: Score change without selection change is metric-invariant
- **WHEN** candidate A's score changes from `0.1` to `0.2` and candidate B's
  score changes from `0.8` to `0.9` while the selected candidate remains B
- **THEN** the predicted decision remains B and the classification metric is
  unchanged

### Requirement: Predicted decision conversion semantics are preserved
The test suite SHALL explicitly verify the `predicted_decision` conversion
behavior: `Classification.selected.value` is cast to `int` to produce
`predicted_decision`, with `None` fallback on `ValueError` or `TypeError`.

#### Scenario: Integer candidate value converts successfully
- **WHEN** the selected candidate is `Candidate("1")`
- **THEN** `predicted_decision == 1`

#### Scenario: Non-integer candidate value falls back to None
- **WHEN** the selected candidate is a non-integer value such as `Candidate("yes")`
- **THEN** `predicted_decision == None`

### Requirement: Candidate configuration is cycle-wide
The test suite SHALL verify that the candidate set has one source of truth
supplied through `RunCycleDeps.candidates`. No `BaselineRunner` construction
inside `run_cycle` SHALL supply a `candidates=` argument other than
`deps.candidates`. The test SHALL target `BaselineRunner` construction calls
specifically, not ban all `Candidate` construction in `run_cycle`.

#### Scenario: BaselineRunner candidates come from deps
- **WHEN** an optimization step constructs a `BaselineRunner`
- **THEN** the `candidates=` argument is derived from `deps.candidates` and not
  from a step-local construction

#### Scenario: No step-local candidate override
- **WHEN** `BaselineRunner(...)` construction calls in `run_cycle` are inspected
- **THEN** no call supplies a step-local candidate list such as
  `[Candidate("0"), Candidate("1")]` as the `candidates=` argument

### Requirement: Scoring implementation is hidden behind the port
The test suite SHALL verify that application code does not import the concrete
scoring infrastructure. Layer-direction boundaries (application → infrastructure)
are enforced by `lint-imports`. Symbol-level AST checks SHALL be added only
where an existing import-linter contract permits an import path that is still
architecturally forbidden.

#### Scenario: Application does not import infrastructure scorer
- **WHEN** `lint-imports` contracts are evaluated
- **THEN** the `application-isolated` and `application-ports-isolated` contracts
  forbid application from importing infrastructure scoring implementations

#### Scenario: Symbol-level check only where needed
- **WHEN** an import-linter contract permits an import path that is still
  architecturally forbidden
- **THEN** a symbol-level AST check is added for that specific case

### Requirement: One scorer instance is reused within a cycle
The test suite SHALL verify that all `BaselineRunner` instances created during
one optimization cycle receive the same scorer and policy object instances from
`RunCycleDeps`. The optimization loop SHALL NOT construct a new scorer per
example, round, step, or runner.

#### Scenario: All runners share scorer identity
- **WHEN** a single optimization round constructs runners for `run_active_on_dev`,
  `run_new_on_dev`, and `run_holdout`
- **THEN** all three runners have `scorer is deps.candidate_scorer` and
  `policy is deps.classification_policy`

#### Scenario: Candidate ordering propagated from deps
- **WHEN** runners are constructed from `RunCycleDeps`
- **THEN** each runner receives `list(deps.candidates)` preserving the
  tuple's order

### Requirement: All BaselineRunner construction paths receive scoring dependencies
The test suite SHALL verify that every `BaselineRunner` construction inside
`run_cycle` receives the scorer, policy, and candidates from `RunCycleDeps`.
All three construction sites — `run_active_on_dev`, `run_new_on_dev`, and
`run_holdout` — SHALL be covered.

#### Scenario: run_active_on_dev receives dependencies
- **WHEN** `run_active_on_dev` constructs a `BaselineRunner`
- **THEN** the runner receives `deps.candidate_scorer`,
  `deps.classification_policy`, and `list(deps.candidates)`

#### Scenario: run_new_on_dev receives dependencies
- **WHEN** `run_new_on_dev` constructs a `BaselineRunner`
- **THEN** the runner receives `deps.candidate_scorer`,
  `deps.classification_policy`, and `list(deps.candidates)`

#### Scenario: run_holdout receives dependencies
- **WHEN** `run_holdout` constructs a `BaselineRunner`
- **THEN** the runner receives `deps.candidate_scorer`,
  `deps.classification_policy`, and `list(deps.candidates)`

### Requirement: Concrete scorer construction has one production source
The test suite SHALL verify that concrete scorer construction has one production
source. Both composition paths — the baseline runner CLI and the
optimization-cycle composition root — SHALL route through the shared
`build_candidate_scorer` factory rather than duplicating scorer construction.
The semantic invariant is: one production construction path for
`LLMLogitCandidateScorer`. The current validation asserts both composition roots
reference `build_candidate_scorer` and neither constructs
`LLMLogitCandidateScorer` directly.

#### Scenario: Both paths route through the shared factory
- **WHEN** the baseline runner CLI and the cycle composition root are inspected
- **THEN** both reference `build_candidate_scorer` and neither constructs
  `LLMLogitCandidateScorer` directly

#### Scenario: Scorer constructed once per process
- **WHEN** the composition root assembles dependencies for a classification
  command
- **THEN** the scorer is constructed once and reused across all optimization
  rounds

### Requirement: No generative classification fallback exists
The test suite SHALL verify that production code does not reference the removed
generative classification symbols `ClassificationResult`, `classify_detailed`,
and `classify_many`. The old classification path SHALL NOT be selectable at
runtime.

#### Scenario: Removed symbols absent from production code
- **WHEN** production source files are scanned for `ClassificationResult`,
  `classify_detailed`, and `classify_many`
- **THEN** no matches are found

#### Scenario: No runtime fallback to generative classification
- **WHEN** production source files are scanned for try/except fallback patterns
  referencing generative classification
- **THEN** no fallback to generative classification exists

### Requirement: No legacy structured-generation classification dependency
The test suite SHALL verify that the classification path does not depend on the
old structured-generation mechanism. The test SHALL target concrete obsolete
runtime artifacts — specific symbols (`ClassificationResult`, `classify_detailed`,
`classify_many`), specific generated-output model types, specific classification
JSON parsers or helpers — rather than searching for vague textual concepts. The
rule is: ban obsolete runtime artifacts, not vocabulary. General LLM generation
unrelated to classification MAY remain if still used elsewhere.

#### Scenario: Concrete legacy symbols absent
- **WHEN** production source files are scanned for concrete obsolete symbols
  and generated-output types specific to generative classification
- **THEN** no matches are found

#### Scenario: General LLM generation may remain elsewhere
- **WHEN** non-classification LLM generation infrastructure is inspected
- **THEN** it MAY remain if still used outside the classification path

### Requirement: No partial scoring result is accepted
The test suite SHALL verify that a failed scoring batch does not produce a
partially successful classification. If a batched model call fails,
`CandidateScoringError` SHALL be propagated and no partial judgments SHALL be
returned as successful output.

#### Scenario: Batch failure propagates error
- **WHEN** a batched model call fails during scoring
- **THEN** `CandidateScoringError` is propagated and no partial judgments are
  returned

#### Scenario: One judgment per candidate on success
- **WHEN** a scorer evaluates `n` candidates successfully
- **THEN** exactly `n` judgments are returned in the same order as the input
  candidates

### Requirement: Scoring failures cross the application error boundary
The test suite SHALL verify that failures at tokenization, model inference, and
logit scoring boundaries propagate as `CandidateScoringError` and that
`BaselineRunner` maps scoring failures to `classify_status = "failed"` without
producing fake classifications. Optimization SHALL NOT silently fall back to
another classification mechanism.

#### Scenario: Scoring failure maps to failed status
- **WHEN** `BaselineRunner` encounters a `CandidateScoringError` during
  classification
- **THEN** `classify_status` is `"failed"`, `predicted_decision` is `None`, and
  no fake classification is produced

#### Scenario: No silent optimization fallback
- **WHEN** a scoring failure occurs during optimization
- **THEN** the optimization does not silently fall back to another
  classification mechanism

### Requirement: LLM logit scorer validates causal token alignment
The test suite SHALL verify that the infrastructure scorer scores candidate
continuation positions in the actual model input using causal next-token
alignment. The token IDs used for scoring SHALL correspond to the candidate
continuation positions, not independently tokenized and concatenated IDs.

#### Scenario: Causal alignment scores correct positions
- **WHEN** a prefix `p0 p1 p2` is followed by candidate tokens `c0 c1 c2`
- **THEN** the scorer reads `logit[p2] → c0`, `logit[c0] → c1`, and
  `logit[c1] → c2`, not `logit[c0] → c0`

#### Scenario: Candidate token IDs match model input positions
- **WHEN** the scorer extracts candidate token IDs for scoring
- **THEN** those IDs exactly occupy the candidate continuation positions in the
  actual model input

#### Scenario: Multi-token mean log-probability
- **WHEN** a multi-token candidate is scored
- **THEN** the score is the arithmetic mean of per-token log-probabilities

### Requirement: Non-blocking async scoring boundary
The test suite SHALL verify that awaiting candidate scoring through the real
`LLMLogitCandidateScorer` does not synchronously block the event loop on model
inference. The test SHALL use deterministic fake tokenizer/model adapters where
the fake model deliberately blocks on a synchronization primitive while a
concurrent `asyncio` task demonstrates event-loop progress. The specific
executor mechanism SHALL remain an infrastructure implementation detail.

#### Scenario: Scoring does not block the event loop
- **WHEN** the real `LLMLogitCandidateScorer` is awaited with a fake model
  adapter that blocks on a synchronization primitive
- **THEN** a concurrent `asyncio` task makes progress while scoring is awaited,
  proving blocking inference is offloaded off the event loop

### Requirement: Unit tests run without a language model
The normal unit-test suite SHALL NOT require downloading a model, network access,
GPU availability, a specific model checkpoint, Hugging Face credentials, or model
weights. Model-backed tests, if retained, SHALL be explicitly separated from
unit/CI tests.

#### Scenario: Unit tests use deterministic fakes
- **WHEN** the unit test suite is executed
- **THEN** no model is loaded, no network access is required, and all tests use
  deterministic fake scorers and stub adapters

#### Scenario: Model-backed tests are opt-in
- **WHEN** a model-backed test exists
- **THEN** it is gated by an explicit opt-in mechanism and skipped by default in
  ordinary CI

### Requirement: Import boundaries are validated
The test suite SHALL verify the intended dependency direction: domain depends
only on the standard library; application depends on ports, not infrastructure
implementations; the composition root MAY import infrastructure. These
boundaries SHALL be enforced by `lint-imports` and supplemented by architectural
tests only where import-linter cannot reach.

#### Scenario: Domain does not import infrastructure
- **WHEN** domain modules are inspected
- **THEN** they do not import `transformers`, PyTorch, tokenizer adapters, model
  adapters, or LLM scoring infrastructure

#### Scenario: Composition root may import infrastructure
- **WHEN** the composition root is inspected
- **THEN** it MAY import infrastructure scorer implementations, tokenizer/model
  adapters, concrete policies, and third-party model libraries

### Requirement: No infrastructure state escapes into domain or application
The test suite SHALL explicitly verify that `Judgment`, `Classification`,
`ResultRow`, and optimization state carry only domain/application-safe
information. No `torch.Tensor`, `token_ids`, `raw_logits`, `attention_mask`, or
model output object SHALL appear as a field on these types. This is one of the
most important architectural outcomes of the migration.

#### Scenario: Judgment carries only domain-safe fields
- **WHEN** the `Judgment` type is inspected
- **THEN** its fields reference only `Candidate`, `float`, or other
  domain-safe types — no tensors, token IDs, logits, or model outputs

#### Scenario: Classification carries only domain-safe fields
- **WHEN** the `Classification` type is inspected
- **THEN** its fields reference only `Candidate`, `Judgment`, or collections
  thereof — no tensors, token IDs, logits, or model outputs

#### Scenario: ResultRow carries only application-safe fields
- **WHEN** the `ResultRow` type is inspected
- **THEN** its fields reference only `str`, `int`, `float`, `bool`, `None`,
  `dict`, or other application-safe types — no tensors, token IDs, logits, or
  model outputs

### Requirement: Existing optimization behavior is preserved
The test suite SHALL verify that existing optimization-cycle behavior —
iteration, candidate queue, active/new/holdout evaluation, rollback, reporting,
state dumping, resumability, stopping conditions, and version comparison —
remains unchanged by the migration. The new scorer dependency SHALL be
transparent to optimization logic.

#### Scenario: Optimization iteration unchanged
- **WHEN** the optimization cycle runs with the candidate-scoring wiring
- **THEN** iteration behavior matches the pre-migration behavior

#### Scenario: Rollback behavior unchanged
- **WHEN** a rollback is triggered during optimization
- **THEN** rollback behavior matches the pre-migration behavior

#### Scenario: Reporting and state dump unchanged
- **WHEN** the optimization cycle produces reports and state dumps
- **THEN** their format and content match the pre-migration behavior

### Requirement: Static validation passes
The migration SHALL be validated with the repository's existing static tooling:
`pytest`, `mypy`, `ruff check`, and `lint-imports`. All SHALL pass with no
regressions. Import-boundary failures and stale legacy references that behavioral
tests miss SHALL be caught by static validation.

#### Scenario: Pytest passes
- **WHEN** `pytest` is run
- **THEN** all tests pass, including architectural and contract tests

#### Scenario: Mypy passes
- **WHEN** `mypy` is run
- **THEN** type checking passes with no regressions

#### Scenario: Ruff passes
- **WHEN** `ruff check` is run
- **THEN** linting passes with no regressions

#### Scenario: Lint-imports passes
- **WHEN** `lint-imports` is run
- **THEN** all import-linter contracts pass, including domain isolation,
  application isolation, and port isolation
