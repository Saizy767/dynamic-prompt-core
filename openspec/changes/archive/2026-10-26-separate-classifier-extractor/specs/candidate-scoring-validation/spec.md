# Spec Delta

## MODIFIED Requirements

### Requirement: Candidate scoring is the classification evidence source
The test suite SHALL verify that classification receives a `Classification`
from the `Classifier` port rather than from generated classification output.
Application classification tests SHALL depend on the `Classifier` port and
SHALL NOT import infrastructure classifier implementations, model adapters,
tokenizer adapters, or logit adapters.

#### Scenario: Application test uses the port
- **WHEN** an application classification test is executed
- **THEN** it depends on `Classifier` and does not import concrete classifier
  implementations, `transformers`, PyTorch model classes, tokenizer adapters, or
  logit adapters

#### Scenario: End-to-end classification path uses classifier
- **WHEN** the classification path is exercised with a deterministic fake classifier
- **THEN** the selected candidate is derived from the classifier's `Classification`
  rather than from generated text, and no generated JSON response is involved

### Requirement: Policy owns candidate selection
The test suite SHALL verify that the `Classifier` produces the final
classification decision (selected candidate and judgments). Application
classification tests SHALL use a real or fake `Classifier` that preserves the
existing decision logic (argmax with input-order tie-breaking) unless the
behavior under test specifically requires a predetermined result.

#### Scenario: Highest score selection
- **WHEN** judgments with distinct scores are classified
- **THEN** the candidate with the highest score is selected

#### Scenario: Tie breaks by original order
- **WHEN** two candidates have equal scores and the first candidate appears
  earlier in the input
- **THEN** the first candidate is selected

#### Scenario: Candidate identity preserved
- **WHEN** the classifier returns a `Classification`
- **THEN** the selected candidate is one of the input candidates and candidate
  identity is preserved through the classification result

### Requirement: Candidate ordering is preserved
The test suite SHALL verify that the ordering supplied to the `Classifier` is
preserved through the returned `Classification.judgments`. The correspondence
`candidates[i] == judgments[i].candidate` SHALL hold for all candidates.

#### Scenario: Judgment order matches candidate order
- **WHEN** the classifier evaluates candidates `[A, B, C]`
- **THEN** `judgments[0].candidate == A`, `judgments[1].candidate == B`, and
  `judgments[2].candidate == C`

#### Scenario: Tie resolution depends on input order
- **WHEN** candidates `A` and `B` both receive score `0.5` and candidate `C`
  receives `0.2`
- **THEN** the classification selects `A` because it appears first in the input
  ordering

### Requirement: Candidate configuration is cycle-wide
The test suite SHALL verify that the candidate set has one source of truth
supplied through `RunCycleDeps`. The `Classifier` SHALL receive candidates from
`RunCycleDeps` at construction time. No `BaselineRunner` construction inside
`run_cycle` SHALL supply a `candidates=` argument other than `deps.candidates`.

#### Scenario: Classifier candidates come from deps
- **WHEN** an optimization step constructs a `Classifier` or `BaselineRunner`
- **THEN** the candidates are derived from `deps.candidates` and not from a
  step-local construction

#### Scenario: No step-local candidate override
- **WHEN** `BaselineRunner(...)` construction calls in `run_cycle` are inspected
- **THEN** no call supplies a step-local candidate list such as
  `[Candidate("0"), Candidate("1")]` as the `candidates=` argument

### Requirement: Scoring implementation is hidden behind the port
The test suite SHALL verify that application code does not import the concrete
classifier infrastructure. Layer-direction boundaries (application →
infrastructure) are enforced by `lint-imports`. Symbol-level AST checks SHALL
be added only where an existing import-linter contract permits an import path
that is still architecturally forbidden.

#### Scenario: Application does not import infrastructure classifier
- **WHEN** `lint-imports` contracts are evaluated
- **THEN** the `application-isolated` and `application-ports-isolated` contracts
  forbid application from importing infrastructure classifier implementations

#### Scenario: Symbol-level check only where needed
- **WHEN** an import-linter contract permits an import path that is still
  architecturally forbidden
- **THEN** a symbol-level AST check is added for that specific case

### Requirement: One scorer instance is reused within a cycle
The test suite SHALL verify that all `BaselineRunner` instances created during
one optimization cycle receive the same `Classifier` and `Extractor` object
instances from `RunCycleDeps`. The optimization loop SHALL NOT construct a new
classifier or extractor per example, round, step, or runner.

#### Scenario: All runners share classifier identity
- **WHEN** a single optimization round constructs runners for `run_active_on_dev`,
  `run_new_on_dev`, and `run_holdout`
- **THEN** all three runners have `classifier is deps.classifier` and
  `extractor is deps.extractor`

#### Scenario: Candidate ordering propagated from deps
- **WHEN** runners are constructed from `RunCycleDeps`
- **THEN** each runner receives `list(deps.candidates)` preserving the
  tuple's order

### Requirement: All BaselineRunner construction paths receive scoring dependencies
The test suite SHALL verify that every `BaselineRunner` construction inside
`run_cycle` receives the `Classifier` and `Extractor` from `RunCycleDeps`. All
three construction sites — `run_active_on_dev`, `run_new_on_dev`, and
`run_holdout` — SHALL be covered.

#### Scenario: run_active_on_dev receives dependencies
- **WHEN** `run_active_on_dev` constructs a `BaselineRunner`
- **THEN** the runner receives `deps.classifier`, `deps.extractor`, and
  `list(deps.candidates)`

#### Scenario: run_new_on_dev receives dependencies
- **WHEN** `run_new_on_dev` constructs a `BaselineRunner`
- **THEN** the runner receives `deps.classifier`, `deps.extractor`, and
  `list(deps.candidates)`

#### Scenario: run_holdout receives dependencies
- **WHEN** `run_holdout` constructs a `BaselineRunner`
- **THEN** the runner receives `deps.classifier`, `deps.extractor`, and
  `list(deps.candidates)`

### Requirement: Concrete scorer construction has one production source
The test suite SHALL verify that concrete classifier construction has one
production source. Both composition paths — the baseline runner CLI and the
optimization-cycle composition root — SHALL route through the shared
`build_classifier` factory rather than duplicating classifier construction.
The semantic invariant is: one production construction path for the
`Classifier` implementation.

#### Scenario: Both paths route through the shared factory
- **WHEN** the baseline runner CLI and the cycle composition root are inspected
- **THEN** both reference `build_classifier` and neither constructs the concrete
  classifier implementation directly

#### Scenario: Classifier constructed once per process
- **WHEN** the composition root assembles dependencies for a classification
  command
- **THEN** the classifier is constructed once and reused across all optimization
  rounds

### Requirement: No partial scoring result is accepted
The test suite SHALL verify that a failed scoring batch does not produce a
partially successful classification. If a batched model call fails, the
classification error SHALL be propagated and no partial results SHALL be
returned as successful output.

#### Scenario: Batch failure propagates error
- **WHEN** a batched model call fails during classification
- **THEN** the classification error is propagated and no partial results are
  returned

#### Scenario: One judgment per candidate on success
- **WHEN** the classifier evaluates `n` candidates successfully
- **THEN** exactly `n` judgments are returned in the same order as the input
  candidates

### Requirement: Scoring failures cross the application error boundary
The test suite SHALL verify that failures at tokenization, model inference, and
logit scoring boundaries propagate as a classification error and that
`BaselineRunner` maps classification failures to `classify_status = "failed"`
without producing fake classifications. Optimization SHALL NOT silently fall
back to another classification mechanism.

#### Scenario: Scoring failure maps to failed status
- **WHEN** `BaselineRunner` encounters a classification error during
  classification
- **THEN** `classify_status` is `"failed"`, `predicted_decision` is `None`, and
  no fake classification is produced

#### Scenario: No silent optimization fallback
- **WHEN** a classification failure occurs during optimization
- **THEN** the optimization does not silently fall back to another
  classification mechanism

### Requirement: Non-blocking async scoring boundary
The test suite SHALL verify that awaiting classification through the real
`Classifier` implementation does not synchronously block the event loop on
model inference. The test SHALL use deterministic fake tokenizer/model adapters
where the fake model deliberately blocks on a synchronization primitive while a
concurrent `asyncio` task demonstrates event-loop progress. The specific
executor mechanism SHALL remain an infrastructure implementation detail.

#### Scenario: Scoring does not block the event loop
- **WHEN** the real `Classifier` implementation is awaited with a fake model
  adapter that blocks on a synchronization primitive
- **THEN** a concurrent `asyncio` task makes progress while classification is
  awaited, proving blocking inference is offloaded off the event loop
