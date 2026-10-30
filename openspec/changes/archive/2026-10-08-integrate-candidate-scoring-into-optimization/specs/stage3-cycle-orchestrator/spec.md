# Spec Delta

## ADDED Requirements

### Requirement: Optimization classification path consistency
The optimization cycle SHALL classify every evaluation example through the
candidate-scoring flow: `CandidateScorer.score` → `ClassificationPolicy.classify`.
The optimization cycle SHALL use the same scorer implementation, policy
implementation, and candidate configuration as production classification.
Within a single process, all `BaselineRunner` instances created by one
`RunCycleDeps` instance SHALL reuse the same scorer and policy instances.
The optimization cycle SHALL NOT invoke generative classification, parse
generated classification JSON, or interpret model-generated confidence. The
optimization cycle SHALL NOT construct a separate scorer or policy for
evaluation.

#### Scenario: Optimization uses candidate scoring
- **WHEN** the optimization cycle evaluates an example
- **THEN** classification is performed via `CandidateScorer.score` followed by `ClassificationPolicy.classify`

#### Scenario: Same scorer and policy implementation as production
- **WHEN** the optimization cycle is wired by the composition root
- **THEN** the `CandidateScorer` and `ClassificationPolicy` are the same implementations and configurations used by production classification

#### Scenario: Same scorer instance within one cycle
- **WHEN** a single optimization cycle creates multiple `BaselineRunner` instances
- **THEN** all runners receive the same `candidate_scorer` and `classification_policy` object instances from `RunCycleDeps`

#### Scenario: No generative classification in optimization
- **WHEN** the optimization cycle code is inspected
- **THEN** it contains no generative classification call, no generated-JSON parsing, and no model-generated confidence interpretation

#### Scenario: No separate evaluation scorer
- **WHEN** the optimization cycle evaluates candidates
- **THEN** it does not construct a scorer or policy distinct from the one supplied via its dependencies

### Requirement: Candidate set supplied to optimization
The optimization cycle SHALL receive the classification candidate set via
its dependency object as a `tuple[Candidate, ...]`. The candidate set SHALL
be cycle-wide classification candidates with the same lifetime as
`RunCycleDeps` — they are not per-example or per-task data. The candidate
set SHALL be supplied once at composition time and reused across all rounds
and evaluations. The optimization cycle SHALL NOT infer, generate, or modify the candidate
set. The optimization cycle has exactly one candidate-set source: the
configured classification candidates. No step, runner, or evaluation
context SHALL construct or override the candidate set. Candidate ordering
SHALL be preserved across all evaluations because
`ArgmaxClassificationPolicy` resolves ties by original order. The `tuple`
type makes this ordering and immutability explicit at the type level.

#### Scenario: Candidates received via dependencies
- **WHEN** the optimization cycle is constructed
- **THEN** the candidate set is supplied through the dependency object as a `tuple[Candidate, ...]`

#### Scenario: Candidates reused across rounds
- **WHEN** the optimization cycle runs multiple rounds
- **THEN** the same candidate tuple is used in every round without reconstruction

#### Scenario: Candidate ordering preserved
- **WHEN** candidates are supplied in a specific order
- **THEN** that order is preserved in every evaluation invocation

#### Scenario: Candidates not inferred at runtime
- **WHEN** the optimization cycle evaluates an example
- **THEN** it uses the supplied candidate set and does not infer or generate candidates

#### Scenario: Candidate set is immutable
- **WHEN** the dependency object is constructed
- **THEN** the candidate set is a `tuple` and cannot be mutated after construction

#### Scenario: Exactly one candidate-set source
- **WHEN** the optimization cycle is inspected
- **THEN** no step, runner, or evaluation context constructs or overrides the candidate set; the only source is the configured classification candidates wired through the dependency object

## MODIFIED Requirements

### Requirement: Dependency object
`run_cycle` SHALL receive all dependencies through a typed dependency object
(`RunCycleDeps`). The dependency object SHALL contain outbound ports:
`llm_client`, `prompt_repository`, `run_repository`, `dataset_repository`,
`embedding_client`, `normalizer`, `candidate_scorer`,
`classification_policy`, `candidates`, and optionally `teacher_llm_client`.
`candidate_scorer` SHALL be a `CandidateScorer` port instance.
`classification_policy` SHALL be a `ClassificationPolicy` instance.
`candidates` SHALL be a `tuple[Candidate, ...]`. The dependency object
SHALL NOT contain concrete infrastructure implementations. The use case
SHALL NOT create dependencies itself or use a service locator.

#### Scenario: Typed dependency object
- **WHEN** the use case is called
- **THEN** it receives a dependency object containing typed ports including `candidate_scorer`, `classification_policy`, and `candidates`

#### Scenario: No concrete dependencies
- **WHEN** the use case executes
- **THEN** it does not import from `infrastructure`

#### Scenario: No service locator
- **WHEN** the use case receives dependencies
- **THEN** they are passed explicitly, not via a global registry

#### Scenario: Scorer is a CandidateScorer port
- **WHEN** the dependency object is inspected
- **THEN** `candidate_scorer` is a `CandidateScorer` port instance, not a concrete infrastructure class

#### Scenario: Policy is a ClassificationPolicy
- **WHEN** the dependency object is inspected
- **THEN** `classification_policy` is a `ClassificationPolicy` instance

#### Scenario: Candidates are an immutable tuple of Candidate
- **WHEN** the dependency object is inspected
- **THEN** `candidates` is a `tuple[Candidate, ...]`

### Requirement: Composition root wiring
`interfaces/cli/main.py` SHALL assemble all dependencies for `run_cycle`:
concrete port implementations from `infrastructure`, including the LLM client,
prompt repository, run repository, dataset repository, embedding client,
normalizer, teacher client (when enabled), `CandidateScorer`,
`ClassificationPolicy`, and the candidate tuple. The `CandidateScorer` SHALL
be constructed via a shared `build_candidate_scorer` helper that is also
used by the baseline runner CLI, so the two composition roots share one
wiring path. The `ClassificationPolicy` SHALL be `ArgmaxClassificationPolicy`.
The candidate tuple SHALL be constructed from the configured classification
candidates. The scorer SHALL be constructed only in the composition path reached by
CLI commands whose dependency graph requires the classification scorer;
CLI commands that do not require classification SHALL NOT load the model.
The composition root SHALL NOT contain business logic. No other module
SHALL create these dependencies.

#### Scenario: Dependencies wired in CLI
- **WHEN** the CLI starts the cycle command
- **THEN** all dependencies including `candidate_scorer`, `classification_policy`, and `candidates` are assembled in `interfaces/cli/main.py`

#### Scenario: Use case receives dependencies
- **WHEN** `run_cycle` is called
- **THEN** it receives a dependency object assembled in the composition root

#### Scenario: Scorer constructed via shared factory
- **WHEN** the composition root constructs the `CandidateScorer`
- **THEN** it calls a shared `build_candidate_scorer` helper that is also used by the baseline runner CLI

#### Scenario: Policy wired as ArgmaxClassificationPolicy
- **WHEN** the composition root constructs the `ClassificationPolicy`
- **THEN** it is `ArgmaxClassificationPolicy`

#### Scenario: Scorer not loaded for unrelated commands
- **WHEN** a CLI command that does not require classification is invoked
- **THEN** the scorer and model are not constructed

### Requirement: Testability
`run_cycle` SHALL be testable with a fake `CandidateScorer` and the real
`ArgmaxClassificationPolicy`. Tests SHALL NOT require a running inference
server or a loaded language model. Tests SHALL verify the step sequence,
counter behavior, rollback handling, report and summary formation,
resumability from a dumped state, and that the scorer and policy are
threaded to the baseline runner. The real `ArgmaxClassificationPolicy`
SHALL be used in tests unless a specific test requires a predetermined
`Classification` result, so that actual selection behavior is exercised.

#### Scenario: Unit test with mocks
- **WHEN** `run_cycle` is tested
- **THEN** ports including `CandidateScorer` are replaced by mock or fake objects and `ArgmaxClassificationPolicy` is used as-is

#### Scenario: Unit test with fake scorer and real policy
- **WHEN** `run_cycle` is tested
- **THEN** the `CandidateScorer` is replaced by a fake and `ArgmaxClassificationPolicy` is used as-is

#### Scenario: No server required
- **WHEN** tests are run
- **THEN** no inference server or language model is required

#### Scenario: Scorer and policy threaded to runner
- **WHEN** the optimization round constructs a `BaselineRunner`
- **THEN** the scorer and policy from the dependency object are passed to the runner
