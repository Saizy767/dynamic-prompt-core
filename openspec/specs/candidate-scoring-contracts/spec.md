# Spec

## Purpose

Define the domain model for candidate-based classification scoring (`Candidate`
and `Judgment`), the `CandidateScorer` application outbound port contract
through which the application requests candidate judgments without depending on
any LLM implementation, and the evaluation-vs-decision boundary that keeps the
scorer from becoming a classifier.

## Requirements

### Requirement: Candidate domain concept
`Candidate` SHALL be an immutable domain model representing one possible
classification hypothesis that can be evaluated against an input. `Candidate`
SHALL NOT contain LLM models, tokenizer information, token IDs, logits,
provider configuration, HTTP configuration, inference parameters, prompt
rendering implementations, or transport information. `Candidate` represents the
semantic option itself, not how it will be evaluated.

#### Scenario: Valid candidate creation
- **WHEN** a candidate is created with a valid semantic value
- **THEN** the candidate exists and exposes that value unchanged

#### Scenario: Candidate immutability
- **WHEN** a candidate has been created
- **THEN** its semantic value cannot be mutated after creation

#### Scenario: Candidate free of evaluation mechanism
- **WHEN** a candidate is inspected
- **THEN** it contains no model, tokenizer, token IDs, logits, provider, transport, or inference configuration

### Requirement: Candidate identity
A `Candidate` SHALL have a stable semantic value that identifies the option.
Identity SHALL be suitable for equality, deterministic comparison, and use as a
key when associating a judgment with a candidate. Identity SHALL NOT depend on
object identity or memory address. Two candidates with the same semantic value
SHALL be equal.

#### Scenario: Equality by semantic value
- **WHEN** two candidates are created with the same semantic value
- **THEN** they are equal

#### Scenario: Inequality by differing semantic value
- **WHEN** two candidates are created with different semantic values
- **THEN** they are not equal

#### Scenario: Hashable by semantic value
- **WHEN** a candidate is used as a dictionary key or set member
- **THEN** identity is determined by the semantic value, not memory address

#### Scenario: Invalid candidate rejection
- **WHEN** a candidate is created with an invalid semantic value
- **THEN** the creation is rejected according to domain validation rules

### Requirement: Candidate value is not auto-normalized
`Candidate` SHALL NOT automatically normalize its semantic value. The value
SHALL be stored verbatim. Operations such as `strip`, `lower`, `casefold`, or
whitespace collapsing SHALL NOT be applied automatically by `Candidate`.
Normalization is the responsibility of the calling code or a future
candidate-generation layer, not the domain model.

#### Scenario: Value stored verbatim
- **WHEN** a candidate is created with a value containing surrounding whitespace or mixed case
- **THEN** the stored value is identical to the supplied value

#### Scenario: No implicit stripping
- **WHEN** a candidate is created with `"  x  "`
- **THEN** its value is `"  x  "`, not `"x"`

#### Scenario: No implicit case folding
- **WHEN** a candidate is created with `"A"` and another with `"a"`
- **THEN** they are not equal

### Requirement: Judgment domain concept
`Judgment` SHALL be an immutable domain model representing the result of
evaluating one `Candidate` against a specific classification input. A `Judgment`
SHALL associate the evaluated candidate and the resulting score. `Judgment`
SHALL NOT expose logits, token IDs, tokenizer state, model objects,
provider-specific response objects, generated text, or transport metadata.

#### Scenario: Judgment associates candidate and score
- **WHEN** a judgment is created for a candidate with a valid score
- **THEN** the judgment exposes that candidate and that score

#### Scenario: Judgment immutability
- **WHEN** a judgment has been created
- **THEN** its candidate and score cannot be mutated after creation

#### Scenario: Repeated evaluation produces new judgment
- **WHEN** a candidate is evaluated again and receives another score
- **THEN** a new judgment is created rather than mutating the previous one

#### Scenario: Judgment free of model internals
- **WHEN** a judgment is inspected
- **THEN** it exposes no logits, token IDs, tokenizer state, model objects, provider response objects, generated text, or transport metadata

### Requirement: Score is an opaque ranking signal
The judgment score SHALL be an opaque numeric ranking signal. A higher score
SHALL mean stronger model-backed support for the candidate. The domain SHALL
NOT assume the score is a probability, a confidence value, a calibrated
likelihood, statistically calibrated, or comparable across different models.
The score SHALL NOT be assumed directly suitable for a threshold without a
later policy. The infrastructure MAY store a monotone normalization of the
ranking signals (e.g., softmax across candidate mean log-probabilities) as
judgment scores; the domain contract remains that only the ORDER of scores
carries meaning — any monotone transform of the ranking signal satisfies the
contract, and no domain or application logic SHALL interpret score magnitudes
as probabilities or confidence.

#### Scenario: Higher score means stronger support
- **WHEN** two judgments for the same input have scores `a > b`
- **THEN** candidate `a` is ranked as having stronger support than candidate `b`

#### Scenario: Score is not a probability
- **WHEN** a judgment score is interpreted
- **THEN** only its order is meaningful; magnitudes are not assumed to be probabilities, confidence values, or calibrated likelihoods

#### Scenario: No probability constraint imposed
- **WHEN** a judgment is created with a score outside `[0, 1]`
- **THEN** the score is accepted provided it is a valid finite numeric value

#### Scenario: Monotone normalization permitted
- **WHEN** the infrastructure stores softmax-normalized scores instead of raw mean log-probabilities
- **THEN** the domain contract is satisfied because the candidate ordering is unchanged

### Requirement: Score validity
The score SHALL have a well-defined numeric representation. The implementation
SHALL reject invalid values according to the domain's existing validation
conventions. Special floating-point values `NaN`, positive infinity, and
negative infinity SHALL NOT be silently accepted as valid candidate scores. The
domain SHALL NOT silently clamp invalid scores into a valid range.

#### Scenario: Valid finite score accepted
- **WHEN** a judgment is created with a finite numeric score
- **THEN** the score is accepted

#### Scenario: NaN score rejected
- **WHEN** a judgment is created with a `NaN` score
- **THEN** the creation is rejected

#### Scenario: Positive infinity score rejected
- **WHEN** a judgment is created with a positive infinity score
- **THEN** the creation is rejected

#### Scenario: Negative infinity score rejected
- **WHEN** a judgment is created with a negative infinity score
- **THEN** the creation is rejected

#### Scenario: No silent clamping
- **WHEN** a judgment is created with an out-of-range finite score
- **THEN** the score is not silently clamped into a different valid range

### Requirement: Candidate and judgment relationship
A `Judgment` SHALL belong to exactly one `Candidate`. A `Candidate` MAY be
evaluated multiple times by the application, producing multiple judgments in
different evaluation contexts. The domain SHALL NOT assume that a candidate has
only one globally valid score. Evaluation concerns (repeated evaluation,
different prompts, different models, different scoring strategies, calibration,
experiments) SHALL NOT be encoded into `Candidate` itself.

#### Scenario: Judgment belongs to one candidate
- **WHEN** a judgment is created
- **THEN** it is associated with exactly one candidate

#### Scenario: Candidate evaluated multiple times
- **WHEN** a candidate is evaluated in two different evaluation contexts
- **THEN** two separate judgments are produced for the same candidate

### Requirement: Domain isolation from inference libraries
`Candidate` and `Judgment` SHALL be usable by importing only domain modules and
the Python standard library. They SHALL NOT import or indirectly require
`torch`, `transformers`, `vllm`, `sglang`, `mlx`, `openai`, `aiohttp`,
provider SDKs, tokenizer libraries, inference frameworks, application ports,
infrastructure modules, or interface modules.

#### Scenario: Domain types import only domain and stdlib
- **WHEN** `Candidate` and `Judgment` are imported
- **THEN** only domain modules and the Python standard library are required

#### Scenario: Domain types do not import inference libraries
- **WHEN** `Candidate` and `Judgment` are inspected for dependencies
- **THEN** they do not require `torch`, `transformers`, `vllm`, `sglang`, `mlx`, `openai`, `aiohttp`, or any inference framework

### Requirement: Dependency direction
The dependency direction SHALL be `domain` ← `application` ←
`infrastructure`, with interfaces remaining outermost. `domain` SHALL NOT
import `application`, `infrastructure`, or `interfaces`. `application` MAY
import `domain` and SHALL NOT import concrete infrastructure classifier
implementations. `infrastructure` MAY import `domain` and `application.ports`
and SHALL NOT import application use cases or `interfaces`. `Classifier` SHALL
be the dependency inversion boundary between application orchestration and
infrastructure classification. `Candidate` and `Judgment` SHALL remain domain
models usable by the `Classifier` implementation internally.

#### Scenario: Domain does not depend on application
- **WHEN** `domain` is inspected
- **THEN** it does not import `application`

#### Scenario: Domain does not depend on infrastructure
- **WHEN** `domain` is inspected
- **THEN** it does not import `infrastructure`

#### Scenario: Application depends on port not concrete classifier
- **WHEN** an application use case classifies input
- **THEN** it imports the `Classifier` port, not a concrete infrastructure classifier

#### Scenario: Infrastructure depends on ports not use cases
- **WHEN** `infrastructure` is inspected
- **THEN** it may import `application.ports` but does not import application use cases or `interfaces`
