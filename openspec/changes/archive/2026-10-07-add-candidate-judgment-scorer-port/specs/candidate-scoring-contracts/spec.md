# Spec Delta

## Purpose

Define the domain model for candidate-based classification scoring (`Candidate`
and `Judgment`), the `CandidateScorer` application outbound port contract
through which the application requests candidate judgments without depending on
any LLM implementation, and the evaluation-vs-decision boundary that keeps the
scorer from becoming a classifier.

## ADDED Requirements

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
later policy.

#### Scenario: Higher score means stronger support
- **WHEN** two judgments for the same input have scores `a > b`
- **THEN** candidate `a` is ranked as having stronger support than candidate `b`

#### Scenario: Score is not a probability
- **WHEN** a judgment score is interpreted
- **THEN** it is not assumed to be a probability, confidence value, or calibrated likelihood

#### Scenario: No probability constraint imposed
- **WHEN** a judgment is created with a score outside `[0, 1]`
- **THEN** the score is accepted provided it is a valid finite numeric value

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

### Requirement: CandidateScorer port location
`CandidateScorer` SHALL be an outbound port residing under
`application/ports/outbound/candidate_scorer`. The port SHALL be owned by the
application layer. The port SHALL be defined as a `typing.Protocol`.
Infrastructure SHALL implement the port without requiring the application to
inherit from an infrastructure class.

#### Scenario: Port resides under application outbound ports
- **WHEN** the `CandidateScorer` port is located
- **THEN** it is under `application/ports/outbound/candidate_scorer`

#### Scenario: Port is a typing.Protocol
- **WHEN** the `CandidateScorer` port is defined
- **THEN** it is a `typing.Protocol`, not an infrastructure base class

#### Scenario: Infrastructure implements without application inheritance
- **WHEN** a concrete scorer is implemented in infrastructure
- **THEN** it implements the port without the application inheriting from an infrastructure class

### Requirement: CandidateScorer contract
The port SHALL accept a classification input and a collection of candidates to
evaluate. The port SHALL be asynchronous, matching existing outbound ports,
because model-backed scoring is expected to perform I/O or asynchronous
inference. The port SHALL NOT expose infrastructure-specific asynchronous
primitives. A successful call SHALL return exactly one `Judgment` per supplied
candidate. If the scorer cannot evaluate any supplied candidate, the call SHALL
raise an exception. The contract SHALL NOT support partial results: the scorer
SHALL NOT return fewer judgments than candidates.

#### Scenario: Accepts input and candidates
- **WHEN** the scorer is invoked
- **THEN** it accepts a classification input and a collection of candidates

#### Scenario: Returns exactly one judgment per candidate
- **WHEN** the scorer successfully evaluates N candidates
- **THEN** it returns exactly N judgments

#### Scenario: No partial results
- **WHEN** the scorer cannot evaluate one or more candidates
- **THEN** the call raises an exception rather than returning a partial set of judgments

#### Scenario: Asynchronous contract
- **WHEN** the scorer is invoked
- **THEN** the operation is asynchronous, consistent with existing outbound ports

#### Scenario: No infrastructure async primitives
- **WHEN** the port contract is inspected
- **THEN** it exposes no infrastructure-specific asynchronous primitives

### Requirement: Multiple candidates per invocation
The port SHALL support evaluating multiple candidates in one invocation. The
API SHALL NOT require the application to call the scorer once per candidate.
Supporting multiple candidates does not require the implementation to perform
batching: the application expresses a set of candidate evaluations and
infrastructure decides how efficiently those evaluations are executed.

#### Scenario: Multiple candidates in one call
- **WHEN** the scorer is invoked with multiple candidates
- **THEN** all candidates are evaluated in a single invocation

#### Scenario: One call per candidate not required
- **WHEN** the application evaluates a set of candidates
- **THEN** it is not required to invoke the scorer once per candidate

### Requirement: Candidate-judgment association
Every returned `Judgment` SHALL explicitly contain its `Candidate`, so callers
do not rely on list position. The application SHALL NOT infer candidate identity
solely from the position of a returned score. Because a successful call returns
exactly one judgment per candidate, silent omission SHALL NOT occur: a candidate
that cannot be evaluated causes the call to fail rather than being omitted.

#### Scenario: Judgment carries its candidate
- **WHEN** the scorer returns judgments
- **THEN** every judgment explicitly contains its candidate

#### Scenario: No positional inference
- **WHEN** the application reads returned judgments
- **THEN** it does not infer candidate identity from list position

#### Scenario: No silent omission
- **WHEN** a candidate cannot be evaluated
- **THEN** the call raises an exception rather than omitting that candidate's judgment

### Requirement: Duplicate candidate validation at application boundary
Candidates in a scoring request SHALL be unique. Duplicate candidates constitute
an invalid request, not a scorer-specific concern. Validation SHALL be performed
at the application boundary before the scorer is invoked. A validation helper
SHALL reject duplicate candidates by raising an exception. The scorer SHALL NOT
silently deduplicate duplicate candidates.

#### Scenario: Duplicate candidates rejected at application boundary
- **WHEN** a scoring request contains duplicate candidates
- **THEN** the application boundary validation rejects the request before the scorer is invoked

#### Scenario: No silent deduplication
- **WHEN** duplicate candidates are supplied
- **THEN** the request is rejected rather than silently deduplicated into one judgment

#### Scenario: Unique candidates accepted
- **WHEN** a scoring request contains only unique candidates
- **THEN** the validation passes

### Requirement: Scoring semantics
`CandidateScorer` SHALL evaluate candidates independently and SHALL NOT decide
the final classification. The scorer SHALL NOT select a winning candidate, rank
candidates for selection, or aggregate judgments into a classification decision.
Score calibration, candidate ranking, candidate selection, and judgment
aggregation are all outside this change. The scorer's sole responsibility is to
produce one judgment per candidate. This boundary prevents the scorer from
becoming a classifier.

#### Scenario: Scorer evaluates candidates independently
- **WHEN** the scorer evaluates a set of candidates
- **THEN** each candidate is evaluated independently and receives its own judgment

#### Scenario: Scorer does not decide classification
- **WHEN** the scorer returns judgments
- **THEN** it does not produce a classification decision

#### Scenario: Scorer does not select winning candidate
- **WHEN** the scorer returns judgments
- **THEN** it does not select a winning candidate

#### Scenario: Scorer does not rank or aggregate
- **WHEN** the scorer returns judgments
- **THEN** it does not rank candidates for selection or aggregate judgments into a decision

#### Scenario: Calibration and aggregation are outside this change
- **WHEN** this change is completed
- **THEN** no score calibration, candidate ranking, candidate selection, or judgment aggregation is introduced

### Requirement: Port information boundary
The `CandidateScorer` port SHALL NOT expose logits, token IDs, tokenizer APIs,
model objects, or provider-specific types. The port SHALL remain valid if the
underlying implementation changes between logits-based LLM scoring,
generated-token probability scoring, another local model, a remote model, a
deterministic test scorer, or another future scoring mechanism.

#### Scenario: No logits exposed
- **WHEN** the port contract is inspected
- **THEN** it exposes no logits

#### Scenario: No token IDs exposed
- **WHEN** the port contract is inspected
- **THEN** it exposes no token IDs

#### Scenario: No tokenizer or model objects exposed
- **WHEN** the port contract is inspected
- **THEN** it exposes no tokenizer APIs or model objects

#### Scenario: No provider types exposed
- **WHEN** the port contract is inspected
- **THEN** it exposes no provider-specific types

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
import `domain` and SHALL NOT import concrete infrastructure scorers.
`infrastructure` MAY import `domain` and `application.ports` and SHALL NOT
import application use cases or `interfaces`. `CandidateScorer` SHALL be the
dependency inversion boundary between application orchestration and
infrastructure scoring.

#### Scenario: Domain does not depend on application
- **WHEN** `domain` is inspected
- **THEN** it does not import `application`

#### Scenario: Domain does not depend on infrastructure
- **WHEN** `domain` is inspected
- **THEN** it does not import `infrastructure`

#### Scenario: Application depends on port not concrete scorer
- **WHEN** an application use case scores candidates
- **THEN** it imports the `CandidateScorer` port, not a concrete infrastructure scorer

#### Scenario: Infrastructure depends on ports not use cases
- **WHEN** `infrastructure` is inspected
- **THEN** it may import `application.ports` but does not import application use cases or `interfaces`

### Requirement: Existing classification behavior preserved
The existing generative classification pipeline SHALL continue to operate
unchanged. `BaselineRunner`, `AsyncTask.classify_detailed`, `LLMClient.classify`,
`ClassificationResult`, existing parsing and validation behavior, `run_cycle`,
and existing classification tests SHALL continue to work unchanged. No existing
caller SHALL be migrated to `CandidateScorer` in this change.

#### Scenario: Generative pipeline unchanged
- **WHEN** this change is completed
- **THEN** the generative classification pipeline continues to operate unchanged

#### Scenario: No caller migrated
- **WHEN** this change is completed
- **THEN** no existing caller is migrated to `CandidateScorer`

#### Scenario: Existing tests pass
- **WHEN** the existing classification test suite is run
- **THEN** all tests pass

### Requirement: No concrete scorer or inference introduced
This change SHALL NOT introduce a concrete `CandidateScorer` implementation,
logits extraction, tokenization, model loading, batching, score calibration, or
classification aggregation. No inference library SHALL be introduced into the
domain or application port.

#### Scenario: No concrete scorer
- **WHEN** this change is completed
- **THEN** no concrete `CandidateScorer` implementation exists in infrastructure

#### Scenario: No logits or tokenization
- **WHEN** this change is completed
- **THEN** no logits extraction, tokenization, or batching code is introduced

#### Scenario: No inference library in domain or port
- **WHEN** the domain or port is inspected
- **THEN** no inference library is introduced

### Requirement: Port boundary architecture validation
An import-linter contract SHALL validate the `CandidateScorer` port boundary:
`application.ports` SHALL NOT import `infrastructure` or `interfaces`, directly
or indirectly. The contract SHALL be added together with the port module. No
forward-looking import-linter contract referencing a nonexistent module SHALL be
added.

#### Scenario: Port boundary contract added with port
- **WHEN** the `CandidateScorer` port module is introduced
- **THEN** an import-linter contract validating the port boundary is added in the same change

#### Scenario: Application ports do not import infrastructure
- **WHEN** `lint-imports` is run
- **THEN** the port boundary contract verifies that `application.ports` does not import `infrastructure` or `interfaces`, directly or indirectly
