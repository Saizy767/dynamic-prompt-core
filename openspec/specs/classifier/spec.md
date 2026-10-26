# Spec

## Purpose

Provide a self-contained classification component that encapsulates the full
classification process — rule validation, candidate handling, logit-based
scoring, and final decision — behind a single outbound port with an independent
system prompt, returning the existing `Classification` domain type.

## Requirements

### Requirement: Classifier port location and protocol
`Classifier` SHALL be an outbound port residing under
`application/ports/outbound/`. The port SHALL be owned by the application layer
and defined as a `typing.Protocol`. Infrastructure SHALL implement the port
without requiring the application to inherit from an infrastructure class. The
port SHALL NOT expose logits, token IDs, tokenizer APIs, model objects, or
provider-specific types.

#### Scenario: Port resides under application outbound ports
- **WHEN** the `Classifier` port is located
- **THEN** it is under `application/ports/outbound/`

#### Scenario: Port is a typing.Protocol
- **WHEN** the `Classifier` port is defined
- **THEN** it is a `typing.Protocol`, not an infrastructure base class

#### Scenario: No infrastructure internals exposed
- **WHEN** the port contract is inspected
- **THEN** it exposes no logits, token IDs, tokenizer APIs, model objects, or provider-specific types

### Requirement: Classifier contract returns Classification
The `Classifier` port SHALL accept a classification input (text) and a
`ClassificationRules` instance. The port SHALL be asynchronous. A successful
call SHALL return a `Classification` — the existing domain type representing the
final classification decision, including the selected candidate and the full
judgment set. The port SHALL NOT return `list[Judgment]` or any intermediate
representation; the decision is produced inside the `Classifier`.

#### Scenario: Accepts text and rules
- **WHEN** the classifier is invoked
- **THEN** it accepts a classification input text and a `ClassificationRules` instance

#### Scenario: Returns Classification directly
- **WHEN** the classifier successfully classifies an input
- **THEN** it returns a `Classification` with a selected candidate and judgment set

#### Scenario: Asynchronous contract
- **WHEN** the classifier is invoked
- **THEN** the operation is asynchronous, consistent with existing outbound ports

### Requirement: Classifier encapsulates the full classification process
The `Classifier` implementation SHALL be solely responsible for: validating the
input rule set, working with candidates, forming the input for logit-based
scoring, obtaining and processing scores, determining the final decision, and
returning the result as the existing `Classification` type. The implementation
SHALL absorb the responsibilities previously held by `CandidateScorer` and
`ClassificationPolicy` as separate abstractions. `BaselineRunner` SHALL NOT
directly use scoring details, candidate evaluation, or decision logic.

#### Scenario: Validation at classifier boundary
- **WHEN** the classifier receives rules
- **THEN** it validates the rule set before proceeding with scoring

#### Scenario: Scoring and decision are internal
- **WHEN** the classifier processes an input
- **THEN** candidate evaluation, scoring, and decision selection all occur inside the classifier

#### Scenario: Runner does not access scoring details
- **WHEN** `BaselineRunner` classifies an example
- **THEN** it calls the `Classifier` port and receives a `Classification`, without accessing scoring internals

### Requirement: Classifier preserves logit-based scoring mechanism
The `Classifier` implementation SHALL preserve the existing logit-based scoring
mechanism: model forward pass, tokenizer, causal next-token logit alignment,
mean candidate-token log-probability, and softmax over candidate scores. The
mechanism SHALL NOT be replaced by generative text generation. The existing
`LogitScorer` algorithm and tokenizer/model adapters SHALL be reused or adapted
inside the `Classifier` implementation.

#### Scenario: Logit scoring preserved
- **WHEN** the classifier scores candidates
- **THEN** it uses logit-based scoring with causal next-token alignment, not text generation

#### Scenario: Softmax over candidate scores
- **WHEN** the classifier produces scores for multiple candidates
- **THEN** the scores are softmax-normalized as in the existing implementation

### Requirement: Classifier preserves decision logic
The `Classifier` implementation SHALL preserve the existing decision logic:
select the candidate with the maximum score, with ties broken by input order
(first candidate wins). The judgment order SHALL be preserved in the returned
`Classification.judgments`. The implementation SHALL NOT calibrate scores,
normalize scores, or redefine the score scale.

#### Scenario: Highest scoring candidate selected
- **WHEN** candidates with scores `A=0.2`, `B=0.8` are classified
- **THEN** the selected candidate is `B`

#### Scenario: Tie broken by input order
- **WHEN** two candidates share the maximum score
- **THEN** the first candidate in input order is selected

#### Scenario: Judgment order preserved
- **WHEN** the classifier returns a `Classification`
- **THEN** `Classification.judgments` preserves the candidate input order

### Requirement: Classifier uses an independent system prompt
The `Classifier` SHALL load and use a dedicated system prompt from a resource
file under `infrastructure/llm/prompts/`. The system prompt SHALL become the
initial part of the text prefix used for logit scoring, followed by rules, input
text, and candidate as in the existing scoring mechanism. A separate `system`
segment SHALL NOT be introduced. Adding or changing a classification rule SHALL
NOT change the `Classifier` system prompt.

#### Scenario: Dedicated system prompt loaded
- **WHEN** the classifier is initialized
- **THEN** it loads its system prompt from a dedicated resource file

#### Scenario: System prompt is prefix of scoring text
- **WHEN** the classifier builds the scoring prefix
- **THEN** the system prompt is the initial part, followed by rules, input text, and candidate

#### Scenario: Rule changes do not affect system prompt
- **WHEN** a classification rule is added or changed
- **THEN** the `Classifier` system prompt remains unchanged

### Requirement: Classifier preserves error handling policy
The `Classifier` SHALL preserve the existing error-handling policy: failures at
tokenization, model inference, and logit scoring SHALL propagate as a
classification error. The `Classifier` SHALL NOT silently substitute a `0`
classification or produce fake classifications on error. Errors SHALL be
distinguishable from the absent-rules case (which sets classification to `0`
without calling the `Classifier`). No partial results SHALL be returned.

#### Scenario: Scoring failure propagates error
- **WHEN** a model inference failure occurs during classification
- **THEN** the error propagates and no fake classification is produced

#### Scenario: Error is not treated as absent rules
- **WHEN** the classifier raises an error
- **THEN** the error is not silently converted to a `0` classification

#### Scenario: No partial results on failure
- **WHEN** a batched model call fails during scoring
- **THEN** the error propagates and no partial judgments are returned

### Requirement: Classifier receives dependencies through DI
The `Classifier` SHALL receive its infrastructure dependencies (tokenizer
adapter, model adapter, logit scorer, prompt builder) through constructor
injection. The composition root SHALL construct the `Classifier` and pass it to
`BaselineRunner` and `RunCycleDeps`. There SHALL be a single production
construction path for the `Classifier` implementation.

#### Scenario: Dependencies injected at construction
- **WHEN** the classifier is constructed
- **THEN** it receives tokenizer, model, logit scorer, and prompt builder via constructor arguments

#### Scenario: Single production construction path
- **WHEN** the composition root assembles dependencies
- **THEN** the `Classifier` is constructed through a shared factory, not duplicated across composition roots

### Requirement: Classifier does not block the event loop
Awaiting classification through the `Classifier` implementation SHALL NOT
synchronously block the event loop on model inference. The existing
non-blocking async mechanism SHALL be preserved.

#### Scenario: Event loop progress during scoring
- **WHEN** the classifier is awaited with a model adapter that blocks on a synchronization primitive
- **THEN** a concurrent `asyncio` task makes progress, proving blocking inference is offloaded
