# Spec Delta

## ADDED Requirements

### Requirement: Active rules placed before candidate continuation
The scoring prompt SHALL place the active classification rules after the fixed
candidate-evaluation instructions and before the input text and candidate
continuation. The rules SHALL be clearly identified as classification criteria.
The prompt builder SHALL preserve the order of the supplied rules.

#### Scenario: Rules placed before input and candidate
- **WHEN** a scoring prompt is constructed with rules
- **THEN** the rules appear after the fixed instructions and before the input text and candidate continuation

#### Scenario: Rules identified as classification criteria
- **WHEN** a scoring prompt is constructed with rules
- **THEN** the rules are clearly identified as classification criteria in the prompt

#### Scenario: Rule order preserved
- **WHEN** a scoring prompt is constructed with rules `[r1, r2, r3]`
- **THEN** the rules appear in the prompt in the order `r1`, `r2`, `r3`

### Requirement: No silent rule substitution or discard
The prompt builder SHALL NOT silently discard supplied rules. The prompt
builder SHALL NOT replace supplied rules with default rules. The prompt
builder SHALL NOT duplicate the default classification rules from
`CLASSIFICATION_PROMPT_V0` or any other default prompt constant. An empty rule
list SHALL be used as supplied and SHALL NOT trigger substitution with default
rules.

#### Scenario: Supplied rules not discarded
- **WHEN** the prompt builder is given a non-empty rule list
- **THEN** all supplied rules appear in the resulting prompt

#### Scenario: Supplied rules not replaced with defaults
- **WHEN** the prompt builder is given a non-empty rule list
- **THEN** the rules in the prompt are the supplied rules, not default rules

#### Scenario: No default rule duplication in builder
- **WHEN** the prompt builder implementation is inspected
- **THEN** it does not contain or reference the default classification rules from `CLASSIFICATION_PROMPT_V0`

#### Scenario: Empty rules not substituted
- **WHEN** the prompt builder is given an empty rule list
- **THEN** the resulting prompt does not contain default rules and the empty list is used as supplied

### Requirement: Backend consistency for active rules
All supported scoring backends SHALL receive the same logical scoring prompt
for the same input text, candidate, and rule set. No backend SHALL omit the
active rules. Backend-specific tokenization and model invocation MAY differ,
but the logical prompt content including rules SHALL be equivalent across
backends.

#### Scenario: All backends receive rules
- **WHEN** the same input, candidate, and rules are scored with different backends
- **THEN** every backend's model input contains the same active rules

#### Scenario: No backend omits rules
- **WHEN** a scoring backend constructs a model input
- **THEN** the active rules are present in the model input

### Requirement: Scoring prefix reuse with active rules
The scorer SHALL construct the common prompt prefix — fixed instructions, active
rules, and input text — once per `score()` invocation and reuse it for every
candidate in that invocation. The prefix SHALL NOT be reconstructed per
candidate. The rules SHALL NOT cause additional model calls or prompt artifact
reloads per candidate.

#### Scenario: Prefix constructed once per invocation
- **WHEN** the scorer evaluates N candidates in one invocation with rules
- **THEN** the common prefix including rules is constructed once and reused for all N candidates

#### Scenario: No per-candidate rule reload
- **WHEN** the scorer evaluates multiple candidates
- **THEN** the rules are not reloaded or re-fetched per candidate

## MODIFIED Requirements

### Requirement: Concrete CandidateScorer implementation
A concrete `CandidateScorer` implementation SHALL be provided in infrastructure
that derives candidate scores from LLM logits. The implementation SHALL satisfy
the `CandidateScorer` port contract: `async def score(text, candidates, rules)
-> list[Judgment]`, returning exactly one `Judgment` per supplied candidate. The
implementation's constructor MAY depend on infrastructure-specific
collaborators (tokenizer adapter, model adapter, prompt builder, logit scorer);
those collaborators SHALL NOT appear in the `CandidateScorer` port. The
application SHALL see only `await scorer.score(text, candidates, rules)`.

#### Scenario: Port contract satisfied
- **WHEN** the concrete scorer is invoked
- **THEN** it satisfies the `CandidateScorer` port contract

#### Scenario: One judgment per candidate
- **WHEN** the concrete scorer successfully evaluates N candidates
- **THEN** it returns exactly N judgments

#### Scenario: Infrastructure collaborators not in port
- **WHEN** the `CandidateScorer` port is inspected
- **THEN** it exposes no tokenizer adapter, model adapter, prompt builder, or logit scorer

#### Scenario: Application sees only score
- **WHEN** the application invokes the scorer
- **THEN** it calls `await scorer.score(text, candidates, rules)` and receives judgments

### Requirement: Prompt construction in infrastructure
Prompt construction SHALL be owned entirely by infrastructure. The scorer
SHALL construct a model-facing scoring prompt from the classification input
text, the candidate, and the active classification rules supplied by the
application. The prompt SHALL represent the candidate explicitly. The domain
and application layers SHALL NOT contain prompt text. The scorer SHALL NOT
accept prompt fragments or prompt templates from the application; the
application provides semantic data (`text`, `candidate`, `rules`), not prompt
parts. The prompt SHALL carry the active classification rules as semantic
classification criteria that define what constitutes a good classification.
The prompt SHALL NOT contain generation-protocol instructions instructing the
model to return JSON, return a confidence value, or use a structured-output
schema. Selection-oriented task instructions SHALL be rewritten into
candidate-evaluation semantics rather than excluded. The prompt SHALL be
candidate-specific: each candidate receives its own logical prompt, and the
candidate set SHALL NOT be embedded into one multi-class prompt. The prompt
SHALL expose the boundary between the contextual prefix and the candidate
continuation so the tokenizer can determine candidate continuation positions
structurally.

#### Scenario: Prompt constructed in infrastructure
- **WHEN** the scorer prepares a model input
- **THEN** the prompt is constructed inside infrastructure

#### Scenario: Candidate represented in prompt
- **WHEN** a scoring prompt is constructed
- **THEN** the candidate is represented explicitly in the prompt

#### Scenario: No prompt text in domain or application
- **WHEN** the domain and application layers are inspected
- **THEN** they contain no scoring prompt text

#### Scenario: Application provides semantic data not prompt parts
- **WHEN** the scorer port contract is inspected
- **THEN** it accepts text, candidates, and rules, not prompt fragments or prompt templates

#### Scenario: Prompt carries semantic classification criteria
- **WHEN** a scoring prompt is constructed with active rules
- **THEN** it contains the active classification rules as semantic criteria that define what constitutes a good classification

#### Scenario: Prompt excludes generation-protocol instructions
- **WHEN** a scoring prompt is inspected
- **THEN** it does not contain instructions to return JSON, return a confidence value, or use a structured-output schema

#### Scenario: Prompt is candidate-specific
- **WHEN** the scorer evaluates multiple candidates
- **THEN** each candidate receives its own logical prompt and no prompt contains more than one candidate

#### Scenario: Prompt exposes prefix/candidate boundary
- **WHEN** a scoring prompt is constructed
- **THEN** the boundary between the contextual prefix and the candidate continuation is explicitly exposed
