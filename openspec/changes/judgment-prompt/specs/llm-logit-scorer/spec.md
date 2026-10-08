# Spec Delta

## MODIFIED Requirements

### Requirement: Prompt construction in infrastructure
Prompt construction SHALL be owned entirely by infrastructure. The scorer
SHALL construct a model-facing scoring prompt from the classification input text
and the candidate. The prompt SHALL represent the candidate explicitly. The
domain and application layers SHALL NOT contain prompt text. The scorer SHALL
NOT accept prompt fragments or prompt templates from the application; the
application provides semantic data (`text`, `candidate`), not prompt parts. The
prompt SHALL carry semantic classification criteria that define what constitutes
a good classification, extracted from the existing classification task
semantics. The prompt SHALL NOT contain generation-protocol instructions
instructing the model to return JSON, return a confidence value, or use a
structured-output schema. Selection-oriented task instructions SHALL be
rewritten into candidate-evaluation semantics rather than excluded. The prompt
SHALL be candidate-specific: each candidate receives its own logical prompt, and
the candidate set SHALL NOT be embedded into one multi-class prompt. The prompt
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
- **THEN** it accepts text and candidates, not prompt fragments or prompt templates

#### Scenario: Prompt carries semantic classification criteria
- **WHEN** a scoring prompt is constructed
- **THEN** it contains semantic classification criteria that define what constitutes a good classification

#### Scenario: Prompt excludes generation-protocol instructions
- **WHEN** a scoring prompt is inspected
- **THEN** it does not contain instructions to return JSON, return a confidence value, or use a structured-output schema

#### Scenario: Prompt is candidate-specific
- **WHEN** the scorer evaluates multiple candidates
- **THEN** each candidate receives its own logical prompt and no prompt contains more than one candidate

#### Scenario: Prompt exposes prefix/candidate boundary
- **WHEN** a scoring prompt is constructed
- **THEN** the boundary between the contextual prefix and the candidate continuation is explicitly exposed
