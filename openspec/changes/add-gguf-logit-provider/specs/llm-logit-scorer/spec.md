# Spec Delta

## ADDED Requirements

### Requirement: Backend selection through configuration
The scorer factory SHALL select between the Hugging Face backend and the GGUF
backend using a configured backend identifier. Selecting the Hugging Face
backend SHALL preserve current behavior exactly: the same adapters, model
loading path, and scoring semantics as before this change. Selecting the GGUF
backend SHALL construct the GGUF model and tokenizer adapters that implement
the existing `BatchedCausalLanguageModel` and `BatchTokenizerAdapter` seams.
Backend selection SHALL be infrastructure-internal; the `CandidateScorer` port
and all domain/application contracts SHALL remain unchanged. The configuration
SHALL extend the existing configuration mechanism rather than introducing a
separate configuration file.

#### Scenario: Hugging Face backend preserves current behavior
- **WHEN** the backend is configured as `"huggingface"`
- **THEN** the scorer factory constructs the existing Hugging Face adapters and scoring behavior is identical to before this change

#### Scenario: GGUF backend constructs GGUF adapters
- **WHEN** the backend is configured as `"gguf"`
- **THEN** the scorer factory constructs the GGUF model and tokenizer adapters implementing the existing seams

#### Scenario: Port contract unchanged
- **WHEN** the `CandidateScorer` port is inspected after backend selection is introduced
- **THEN** its signature and semantics are identical to before this change

#### Scenario: No separate configuration file
- **WHEN** backend selection configuration is inspected
- **THEN** it extends the existing configuration mechanism and does not introduce a new configuration file

### Requirement: Backend selection fails fast at initialization
Invalid backend names, missing model files, unsupported settings, and
unavailable optional dependencies SHALL fail during initialization rather than
halfway through an optimization cycle. A backend selection failure SHALL be
translated through the existing `CandidateScoringError` boundary when it
surfaces during scoring, but configuration and dependency failures SHALL be
raised at construction time with actionable messages.

#### Scenario: Invalid backend name fails at init
- **WHEN** the backend is configured with an unrecognized name
- **THEN** initialization raises an actionable error naming the valid backends before any scoring occurs

#### Scenario: Missing model file fails at init
- **WHEN** the configured model path does not exist
- **THEN** initialization raises an actionable error before any scoring occurs

#### Scenario: Missing optional dependency fails at init
- **WHEN** the GGUF backend is selected but the optional dependency is unavailable
- **THEN** initialization raises an actionable error before any scoring occurs

#### Scenario: Failures surface before optimization cycle
- **WHEN** an invalid backend configuration is provided
- **THEN** the failure occurs during initialization, not during a scoring invocation inside an optimization cycle

### Requirement: GGUF backend reuses shared scoring components
When the GGUF backend is selected, the scorer SHALL use the same
`LLMLogitCandidateScorer`, `LogitScorer`, `ScoringPromptBuilder`, causal
logit alignment, and mean-log-probability scoring algorithm as the Hugging
Face backend. The only difference SHALL be the concrete model and tokenizer
adapters. The GGUF backend SHALL NOT introduce a second implementation of
classification metrics, candidate acceptance, prompt versioning, or rollback
logic.

#### Scenario: Shared scorer and logit scorer
- **WHEN** the GGUF backend is selected
- **THEN** the scorer uses the same `LLMLogitCandidateScorer` and `LogitScorer` as the Hugging Face backend

#### Scenario: Shared prompt construction
- **WHEN** the GGUF backend is selected
- **THEN** prompt construction uses the same `ScoringPromptBuilder` and prompt semantics as the Hugging Face backend

#### Scenario: No duplicated metrics or acceptance logic
- **WHEN** the GGUF backend is inspected
- **THEN** it contains no separate classification metrics, candidate acceptance, prompt versioning, or rollback implementation
