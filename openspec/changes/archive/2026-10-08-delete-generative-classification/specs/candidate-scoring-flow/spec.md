# Spec Delta

## MODIFIED Requirements

### Requirement: Classification domain concept
A `Classification` SHALL be an immutable domain model representing the
application-level classification decision produced from a set of judgments. It
SHALL associate the selected `Candidate` with the full judgment set from which
the decision was derived, stored as a `tuple[Judgment, ...]` so that the
judgment sequence is deeply immutable. `Classification` SHALL NOT expose logits,
token IDs, tokenizer state, model objects, provider response objects, prompt
text, or transport metadata. `Classification` is the sole application-level
classification result; no generative `ClassificationResult` schema exists.

#### Scenario: Classification associates selected candidate and judgments
- **WHEN** a classification is created from a selected candidate and its judgments
- **THEN** it exposes that candidate and those judgments as a tuple

#### Scenario: Classification immutability
- **WHEN** a classification has been created
- **THEN** its selected candidate cannot be mutated and its judgments tuple cannot be mutated or appended to

#### Scenario: Classification free of model internals
- **WHEN** a classification is inspected
- **THEN** it exposes no logits, token IDs, tokenizer state, model objects, prompt text, or transport metadata

#### Scenario: Classification is the sole classification result
- **WHEN** the classification flow produces a decision
- **THEN** the result is a `Classification` and no `ClassificationResult` type exists in the codebase

## REMOVED Requirements

### Requirement: Existing generative classification preserved
**Reason**: The generative classification path is being deleted. Candidate
scoring is now the sole supported classification mechanism.
**Migration**: All production classification callers use
`CandidateScorer.score(text, candidates)` → `ClassificationPolicy.classify(judgments)`
→ `Classification`. No compatibility layer or fallback to generative
classification SHALL exist.
