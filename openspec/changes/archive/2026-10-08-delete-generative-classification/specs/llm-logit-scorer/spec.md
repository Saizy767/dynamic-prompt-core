# Spec Delta

## MODIFIED Requirements

### Requirement: Batch scoring preserves existing port and information boundary
The `CandidateScorer` port, `ClassifyInput`, `Classification`, `Judgment`,
and `ClassificationPolicy` SHALL remain unchanged. No model, tokenizer,
tensor, padding, attention mask, or batch structure details SHALL leak into
domain or application. Batch scoring SHALL be additive to the infrastructure
layer only. The generative classification path SHALL NOT exist; candidate
scoring is the sole classification mechanism.

#### Scenario: Port contract unchanged
- **WHEN** the `CandidateScorer` port is inspected after batch scoring is introduced
- **THEN** its signature and semantics are identical to before

#### Scenario: No batch internals leak to domain or application
- **WHEN** the domain and application layers are inspected
- **THEN** they contain no token IDs, padding, attention masks, tensor shapes, or batch structures

#### Scenario: No generative classification path
- **WHEN** the codebase is inspected for generative classification
- **THEN** no `classify_detailed`, `ClassificationResult`, or generative classification fallback exists
