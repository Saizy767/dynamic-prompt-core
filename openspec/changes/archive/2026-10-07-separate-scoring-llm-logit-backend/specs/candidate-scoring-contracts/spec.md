# Spec Delta

## REMOVED Requirements

### Requirement: No concrete scorer or inference introduced
**Reason**: This was a deferral marker for the contract-introduction stage
(`candidate-scoring-architecture-prep`), which intentionally introduced only the
`Candidate`, `Judgment`, and `CandidateScorer` port contracts without a concrete
implementation. This change introduces the concrete `CandidateScorer`
implementation backed by LLM logits and the application-level
scoring-classification flow, so the prohibition on a concrete scorer, logits
extraction, and tokenization no longer holds.
**Migration**: The concrete scorer and its infrastructure seams are defined by
the new `llm-logit-scorer` capability; the application flow is defined by the
new `candidate-scoring-flow` capability. All other
`candidate-scoring-contracts` requirements (port location, information boundary,
scoring semantics, candidate-judgment association, duplicate validation, domain
isolation, dependency direction, existing-classification preservation) remain
unchanged and continue to govern the contracts.
