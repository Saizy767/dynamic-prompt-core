# Spec Delta

## REMOVED Requirements

### Requirement: Existing generative classification preserved
**Reason**: The generative classification pipeline is being deleted. The
judgment prompt is now the sole classification-oriented prompt, used by the
candidate-scoring infrastructure. There is no generative classification path
to preserve.
**Migration**: The judgment prompt continues to serve the candidate-scoring
architecture. The generative classification prompt's generation-only
components are removed. The semantic criteria are already represented by
`ScoringPromptBuilder`.
