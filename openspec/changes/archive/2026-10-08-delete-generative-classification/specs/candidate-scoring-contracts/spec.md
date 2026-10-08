# Spec Delta

## REMOVED Requirements

### Requirement: Existing classification behavior preserved
**Reason**: The generative classification pipeline is being deleted. Candidate
scoring is now the sole classification mechanism.
**Migration**: All production classification callers use `CandidateScorer`.
`BaselineRunner` is migrated to the candidate-scoring flow.
`AsyncTask.classify_detailed`, `LLMClient.classify`, and `ClassificationResult`
are removed. `run_cycle` and prompt management utilities are updated to remove
dead references to the generative classification path.
