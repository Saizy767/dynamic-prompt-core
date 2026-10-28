# Spec Delta

## REMOVED Requirements

### Requirement: Sequential compatibility adapter is temporary
**Reason**: The sequential compatibility adapter is dead code — defined in
infrastructure but referenced by no production path and no test. With the
HuggingFace and GGUF backends removed, the only model adapter is the
llama-server HTTP adapter, which evaluates batch items via concurrent HTTP
requests; no sequential-delegation compatibility mechanism exists or is
needed.
**Migration**: None. The adapter class is deleted; no caller exists. The
underlying guarantees it existed for (application never branches on batch
support; batch contract preserved) remain covered by the "Multi-candidate API
is an application contract not a model batch" and "Batch scoring preserves
existing port and information boundary" requirements.
