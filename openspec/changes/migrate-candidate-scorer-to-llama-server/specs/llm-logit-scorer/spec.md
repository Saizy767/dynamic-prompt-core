# Spec Delta

## MODIFIED Requirements

### Requirement: Sequential compatibility adapter is temporary
If a concrete model adapter cannot support native batching, that limitation
SHALL be handled inside infrastructure. A compatibility adapter MAY implement
batch execution by delegating to sequential execution (`forward_batch(items)`
calls `forward(item)` for each item). For local in-process adapters, this is
permitted ONLY as a temporary transitional compatibility mechanism for adapters
that lack native batch support. The production local-model adapter used by
`LLMLogitCandidateScorer` MUST use native tensor batching. Remote/HTTP model
adapters (e.g., a `llama-server` adapter) MAY permanently use sequential or
concurrent per-item delegation when the remote endpoint does not expose a
native batched-logit tensor API, because each HTTP request is independently
GPU-accelerated and the network round-trip — not local tensor batching — is
the dominant cost. Sequential or concurrent delegation is permitted for
explicitly supported compatibility adapters and for remote/HTTP adapters.
The application SHALL NOT branch on whether batching is supported.

#### Scenario: Compatibility adapter preserves contract
- **WHEN** a local model adapter does not support native batching
- **THEN** a temporary compatibility adapter delegates to sequential execution and preserves the scoring contract

#### Scenario: Production local adapter uses native batching
- **WHEN** the production local-model adapter is used
- **THEN** it performs native tensor batching, not sequential delegation

#### Scenario: Remote adapter may delegate permanently
- **WHEN** a remote/HTTP model adapter is used and the endpoint lacks a native batched-logit API
- **THEN** the adapter MAY permanently delegate batch execution to per-item HTTP requests

#### Scenario: Application does not branch on batch support
- **WHEN** the application invokes the scorer
- **THEN** it does not check whether the model adapter supports batching
