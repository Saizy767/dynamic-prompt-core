# Spec Delta

## ADDED Requirements

### Requirement: Active classification rules as scoring input
The `CandidateScorer` port SHALL receive the active classification rules as an
explicit argument to every `score()` invocation. The rules SHALL originate from
the active `PromptArtifact.layers.rules` selected for the classification
operation. The scorer SHALL NOT derive rules from a hardcoded default, from a
separately loaded prompt, from global mutable state, or by parsing rendered
prompt text when the structured `PromptArtifact.layers.rules` data is
available. An empty rule list SHALL NOT silently mean that default rules
should be used; the caller is responsible for supplying the correct rule set.

#### Scenario: Rules from active prompt version
- **WHEN** the scorer is invoked with rules
- **THEN** the rules originate from the active prompt version's `PromptArtifact.layers.rules`

#### Scenario: No hardcoded default rules in scorer
- **WHEN** the scorer implementation is inspected
- **THEN** it does not recover the active rule set from a hardcoded default prompt constant

#### Scenario: No global mutable state for rules
- **WHEN** the scorer receives rules
- **THEN** the rules are passed as an explicit argument, not communicated through global mutable state

#### Scenario: Empty rules are not silently defaulted
- **WHEN** the scorer is invoked with an empty rule list for a non-default prompt
- **THEN** the empty list is used as supplied and is not silently replaced with default rules

## MODIFIED Requirements

### Requirement: CandidateScorer contract
The port SHALL accept a classification input, a collection of candidates to
evaluate, and the active classification rules as a sequence of strings. The
port SHALL be asynchronous, matching existing outbound ports, because
model-backed scoring is expected to perform I/O or asynchronous inference. The
port SHALL NOT expose infrastructure-specific asynchronous primitives. A
successful call SHALL return exactly one `Judgment` per supplied candidate. If
the scorer cannot evaluate any supplied candidate, the call SHALL raise an
exception. The contract SHALL NOT support partial results: the scorer SHALL
NOT return fewer judgments than candidates. The rules SHALL be semantic
classification criteria from the active prompt version's
`PromptArtifact.layers.rules`, not prompt fragments, prompt templates, or
rendered prompt text.

#### Scenario: Accepts input and candidates
- **WHEN** the scorer is invoked
- **THEN** it accepts a classification input, a collection of candidates, and the active classification rules

#### Scenario: Returns exactly one judgment per candidate
- **WHEN** the scorer successfully evaluates N candidates
- **THEN** it returns exactly N judgments

#### Scenario: No partial results
- **WHEN** the scorer cannot evaluate one or more candidates
- **THEN** the call raises an exception rather than returning a partial set of judgments

#### Scenario: Asynchronous contract
- **WHEN** the scorer is invoked
- **THEN** the operation is asynchronous, consistent with existing outbound ports

#### Scenario: No infrastructure async primitives
- **WHEN** the port contract is inspected
- **THEN** it exposes no infrastructure-specific asynchronous primitives

#### Scenario: Rules are semantic criteria not prompt fragments
- **WHEN** the port contract is inspected
- **THEN** the rules parameter accepts semantic classification criteria as strings, not prompt fragments, prompt templates, or rendered prompt text
