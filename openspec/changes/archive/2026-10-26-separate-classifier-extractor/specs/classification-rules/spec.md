# Spec Delta

## Purpose

Provide an immutable domain model representing the normalized set of
classification rules extracted from the active prompt artifact, distinguishing
absence of rules from invalid rules and preserving rule order and text.

## ADDED Requirements

### Requirement: ClassificationRules is an immutable domain model
`ClassificationRules` SHALL be a frozen dataclass residing in
`domain/models/classification_rules.py`. It SHALL expose a single field
`rules: tuple[str, ...]`. The model SHALL be deeply immutable: the tuple
container SHALL not permit mutation, and the individual rule strings SHALL not
be altered after construction. `ClassificationRules` SHALL NOT import
application, infrastructure, or interface modules.

#### Scenario: Immutable rules tuple
- **WHEN** a `ClassificationRules` instance is created with rules `("a", "b")`
- **THEN** the `rules` field is a tuple and cannot be mutated or appended to

#### Scenario: Domain-only imports
- **WHEN** `ClassificationRules` is imported
- **THEN** only domain modules and the Python standard library are required

### Requirement: Conversion from PromptArtifact preserves order and text
A conversion function SHALL transform `PromptArtifact.layers.rules` into
`ClassificationRules`. The conversion SHALL preserve the order of rules and the
exact text of each rule. The conversion SHALL NOT normalize, trim, or otherwise
alter rule text unless the existing contract requires it. When `layers` is
`None` or `layers.rules` is an empty list, the conversion SHALL produce an empty
`ClassificationRules` (zero rules).

#### Scenario: Rules preserved in order
- **WHEN** a `PromptArtifact` with `layers.rules = ["rule C", "rule A", "rule B"]` is converted
- **THEN** the resulting `ClassificationRules.rules` is `("rule C", "rule A", "rule B")`

#### Scenario: Rule text unchanged
- **WHEN** a rule with leading/trailing whitespace or mixed case is converted
- **THEN** the rule text in `ClassificationRules` is identical to the source text

#### Scenario: None layers produces empty rules
- **WHEN** a `PromptArtifact` with `layers = None` is converted
- **THEN** the resulting `ClassificationRules.rules` is an empty tuple

#### Scenario: Empty rules list produces empty rules
- **WHEN** a `PromptArtifact` with `layers.rules = []` is converted
- **THEN** the resulting `ClassificationRules.rules` is an empty tuple

### Requirement: Empty rules represent absence, not an error
An empty `ClassificationRules` (zero rules) SHALL be a valid representation of
the absence of classification rules. It SHALL NOT raise an exception at
construction or conversion. The orchestration layer SHALL use the empty
representation to skip classifier invocation and set classification to `0`
without calling the `Classifier`. An empty `ClassificationRules` SHALL NOT
trigger `build_classification_prompt()`.

#### Scenario: Empty rules are valid
- **WHEN** `ClassificationRules` is constructed with an empty tuple
- **THEN** no exception is raised and `len(rules) == 0`

#### Scenario: Empty rules do not trigger prompt building
- **WHEN** the orchestration layer receives an empty `ClassificationRules`
- **THEN** it does not call `build_classification_prompt()` and sets classification to `0`

### Requirement: Invalid rules are distinguished from empty rules
The conversion SHALL distinguish between absent rules (valid, empty
`ClassificationRules`) and invalid rules (malformed, non-string elements, or
otherwise corrupt). Invalid rules SHALL trigger the existing error-handling
mechanism and SHALL NOT be silently converted to an empty
`ClassificationRules`. The boundary between valid emptiness and invalid data
SHALL be tested.

#### Scenario: Non-string rule element is invalid
- **WHEN** a `PromptArtifact` contains a non-string element in `layers.rules`
- **THEN** the conversion raises an error and does not produce an empty `ClassificationRules`

#### Scenario: Malformed rules are not silently emptied
- **WHEN** a `PromptArtifact` has structurally present but corrupt rules
- **THEN** the conversion raises an error rather than returning an empty `ClassificationRules`

### Requirement: Rule count constraint preserved for prompt building
The existing constraint of 3-5 rules for `build_classification_prompt()` SHALL
be preserved. `ClassificationRules` itself SHALL NOT enforce the 3-5 limit
(allowing zero rules for absence), but when rules are non-empty and
`build_classification_prompt()` is invoked, the existing validation SHALL apply.
The `Classifier` SHALL validate the rule set at its boundary before invoking the
scoring mechanism.

#### Scenario: Empty rules bypass the 3-5 constraint
- **WHEN** `ClassificationRules` has zero rules
- **THEN** no 3-5 validation is triggered because `build_classification_prompt()` is not called

#### Scenario: Non-empty rules validated at classifier boundary
- **WHEN** the `Classifier` receives non-empty `ClassificationRules`
- **THEN** the existing 3-5 rule validation applies before scoring proceeds
