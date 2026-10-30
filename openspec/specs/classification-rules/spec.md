# Spec

## Purpose

Provide an immutable domain model representing the normalized set of
classification rules extracted from the active prompt artifact, distinguishing
absence of rules from invalid rules and preserving rule order and text.

## Requirements

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
construction or conversion. The orchestration layer SHALL pass the empty
representation to the `Classifier` as an honest zero-shot baseline: the
classifier prompt is built with an empty rules section.

#### Scenario: Empty rules are valid
- **WHEN** `ClassificationRules` is constructed with an empty tuple
- **THEN** no exception is raised and `len(rules) == 0`

#### Scenario: Empty rules run zero-shot
- **WHEN** the orchestration layer receives an empty `ClassificationRules`
- **THEN** the `Classifier` is called and the classifier prompt contains an empty rules section

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

### Requirement: Rule count constraint enforced at composition time
The rule-count limit for classification prompts SHALL be enforced at
composition time by the prompt composer (`max_rules`, default 5).
`ClassificationRules` itself SHALL NOT enforce a count limit (allowing zero
rules for the zero-shot absence case), and the `Classifier` SHALL NOT
re-validate the count at its boundary — it trusts the `ClassificationRules`
produced upstream.

#### Scenario: Empty rules carry no count constraint
- **WHEN** `ClassificationRules` has zero rules
- **THEN** no count validation is triggered anywhere and the classifier runs the zero-shot baseline

#### Scenario: Composer enforces the limit
- **WHEN** the composer assembles a new prompt version
- **THEN** the number of rules respects `max_rules` (default 5)

#### Scenario: Classifier trusts upstream validation
- **WHEN** the `Classifier` receives non-empty `ClassificationRules`
- **THEN** no rule-count re-validation runs before scoring
