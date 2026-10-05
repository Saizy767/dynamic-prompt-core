# stage3-prompt-composer-v2 Specification

## Purpose

Load a rule-candidate artifact, formulate each candidate into a single-sentence
rule via an LLM, verify semantic fidelity against the source cluster centroid,
assemble a new classification prompt version from the unchanged base layers plus
accepted rules, validate hard limits, and persist a reloadable prompt-version
artifact with full lineage back to source clusters.

## Requirements

### Requirement: Load rule-candidate artifact
The component SHALL load a JSON artifact containing the selected candidates with
fields `cluster_id`, `representative_theses`, `precision`, `frequency`,
`positive_hits`, `negative_hits`, and `rank`. The component SHALL fail with an
error naming the first problematic candidate and field when a candidate does not
conform to the schema.

#### Scenario: Valid artifact loads
- **WHEN** the artifact contains candidates with all required fields
- **THEN** the candidates are loaded for rule formulation

#### Scenario: Missing candidate field
- **WHEN** a candidate is missing `representative_theses`
- **THEN** loading fails with an error naming the `cluster_id` and the missing
  field

### Requirement: Load base prompt layers
The component SHALL load the base layers (`role`, `task`, `output contract`,
`fallback`) from the currently active prompt version. The base layers SHALL NOT
be modified during composition. The component SHALL fail with an error naming the
missing layer when any base layer is absent.

#### Scenario: Base layers loaded
- **WHEN** the active prompt version contains `role`, `task`, `output contract`,
  and `fallback`
- **THEN** those four layers are carried over unchanged into the new version

#### Scenario: Missing base layer
- **WHEN** the active prompt version is missing `fallback`
- **THEN** composition fails with an error naming `fallback`

### Requirement: Formulate rules via LLM
The component SHALL formulate one rule per candidate via an LLM call. The LLM
input SHALL be the candidate's `representative_theses`, `precision`, and
`frequency`. The LLM output SHALL be a single sentence of at most
`max_rule_words` words. The component SHALL use a fixed formulation prompt that
SHALL NOT be optimized.

#### Scenario: Rule formulated
- **WHEN** a cluster contains theses "отмена заявка", "отказ карта",
  "отменить оформление"
- **THEN** the LLM returns a rule such as "Если в тексте упоминается отмена или
  отказ от оформления — класс 1"

#### Scenario: Rule length within limit
- **WHEN** the LLM returns a rule longer than `max_rule_words`
- **THEN** the rule is either truncated to the limit or rejected with a log entry

### Requirement: Check for semantic distortion
The component SHALL verify each formulated rule for semantic distortion relative
to its source cluster. The check SHALL compare the embedding of the rule against
the centroid of the source cluster using cosine similarity. The threshold SHALL
be configurable via config. When similarity is below the threshold, the rule
SHALL be rejected, or a single reformulation attempt SHALL be made before
rejection.

#### Scenario: Rule passes distortion check
- **WHEN** the cosine similarity between the rule embedding and the cluster
  centroid is at or above `distortion_threshold`
- **THEN** the rule is accepted

#### Scenario: Rule fails distortion check
- **WHEN** the cosine similarity is below `distortion_threshold`
- **THEN** the rule is rejected and the fact is logged

#### Scenario: Reformulation attempt
- **WHEN** a rule fails the distortion check on the first attempt and
  `allow_reformulation` is true
- **THEN** the component makes one reformulation attempt; if it also fails, the
  rule is rejected

### Requirement: Assemble new prompt version
The component SHALL assemble a new version of the classification system prompt
from the base layers plus the accepted rules. The rules section SHALL be replaced
entirely by the new set of accepted rules. The assembled prompt SHALL preserve
the order: `role`, `task`, `rules`, `output contract`, `fallback`.

#### Scenario: Prompt assembled
- **WHEN** 4 rules are accepted
- **THEN** the new prompt version contains `role`, `task`, 4 rules,
  `output contract`, and `fallback` in that order

#### Scenario: Base layers unchanged
- **WHEN** a new version is assembled
- **THEN** the `role`, `task`, `output contract`, and `fallback` are identical to
  those in the active version

### Requirement: Validate prompt limits
The component SHALL validate the new prompt version against hard limits: number
of rules SHALL NOT exceed `max_rules`, total prompt tokens SHALL NOT exceed
`max_prompt_tokens`, and `output contract` and `fallback` SHALL NOT be empty.
When a limit is violated, assembly SHALL fail with an error naming the violated
limit.

#### Scenario: Limits satisfied
- **WHEN** the new version contains 5 rules and 140 tokens
- **THEN** validation passes and the version is accepted

#### Scenario: Too many rules
- **WHEN** 7 rules are assembled and `max_rules` is 5
- **THEN** assembly fails with "too many rules: 7 > 5"

#### Scenario: Prompt too long
- **WHEN** the prompt contains 160 tokens and `max_prompt_tokens` is 150
- **THEN** assembly fails with "prompt too long: 160 > 150"

#### Scenario: Empty fallback
- **WHEN** the fallback layer is empty
- **THEN** assembly fails with "fallback is empty"

### Requirement: Prompt version artifact
The component SHALL persist the new prompt version as a JSON artifact whose
filename contains `version`, `run_id`, and `timestamp`. The artifact SHALL
contain: `version`, `text` (full prompt text), `hash` (sha256 of the `text`),
`rules` (list of rule texts), `source_candidates` (list of `cluster_id` used),
`base_version` (the version the new one derives from), and `created_at`.

#### Scenario: Artifact named consistently
- **WHEN** composition completes
- **THEN** the file is named `prompt_v{version}_{run_id}_{timestamp}.json`

#### Scenario: Artifact reloaded
- **WHEN** the artifact is loaded
- **THEN** the `version`, `text`, `hash`, and `rules` are available without
  re-composition

### Requirement: Version lineage
The component SHALL record the link between each rule in the new version and the
`cluster_id` from which it was derived. The component SHALL support reconstructing
the chain `version → rules → clusters → theses`.

#### Scenario: Lineage preserved
- **WHEN** a rule in version v3 derives from cluster 7
- **THEN** the artifact for v3 contains `source_candidates: [7]` and the rule
  entry carries `cluster_id: 7`

#### Scenario: Lineage reconstructable
- **WHEN** the chain is queried for a rule
- **THEN** the source cluster and its theses are retrievable from the stored
  artifacts

### Requirement: Logging of composition
The component SHALL log the composition process: how many candidates were
received, how many rules were formulated, how many were rejected for distortion,
how many were rejected for limits, and the final count of rules in the version.

#### Scenario: Composition logged
- **WHEN** composition completes
- **THEN** the log contains `candidates_in`, `rules_formulated`,
  `rejected_distortion`, `rejected_limits`, and `final_rules_count`

### Requirement: Configurable parameters
The component SHALL support configuration via config for `max_rules` (default 5),
`max_prompt_tokens` (default 300), `max_rule_words` (default 15),
`distortion_threshold` (default 0.75), and `allow_reformulation` (default true).

#### Scenario: Custom parameters
- **WHEN** config sets `max_rules=3`
- **THEN** assembly limits the number of rules to three

#### Scenario: Reformulation disabled
- **WHEN** config sets `allow_reformulation=false`
- **THEN** a rule failing the distortion check is rejected immediately without a
  reformulation attempt
