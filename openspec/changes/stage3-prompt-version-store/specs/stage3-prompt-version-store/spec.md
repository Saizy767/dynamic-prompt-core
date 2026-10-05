# Spec Delta

## Purpose

Own the lifecycle of prompt versions: persist every version produced by the
prompt composer, assign monotonic version numbers, keep exactly one version
active at a time, archive displaced versions, enforce immutability of saved
content, and reconstruct version lineage. The store is the single source of
truth for which prompt version is currently in use.

## ADDED Requirements

### Requirement: Prompt version schema
The component SHALL define a prompt version record with the following fields:
`version` (integer, monotonically increasing), `text` (full prompt text), `hash`
(sha256 of `text`), `rules` (list of rule texts), `source_candidates` (list of
`cluster_id`), `base_version` (the version this one derives from), `created_at`
(ISO timestamp), `active` (boolean), `archived` (boolean), and `reason` (why the
version was created). The record SHALL be stored as a JSON object.

#### Scenario: Valid record
- **WHEN** a version is saved
- **THEN** all required fields are present and `hash` equals sha256 of `text`

#### Scenario: Missing required field
- **WHEN** a record is missing `version`
- **THEN** saving fails with an error naming the missing field

### Requirement: Persist prompt version
The component SHALL persist every new prompt version to the store. The version
number SHALL be assigned by the store, not the caller, and SHALL be one greater
than the highest existing version number. The component SHALL fail with an error
when a version with that number already exists.

#### Scenario: New version persisted
- **WHEN** the composer hands a new prompt version to the store
- **THEN** the store assigns the next version number and writes the record

#### Scenario: Version collision
- **WHEN** a version with the assigned number already exists
- **THEN** saving fails with an error naming the collision

#### Scenario: Monotonic numbering
- **WHEN** versions v0, v1, v2 exist
- **THEN** the new version is assigned number 3

### Requirement: Retrieve by version id
The component SHALL support retrieving a prompt version by version number. The
component SHALL fail with an error naming the missing version when the version is
not found.

#### Scenario: Version retrieved
- **WHEN** version number 2 is requested and it exists
- **THEN** the full record is returned

#### Scenario: Version not found
- **WHEN** the requested version number does not exist
- **THEN** retrieval fails with an error naming the missing version

### Requirement: Retrieve active version
The component SHALL support retrieving the current active version. At any time
exactly one active version SHALL exist. The component SHALL fail with an error
when no active version is set.

#### Scenario: Active version retrieved
- **WHEN** an active version exists
- **THEN** the active version record is returned

#### Scenario: No active version
- **WHEN** no version is marked active
- **THEN** retrieval fails with an error naming the missing active version

### Requirement: Mark version as active
The component SHALL support marking a version as active. When a version is marked
active, the previous active version SHALL be marked inactive and archived. The
component SHALL fail with an error when the requested version does not exist.

#### Scenario: Version activated
- **WHEN** a version is marked active
- **THEN** that version has `active=true` and all others have `active=false`

#### Scenario: Previous version archived
- **WHEN** a new version becomes active
- **THEN** the previous active version has `archived=true` and `active=false`

#### Scenario: Activation of missing version
- **WHEN** activation is requested for a non-existent version
- **THEN** activation fails with an error naming the missing version

### Requirement: Immutability of persisted versions
The component SHALL forbid modification of saved version record content. After a
version is saved, its `text`, `hash`, `rules`, `source_candidates`,
`base_version`, and `created_at` fields SHALL NOT change. Only `active` and
`archived` MAY change.

#### Scenario: Immutable fields
- **WHEN** an update attempt touches `text`
- **THEN** the update is rejected with an error

#### Scenario: Mutable fields
- **WHEN** an update touches `active` or `archived`
- **THEN** the update is accepted

### Requirement: List all versions
The component SHALL support listing all versions sorted by version number with
their metadata (`version`, `hash`, `active`, `archived`, `created_at`,
`base_version`). The list SHALL NOT include the full `text` field.

#### Scenario: List versions
- **WHEN** all versions are requested
- **THEN** the list contains each version's metadata, sorted by version number

#### Scenario: Empty store
- **WHEN** the store is empty
- **THEN** the list is empty

### Requirement: Version lineage
The component SHALL support reconstructing a version's origin by following
`base_version` links. The lineage SHALL be returned as an ordered list from the
oldest ancestor to the requested version.

#### Scenario: Lineage reconstructed
- **WHEN** lineage is requested for v3 with `base_version=2`
- **THEN** the lineage is `[v0, v1, v2, v3]`

#### Scenario: Broken lineage
- **WHEN** a version references a `base_version` that does not exist
- **THEN** lineage reconstruction fails with an error naming the missing ancestor

### Requirement: Persistence format
The component SHALL persist the store as a single JSON file at a configurable
path. The file SHALL contain all version records in one structure. The component
SHALL support atomic writes to prevent corruption on crash.

#### Scenario: Store persisted
- **WHEN** a version is added or updated
- **THEN** the store file is rewritten atomically

#### Scenario: Store reloaded
- **WHEN** the store file is loaded
- **THEN** all version records and `active`/`archived` flags are restored

#### Scenario: Corrupted store
- **WHEN** the store file is corrupted
- **THEN** loading fails with an error naming the corrupted file

### Requirement: Version count limit
The component MAY enforce a maximum number of stored versions. When the limit is
reached, the oldest archived versions SHALL become candidates for deletion.
Deletion SHALL NOT affect the active version or any version in the active
version's lineage.

#### Scenario: Pruning eligible
- **WHEN** the number of versions exceeds the limit
- **THEN** the oldest archived versions are deleted

#### Scenario: Active version protected
- **WHEN** pruning runs
- **THEN** the active version and its ancestors are preserved

### Requirement: Logging of store operations
The component SHALL log all store operations: save version, activate, archive,
retrieve, reload. The log SHALL be append-only.

#### Scenario: Operations logged
- **WHEN** any store operation occurs
- **THEN** it is appended to the store log with a timestamp and version number

### Requirement: Configurable parameters
The component SHALL support configuration via config for `store_path` (required),
`max_versions` (default unlimited), `atomic_writes` (default true), and
`log_path` (default `prompt_store.jsonl`).

#### Scenario: Custom store path
- **WHEN** config sets `store_path=/custom/path.json`
- **THEN** the store is persisted at that path

#### Scenario: Pruning enabled
- **WHEN** config sets `max_versions=10` and 15 versions exist
- **THEN** the 5 oldest archived versions are deleted
