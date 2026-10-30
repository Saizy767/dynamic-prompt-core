# Spec

## Purpose

Own the lifecycle of prompt versions: persist every version produced by the
prompt composer, assign monotonic version numbers, keep exactly one version
active at a time, archive displaced versions, enforce immutability of saved
content, and reconstruct version lineage. The store is the single source of
truth for which prompt version is currently in use.

## Requirements

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
when a version with that number already exists. The component SHALL NOT report
success unless the new record is fully persisted to disk via an atomic replace.
On any failure before the atomic replace completes, the previous confirmed state
SHALL be preserved both in memory and on disk, and the error SHALL propagate to
the caller without masking.

#### Scenario: New version persisted
- **WHEN** the composer hands a new prompt version to the store
- **THEN** the store assigns the next version number and writes the record

#### Scenario: Version collision
- **WHEN** a version with the assigned number already exists
- **THEN** saving fails with an error naming the collision

#### Scenario: Monotonic numbering
- **WHEN** versions v0, v1, v2 exist
- **THEN** the new version is assigned number 3

#### Scenario: Save failure preserves previous state
- **WHEN** a save fails after the record is computed but before the atomic replace
- **THEN** the in-memory state and the on-disk file both retain the previous
  confirmed state, and the caller receives the error

#### Scenario: No silent success on write error
- **WHEN** the underlying write or replace operation raises an error
- **THEN** the save method raises an error and does not return a record

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
SHALL write atomically: serialize the full document from a snapshot, write it to
a unique temporary file in the same directory as the target file, flush and
fsync the temporary file, read it back and validate it as well-formed JSON, then
atomically replace the target file via `os.replace`. The component SHALL best-effort
fsync the parent directory after the replace for crash durability. The component
SHALL NOT open the target file for direct in-place overwrite. Each write operation
SHALL use a unique temporary file name so that concurrent operations in the same
process do not collide. Temporary files SHALL be removed on failure if they have
not yet become the target file.

#### Scenario: Store persisted
- **WHEN** a version is added or updated
- **THEN** the store file is rewritten atomically via a unique temporary file and
  `os.replace`

#### Scenario: Store reloaded
- **WHEN** the store file is loaded
- **THEN** all version records and `active`/`archived` flags are restored

#### Scenario: Corrupted store
- **WHEN** the store file is corrupted
- **THEN** loading fails with an error naming the corrupted file and the store is
  not silently reinitialized as empty

#### Scenario: Write failure leaves file intact
- **WHEN** a write to the temporary file fails
- **THEN** the target file is unchanged and contains the last confirmed state

#### Scenario: Replace failure leaves file intact
- **WHEN** the atomic replace fails
- **THEN** the target file is unchanged, the temporary file is removed, and the
  error propagates to the caller

#### Scenario: No partial JSON after confirmed error
- **WHEN** a save operation raises an error
- **THEN** the target file is either the previous confirmed state or absent, never
  a partially written or truncated JSON document

#### Scenario: Temporary file validated before replace
- **WHEN** the temporary file is written
- **THEN** it is read back and validated as well-formed JSON before the atomic
  replace is attempted

#### Scenario: Crash recovery
- **WHEN** the process crashes during a write
- **THEN** the target file is either the last confirmed state or absent, and any
  orphaned temporary file is not treated as valid store data

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

### Requirement: Transactional consistency of memory and disk
The component SHALL treat every mutating operation as a transaction: either the
new state is fully confirmed on disk and then published in memory, or the
previous confirmed state remains in memory and on disk. The component SHALL NOT
mutate the published in-memory state before the atomic replace succeeds, unless a
reliable rollback to the prior state is guaranteed on any failure. The component
SHALL NOT leave the in-memory state diverged from the on-disk state after a
confirmed error.

#### Scenario: Memory unchanged on disk write failure
- **WHEN** a mutating operation fails during the disk write
- **THEN** the in-memory state matches the on-disk state, which is the previous
  confirmed state

#### Scenario: Memory updated only after successful persist
- **WHEN** a mutating operation succeeds
- **THEN** the in-memory state and the on-disk state both reflect the new state

#### Scenario: Activate failure preserves state
- **WHEN** an activate operation fails during persistence
- **THEN** the previous active and archived flags are unchanged in memory and on
  disk

### Requirement: Concurrency safety within a process
The component SHALL serialize all mutating operations on a single store instance
using a reentrant lock that protects the entire read-modify-write cycle: reading
the current state, computing the new version or flag changes, mutating the
state, and persisting to disk. The component SHALL NOT use a single fixed
temporary file path for different operations. The component SHALL NOT silently
overwrite a newer state with a stale snapshot. Concurrent reads SHALL observe
either the complete previous state or the complete new state, never a partial
state.

#### Scenario: Concurrent saves are serialized
- **WHEN** two threads simultaneously save new versions to the same store
- **THEN** both versions are persisted with distinct version numbers and neither
  update is lost

#### Scenario: Concurrent activate and save
- **WHEN** one thread activates a version while another saves a new version
- **THEN** the final on-disk state reflects a consistent ordering of both
  operations with no corruption

#### Scenario: Concurrent read during save
- **WHEN** a reader retrieves a version while a save is in progress
- **THEN** the reader observes either the complete previous state or the complete
  new state

#### Scenario: No stale overwrite
- **WHEN** a stale snapshot attempts to replace a newer state
- **THEN** the newer state is not silently overwritten

### Requirement: Load integrity and corruption handling
The component SHALL distinguish an absent store file from a corrupted store file
from a regular access error. An absent file SHALL initialize an empty store per
existing policy. A corrupted JSON file SHALL raise an explicit error naming the
file and the cause, and SHALL NOT be silently replaced with an empty store. A
file with an invalid top-level structure or missing `versions` dict SHALL raise
an explicit error. Log messages SHALL include the operation type and file path
but SHALL NOT include prompt text content.

#### Scenario: Absent file initializes empty
- **WHEN** the store file does not exist
- **THEN** the store initializes as empty

#### Scenario: Corrupted JSON rejected
- **WHEN** the store file contains invalid JSON
- **THEN** loading raises an error naming the file and the parse cause, and the
  store is not initialized as empty

#### Scenario: Invalid top-level structure rejected
- **WHEN** the store file is valid JSON but the top level is not an object or
  `versions` is not a dict
- **THEN** loading raises an error naming the file and the structural problem

#### Scenario: Prompt text not leaked in logs
- **WHEN** a load or save error is logged
- **THEN** the log entry contains the operation type and file path but not the
  prompt text

### Requirement: Input validation before persistence
The component SHALL validate the full document before serialization: the
top-level structure, the presence and type of all required record fields, the
correctness of each record's hash, the absence of malformed entries in the
version history, and JSON serializability. Validation errors SHALL be explicit
and SHALL distinguish a validation failure from a write failure.

#### Scenario: Invalid record rejected before write
- **WHEN** a record is missing a required field or has an incorrect hash
- **THEN** the save fails with a validation error before any disk write is
  attempted

#### Scenario: Non-serializable content rejected
- **WHEN** a record contains content that cannot be serialized to JSON
- **THEN** the save fails with a validation error before any disk write is
  attempted

#### Scenario: Validation error distinct from write error
- **WHEN** a validation fails
- **THEN** the error type or message distinguishes it from a disk write failure
