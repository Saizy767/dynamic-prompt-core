# Spec Delta

## MODIFIED Requirements

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

## ADDED Requirements

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
