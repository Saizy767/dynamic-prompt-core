# Spec Delta

## Purpose

Owns crash-safe persistence of experiment results: atomic JSONL writes, an
explicit run-status lifecycle, idempotent saves, concurrent-write protection,
corrupted-file detection, and recovery of interrupted runs.

## ADDED Requirements

### Requirement: Atomic result persistence
The component SHALL persist results by serializing the full JSONL payload to a
unique temporary file in the same directory as the target, flushing and fsyncing
the temp file, reading it back to validate every line as well-formed JSON, then
atomically replacing the target via `os.replace`. The component SHALL NOT open
the target file for direct in-place overwrite. On any failure before the atomic
replace, the target file SHALL be left untouched and the error SHALL propagate
to the caller. Each write operation SHALL use a unique temporary file name.

#### Scenario: Successful save writes valid file
- **WHEN** results are saved successfully
- **THEN** the target file contains all records as valid JSONL and is readable

#### Scenario: Write failure leaves target intact
- **WHEN** a write to the temporary file fails
- **THEN** the target file is unchanged and the error propagates to the caller

#### Scenario: Replace failure leaves target intact
- **WHEN** the atomic replace fails
- **THEN** the target file is unchanged, the temporary file is removed, and the
  error propagates

#### Scenario: No partial file after confirmed error
- **WHEN** a save operation raises an error
- **THEN** the target file is either the previous confirmed content or absent,
  never a partially written or truncated document

### Requirement: Run-status lifecycle
The component SHALL track an explicit status for each run: `running`,
`completed`, `failed`, or `interrupted`. A new run SHALL be registered as
`running`. The component SHALL set `completed` only after all required result
data is validated and persisted to disk. A save failure SHALL NOT produce
`completed`. The component SHALL support transitioning a run to `failed` with
diagnostic information. The component SHALL support retrieving the current
status of a run.

#### Scenario: New run registered as running
- **WHEN** a run is started
- **THEN** its status is `running`

#### Scenario: Completed only after validated persist
- **WHEN** results are saved and validated
- **THEN** the run status becomes `completed`

#### Scenario: Save failure does not complete
- **WHEN** a save fails
- **THEN** the run status is not `completed`

#### Scenario: Failed status records diagnostics
- **WHEN** a run is marked as failed
- **THEN** the status includes diagnostic information about the failure

### Requirement: Idempotent saves
The component SHALL treat re-saving the same `run_id` with identical result data
as a no-op that returns the existing path. The component SHALL raise a conflict
error when a save with the same `run_id` provides different data than what is
already persisted, rather than silently overwriting.

#### Scenario: Duplicate save with same data is no-op
- **WHEN** the same `run_id` is saved twice with identical results
- **THEN** the second save returns the existing path without modifying the file

#### Scenario: Duplicate save with different data raises conflict
- **WHEN** the same `run_id` is saved with different results than already
  persisted
- **THEN** the save raises a conflict error and the existing file is unchanged

### Requirement: Concurrency safety within a process
The component SHALL serialize all mutating operations on a single repository
instance using a reentrant lock that protects the entire read-modify-write
cycle. Concurrent reads SHALL observe either the complete previous state or the
complete new state, never a partial state.

#### Scenario: Concurrent saves are serialized
- **WHEN** two threads simultaneously save results to the same repository
- **THEN** both saves complete without corruption or lost data

#### Scenario: Concurrent read during save
- **WHEN** a reader loads results while a save is in progress
- **THEN** the reader observes either the complete previous state or the
  complete new state

### Requirement: Corrupted file detection on load
The component SHALL detect a corrupted last line in a JSONL file (partial write
from a crash) and raise an explicit error. Correct records before the corruption
boundary SHALL remain readable via a recovery read. The component SHALL NOT
silently accept a corrupted record as valid or silently drop it. The component
SHALL distinguish a corrupted file from an absent file.

#### Scenario: Corrupted last line raises error
- **WHEN** a JSONL file has a truncated or malformed last line
- **THEN** loading raises an error identifying the file and the corrupt line

#### Scenario: Correct records before corruption are readable
- **WHEN** a JSONL file has valid records followed by a corrupted last line
- **THEN** a recovery read returns the valid records and reports the corruption

#### Scenario: Absent file is distinct from corruption
- **WHEN** the results file does not exist
- **THEN** loading raises a distinct error from a corrupted-file error

### Requirement: Recovery of interrupted runs
The component SHALL provide a recovery method that identifies runs left in
`running` status from a previous process, transitions them to `interrupted`, and
reports diagnostics. The recovery SHALL NOT delete or modify result files. The
recovery SHALL be idempotent: running it twice produces the same result without
duplicates or data loss.

#### Scenario: Interrupted run detected on restart
- **WHEN** a run was left in `running` status by a previous process
- **THEN** recovery identifies it, marks it `interrupted`, and reports
  diagnostics

#### Scenario: Recovery does not delete results
- **WHEN** recovery processes an interrupted run
- **THEN** the run's result files are not deleted or modified

#### Scenario: Recovery is idempotent
- **WHEN** recovery is run twice
- **THEN** the second run produces no changes and no duplicates

### Requirement: Error classification
The component SHALL distinguish error classes: validation errors, write errors,
corruption errors, and conflict errors. Error messages SHALL include the
operation type and `run_id` when known. Error messages SHALL NOT include prompt
text, sensitive configuration data, or large model responses.

#### Scenario: Validation error distinct from write error
- **WHEN** a validation fails
- **THEN** the error type distinguishes it from a disk write failure

#### Scenario: Error messages contain run_id not prompt text
- **WHEN** an error occurs during a save or load
- **THEN** the error message includes the operation type and `run_id` if known,
  but not prompt text or sensitive data

### Requirement: Backward-compatible format
The component SHALL continue to read existing JSONL result files written by the
previous implementation. The component SHALL add a `schema_version` field to
new result documents if absent. The public `save_results` and `load_results`
signatures SHALL remain unchanged.

#### Scenario: Old JSONL file loads correctly
- **WHEN** a JSONL file written by the previous implementation is loaded
- **THEN** all records are returned correctly

#### Scenario: Schema version added on save
- **WHEN** results are saved without a `schema_version`
- **THEN** the saved document includes a `schema_version` field
