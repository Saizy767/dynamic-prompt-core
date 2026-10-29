# Spec Delta

## Purpose

Owns crash-safe persistence of experiment results: atomic JSONL writes,
concurrent-write protection, and corrupted-file detection.

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
The component SHALL detect a corrupted line in a JSONL file (partial write from
a crash) and raise an explicit error naming the file and line. Correct records
before the corruption boundary SHALL remain readable via a recovery read that
returns them together with corruption diagnostics. The recovery read SHALL NOT
modify the file. An absent file SHALL raise an error identifying the file.

#### Scenario: Corrupted line raises error
- **WHEN** a JSONL file has a truncated or malformed line
- **THEN** loading raises an error identifying the file and the corrupt line

#### Scenario: Correct records before corruption are readable
- **WHEN** a JSONL file has valid records followed by a corrupted line
- **THEN** a recovery read returns the valid records and reports the corruption

#### Scenario: Absent file raises error
- **WHEN** the results file does not exist
- **THEN** loading raises an error identifying the missing file

### Requirement: Error classification
The component SHALL distinguish error classes: validation errors, write errors,
and corruption errors. Error messages SHALL include the operation type and
`run_id` or file path when known. Error messages SHALL NOT include prompt text,
sensitive configuration data, or large model responses.

#### Scenario: Validation error distinct from write error
- **WHEN** a validation fails
- **THEN** the error type distinguishes it from a disk write failure

#### Scenario: Error messages contain run_id not prompt text
- **WHEN** an error occurs during a save or load
- **THEN** the error message includes the operation type and `run_id` or file
  path if known, but not prompt text or sensitive data

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
