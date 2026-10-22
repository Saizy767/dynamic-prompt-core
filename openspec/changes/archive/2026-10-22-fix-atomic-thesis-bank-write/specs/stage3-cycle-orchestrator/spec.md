# Spec Delta

## ADDED Requirements

### Requirement: Atomic thesis bank persistence

When `update_in_prompt_flags` writes the updated thesis bank, it SHALL do so
atomically: the existing file SHALL remain untouched until the new content is
fully serialized, fsynced, and read-back validated as parseable JSON. The
replacement SHALL use an atomic rename (`os.replace`). If any step fails, the
original file SHALL be left intact and the temporary file SHALL be removed.

#### Scenario: Successful update replaces file
- **WHEN** `update_in_prompt_flags` writes a valid updated thesis bank
- **THEN** the file at `thesis_bank_path` contains the new content and the
  old content is fully replaced

#### Scenario: Original file preserved on write failure
- **WHEN** the write fails after the original file exists
- **THEN** the original file at `thesis_bank_path` is unchanged and no
  temporary file remains in the directory

#### Scenario: Temp file cleaned up on failure
- **WHEN** an exception occurs during serialization or validation
- **THEN** the temporary file is removed and does not persist in the
  thesis bank's directory
