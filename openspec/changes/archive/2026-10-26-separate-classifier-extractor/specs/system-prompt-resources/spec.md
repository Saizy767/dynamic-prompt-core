# Spec Delta

## Purpose

Provide file-based system prompt resources for the `Classifier` and `Extractor`
components, packaged with the Python distribution and loaded at component
initialization, with error handling for missing or unreadable files.

## ADDED Requirements

### Requirement: Prompt resource files exist in a dedicated directory
Two separate system prompt files SHALL reside under
`infrastructure/llm/prompts/`: one for the `Classifier` and one for the
`Extractor`. The file format SHALL be simple text unless the existing
infrastructure requires otherwise. The directory SHALL NOT contain
classification rules; rules remain in `PromptArtifact`.

#### Scenario: Classifier prompt file exists
- **WHEN** the prompt resource directory is inspected
- **THEN** a file for the classifier system prompt exists under `infrastructure/llm/prompts/`

#### Scenario: Extractor prompt file exists
- **WHEN** the prompt resource directory is inspected
- **THEN** a file for the extractor system prompt exists under `infrastructure/llm/prompts/`

#### Scenario: Files are separate
- **WHEN** the two prompt files are compared
- **THEN** they are distinct files with independent content

### Requirement: Each component loads its own prompt at initialization
The `Classifier` and `Extractor` SHALL each independently load their system
prompt from their respective resource file at initialization time. The
`PromptArtifact` SHALL NOT be used to store the fixed system prompts of the new
components. Existing entities in `domain/prompts/fixed.py` SHALL be removed only
when they become unused; other prompts and code SHALL be preserved.

#### Scenario: Classifier loads its prompt
- **WHEN** the classifier is initialized
- **THEN** it reads its system prompt from its resource file

#### Scenario: Extractor loads its prompt
- **WHEN** the extractor is initialized
- **THEN** it reads its system prompt from its resource file

#### Scenario: PromptArtifact not used for fixed system prompts
- **WHEN** the new components are initialized
- **THEN** they do not use `PromptArtifact` to store or retrieve their fixed system prompts

### Requirement: Prompt resources are packaged with the Python distribution
The prompt resource files SHALL be included in the Python package distribution
via `package-data` or equivalent mechanism. The files SHALL be accessible after
`pip install` and not only when running from the repository root. The packaging
configuration SHALL be updated in `pyproject.toml`.

#### Scenario: Files accessible after installation
- **WHEN** the package is installed via `pip install`
- **THEN** the prompt resource files are accessible at the expected import path

#### Scenario: Files accessible from repository root
- **WHEN** the package is run from the repository root
- **THEN** the prompt resource files are accessible at the expected path

### Requirement: Missing prompt file uses existing error handling
When a prompt resource file is missing or unreadable, the component SHALL use
the project's existing error-handling mechanism. The component SHALL NOT silently
substitute an empty or default prompt. The error SHALL be clear and actionable.

#### Scenario: Missing file raises clear error
- **WHEN** a prompt resource file does not exist at initialization
- **THEN** the component raises a clear, actionable error

#### Scenario: Unreadable file raises clear error
- **WHEN** a prompt resource file exists but cannot be read
- **THEN** the component raises a clear, actionable error

#### Scenario: No silent fallback
- **WHEN** a prompt resource file is missing
- **THEN** the component does not silently substitute an empty or default prompt
