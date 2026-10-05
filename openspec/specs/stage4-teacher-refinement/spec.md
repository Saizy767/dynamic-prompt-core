# Spec

## Purpose

Refine Tiny-model thesis candidates using a larger teacher model over a
separate endpoint, producing a cleaner, more stable set of extractive theses
for downstream collection while preserving the originals and isolating
per-example teacher failures.

## Requirements

### Requirement: Teacher model endpoint
The component SHALL use a separate endpoint for the teacher model, distinct
from the working model's endpoint. The endpoint and model name SHALL be
configurable via config. The component SHALL fail at initialization with an
error naming the missing key when `teacher.endpoint` is not set.

#### Scenario: Teacher endpoint configured
- **WHEN** config contains `teacher.endpoint` and `teacher.model_name`
- **THEN** the component uses them for teacher-model calls

#### Scenario: Missing teacher endpoint
- **WHEN** config does not contain `teacher.endpoint`
- **THEN** initialization fails with an error naming the missing key

### Requirement: Load thesis candidates
The component SHALL load theses produced by the Tiny model from a baseline
runner results artifact (`theses_raw` and `theses_norm`). The component SHALL
process each example separately. The component SHALL NOT modify the source
results artifact.

#### Scenario: Candidates loaded
- **WHEN** the results artifact contains `theses_raw` for each example
- **THEN** the theses are loaded for teacher-model review

#### Scenario: Missing theses
- **WHEN** an example has `extract_status=failed`
- **THEN** the example is skipped and the fact is logged

### Requirement: Filter noisy theses
The component SHALL identify noisy theses — formulations the teacher model
considers unrelated to the text content or uninformative for classification.
Noisy theses SHALL be marked and SHALL NOT enter the refined collection.

#### Scenario: Noisy thesis filtered
- **WHEN** a thesis is a generic phrase applicable to any text
- **THEN** the thesis is marked as noisy and excluded from the refined collection

#### Scenario: Informative thesis preserved
- **WHEN** a thesis reflects a concrete feature of the text
- **THEN** the thesis is preserved

### Requirement: Filter interpretive theses
The component SHALL distinguish extractive and interpretive theses. Extractive
theses describe features present in the text. Interpretive theses describe
conclusions the model draws about the text. Interpretive theses SHALL be either
reformulated into extractive form or excluded.

#### Scenario: Interpretive thesis reformulated
- **WHEN** a thesis is phrased as "клиент недоволен"
- **THEN** the teacher model reformulates it into extractive form, e.g. "упоминание недовольства"

#### Scenario: Interpretive thesis excluded
- **WHEN** a thesis cannot be reformulated without losing meaning
- **THEN** the thesis is excluded and the fact is logged

### Requirement: Reformulate unstable theses
The component SHALL reformulate theses the teacher model considers unstable —
those whose phrasing may change significantly under minor input changes. A
reformulated thesis SHALL preserve the original meaning and SHALL NOT exceed
the original length.

#### Scenario: Thesis reformulated
- **WHEN** a thesis is phrased in a word-order-dependent form
- **THEN** the teacher model reformulates it into a stable form

#### Scenario: Length preserved
- **WHEN** a thesis is reformulated
- **THEN** the reformulated thesis length does not exceed the original length

### Requirement: Add missed theses
The component SHALL allow the teacher model to add theses the Tiny model missed,
when they are present in the text and relevant for classification. Added theses
SHALL be tagged as produced by the teacher model, not the Tiny model.

#### Scenario: Missed thesis added
- **WHEN** the teacher model finds a feature in the text that the Tiny model missed
- **THEN** the thesis is added with `source=teacher`

#### Scenario: Source tracked
- **WHEN** a thesis is added by the teacher model
- **THEN** the collection stores `source=teacher` for it

### Requirement: Preserve Tiny-model theses
The component SHALL preserve the original Tiny-model theses in a separate field
to enable comparison and rollback. Reformulated and filtered variants SHALL be
stored separately from the originals.

#### Scenario: Original preserved
- **WHEN** a thesis is reformulated
- **THEN** the original form is preserved in `theses_raw_tiny`

#### Scenario: Refined stored separately
- **WHEN** the teacher model finishes processing an example
- **THEN** the result is stored in `theses_refined` and the originals remain in `theses_raw_tiny`

### Requirement: Refined theses artifact
The component SHALL persist the refinement result as an artifact whose filename
contains `run_id`, `prompt_version`, and `timestamp`. The format SHALL be jsonl.
The artifact SHALL contain, for each example: `id`, `theses_raw_tiny`,
`theses_refined`, `filtered_out` (list of excluded theses with reason), and
`added` (list of added theses).

#### Scenario: Artifact named consistently
- **WHEN** processing completes
- **THEN** the file is named `theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl`

#### Scenario: Artifact reloaded
- **WHEN** the artifact is loaded
- **THEN** results are available without re-calling the teacher model

### Requirement: Configurable use of refined theses
The component SHALL support a config setting for whether downstream stages use
`theses_refined` or `theses_raw_tiny`. The setting SHALL be available via config.
By default, `theses_refined` SHALL be used when the artifact is available,
otherwise `theses_raw_tiny`.

#### Scenario: Refined used by default
- **WHEN** the `theses_refined` artifact exists
- **THEN** downstream stages use `theses_refined`

#### Scenario: Fallback to tiny
- **WHEN** the `theses_refined` artifact is absent
- **THEN** downstream stages use `theses_raw_tiny`

### Requirement: Teacher failures do not break pipeline
The component SHALL continue processing when the teacher model errors on a
single example and SHALL preserve that example's original Tiny-model theses
unchanged. The component MUST NOT abort processing due to one error.

#### Scenario: Teacher failure on one example
- **WHEN** the teacher model returns an error for the example with id=42
- **THEN** id=42 keeps its original Tiny-model theses and processing continues

#### Scenario: Summary at the end
- **WHEN** processing completes
- **THEN** a summary is printed with counts of processed, teacher-errors, filtered, reformulated, and added theses

### Requirement: Cost and latency accounting
The component SHALL log, for each teacher-model call: latency, input token
count, output token count, and status. The component SHALL aggregate costs
across the whole run and SHALL include them in the summary.

#### Scenario: Per-call accounting
- **WHEN** the teacher model responds
- **THEN** a log entry is written with latency and token counts

#### Scenario: Summary accounting
- **WHEN** processing completes
- **THEN** the summary contains total tokens and total time

### Requirement: Configurable parameters
The component SHALL support configuration via config for `teacher.endpoint`
(required), `teacher.model_name` (required), `teacher.temperature` (default 0),
`teacher.max_tokens` (default 512), `teacher.timeout` (default 60),
`teacher.max_retries` (default 3), `teacher.use_refined` (default true),
`teacher.filter_noisy` (default true), `teacher.filter_interpretive` (default
true), and `teacher.allow_additions` (default true).

#### Scenario: Custom temperature
- **WHEN** config sets `teacher.temperature=0.2`
- **THEN** teacher-model calls use that temperature

#### Scenario: Additions disabled
- **WHEN** config sets `teacher.allow_additions=false`
- **THEN** the teacher model does not add new theses, only filters and reformulates

### Requirement: Logging of refinement
The component SHALL log all refinement operations: load candidates, teacher-model
call, filter, reformulate, add, write artifact. The log SHALL be append-only.

#### Scenario: Operations logged
- **WHEN** any refinement operation occurs
- **THEN** it is appended to the log with a timestamp and the example id
