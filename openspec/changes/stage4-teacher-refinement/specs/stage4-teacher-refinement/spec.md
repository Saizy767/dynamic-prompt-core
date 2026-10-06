# Spec Delta

## ADDED Requirements

### Requirement: Teacher model port
The component SHALL define a `TeacherLLMClient` port in
`application/ports/outbound/teacher_llm_client/`. The port SHALL define the
contract for teacher-model refinement calls: a `review_theses` method that
accepts a text and a list of candidate theses and returns a structured
refinement plan (keep, reformulate, drop, add). The port SHALL use
`typing.Protocol`. The port SHALL NOT depend on `infrastructure`.

#### Scenario: Port used by use case
- **WHEN** the refinement use case is invoked
- **THEN** it depends on `application.ports.outbound.teacher_llm_client`, not on
  `infrastructure`

#### Scenario: Port uses Protocol
- **WHEN** the port is defined
- **THEN** `typing.Protocol` is used

#### Scenario: Port independent of infrastructure
- **WHEN** the port module is imported
- **THEN** it does not import `infrastructure`

### Requirement: Teacher client implementation
The `TeacherLLMClient` port SHALL be implemented in
`infrastructure/llm/teacher_client.py`. The implementation SHALL use `AsyncTask`
from `infrastructure.llm` for teacher-model calls. The teacher model SHALL use a
separate endpoint, distinct from the working model's endpoint. The
implementation SHALL NOT contain refinement business logic — only transport,
prompt assembly, and response parsing.

#### Scenario: Teacher client uses AsyncTask
- **WHEN** the teacher model is called
- **THEN** an `AsyncTask` instance pointed at the teacher endpoint is used

#### Scenario: Separate endpoint
- **WHEN** config sets `teacher.endpoint`
- **THEN** it is distinct from the working model's `[llm]` endpoint

#### Scenario: No business logic in client
- **WHEN** `infrastructure/llm/teacher_client.py` is inspected
- **THEN** it contains no filtering, reformulation, or refinement decisions

### Requirement: Use case contract
The refinement use case SHALL reside in `application/use_cases/refine_theses/`.
The use case SHALL receive all dependencies through a typed dependency object
(`RefineThesesDeps`). The use case SHALL NOT create concrete implementations
itself. The use case SHALL return a defined result (processing statistics) or
terminate with an exception.

#### Scenario: Use case invoked with dependency object
- **WHEN** refinement is started
- **THEN** `application.use_cases.refine_theses` is called with a
  `RefineThesesDeps` dependency object

#### Scenario: No concrete dependencies created
- **WHEN** the use case executes
- **THEN** it uses ports, not concrete infrastructure implementations

#### Scenario: Use case returns result
- **WHEN** the use case completes successfully
- **THEN** a defined typed result with processing statistics is returned

### Requirement: Composition root wiring
The `TeacherClient` implementation SHALL be assembled in the composition root
(`interfaces/cli/main.py`). No other module SHALL instantiate `TeacherClient`
directly. The composition root SHALL pass the `TeacherClient` through the
`RefineThesesDeps` dependency object.

#### Scenario: Wired in composition root
- **WHEN** the CLI is started
- **THEN** `TeacherClient` is created in `interfaces/cli/main.py`

#### Scenario: No wiring elsewhere
- **WHEN** the use case executes
- **THEN** it receives `teacher_llm_client` through the dependency object, not
  by constructing it

### Requirement: Domain purity
Pure refinement logic — applying a refinement plan to candidate theses (keep,
reformulate with length enforcement, drop with reason, add with source tagging)
— SHALL reside in `domain/services/thesis_refinement.py` when it does not
depend on framework-specific libraries. Interaction with the teacher model
SHALL occur through the port in the application layer.

#### Scenario: Pure logic in domain
- **WHEN** a refinement plan is applied to candidate theses
- **THEN** the logic resides in `domain.services.thesis_refinement`

#### Scenario: LLM calls via port
- **WHEN** the teacher model is called
- **THEN** the call goes through `application.ports.outbound.teacher_llm_client`

#### Scenario: Domain free of framework imports
- **WHEN** `domain/services/thesis_refinement.py` is inspected
- **THEN** it does not import `infrastructure`, `aiohttp`, or `pydantic`

## MODIFIED Requirements

### Requirement: Configurable use of refined theses
The component SHALL support a config setting for whether downstream stages use
`theses_refined` or `theses_raw_tiny`. The setting SHALL be available via config.
By default, `theses_refined` SHALL be used when the artifact is available,
otherwise `theses_raw_tiny`. When `teacher.use_refined` is set to `false`,
downstream stages SHALL use `theses_raw_tiny` even when the refined artifact is
available.

#### Scenario: Refined used by default
- **WHEN** the `theses_refined` artifact exists
- **THEN** downstream stages use `theses_refined`

#### Scenario: Fallback to tiny
- **WHEN** the `theses_refined` artifact is absent
- **THEN** downstream stages use `theses_raw_tiny`

#### Scenario: Explicit override
- **WHEN** config sets `teacher.use_refined=false`
- **THEN** downstream stages use `theses_raw_tiny`, even when the refined
  artifact is available

### Requirement: Logging of refinement
The component SHALL log all refinement operations: load candidates, teacher-model
call, filter, reformulate, add, write artifact. The log SHALL be append-only.
The component SHALL NOT call `logging.basicConfig`; logging configuration SHALL
reside in the composition root or entrypoint only.

#### Scenario: Operations logged
- **WHEN** any refinement operation occurs
- **THEN** it is appended to the log with a timestamp and the example id

#### Scenario: No logging configuration in component
- **WHEN** a refinement module is imported
- **THEN** `logging.basicConfig` is not called
