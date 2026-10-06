# Spec

## Purpose

Define the architectural contracts for dynamic-prompt-core: layered structure,
dependency boundaries, port policy, composition roots, public API policy,
contract evolution, and static-check enforcement. The specification fixes what
is an architectural contract (dependency boundaries and public APIs) versus what
is free to refactor (implementations, files, internal modules).

## Requirements

### Requirement: Layered project structure
The project SHALL have four layers: `domain`, `application`, `infrastructure`,
and `interfaces`. Each layer SHALL be a separate Python package under
`src/dynamic_prompt_core/`. Dependencies SHALL flow toward inner layers.
`domain` SHALL be independent of all other layers. `application` SHALL be
independent of `infrastructure` and `interfaces`. `infrastructure` SHALL be
independent of `interfaces`. `interfaces` MAY depend on `application` and
`infrastructure`.

#### Scenario: Correct layer direction
- **WHEN** `interfaces` imports `application`
- **THEN** the import is allowed

#### Scenario: Domain isolated
- **WHEN** `domain` imports `infrastructure`
- **THEN** the import is forbidden by the linter

#### Scenario: Application isolated from infrastructure
- **WHEN** `application` imports `infrastructure`
- **THEN** the import is forbidden by the linter

### Requirement: Infrastructure dependency restriction
`infrastructure` MAY depend on `domain` and `application.ports`, but SHALL NOT
depend on `application.use_cases` or `interfaces`. This rule SHALL be checked by
the linter.

#### Scenario: Infrastructure depends on ports
- **WHEN** `infrastructure/llm/` imports `application/ports/outbound/llm_client/`
- **THEN** the import is allowed

#### Scenario: Infrastructure depends on use case
- **WHEN** `infrastructure/llm/` imports `application/use_cases/run_cycle/`
- **THEN** the import is forbidden by the linter

### Requirement: Domain isolation
`domain` SHALL NOT import `application`, `infrastructure`, or `interfaces`. By
default `domain` SHALL depend only on the Python standard library. Use of
third-party libraries in `domain` SHALL be separately justified and explicitly
permitted by the architectural contract.

#### Scenario: Domain uses stdlib only
- **WHEN** `domain/models.py` imports `dataclasses` and `typing`
- **THEN** the check passes

#### Scenario: Domain uses third-party library
- **WHEN** `domain/services.py` imports `numpy`
- **THEN** the check fails unless `numpy` is explicitly permitted by the
  architectural contract

### Requirement: Ports
Ports SHALL reside in `application/ports/`. Ports SHALL define contracts for
external dependencies and SHALL NOT depend on `infrastructure`. Ports SHALL be
separated into `application/ports/inbound/` and `application/ports/outbound/`
subpackages. For new ports, `typing.Protocol` SHALL be used by default.
`abc.ABC` MAY be used only when runtime inheritance is required, with explicit
justification.

#### Scenario: Port uses Protocol
- **WHEN** a new port `llm_client` is created
- **THEN** `typing.Protocol` is used

#### Scenario: Port uses ABC
- **WHEN** a port requires runtime inheritance
- **THEN** `abc.ABC` MAY be used with explicit justification

#### Scenario: Port independent of infrastructure
- **WHEN** a port is defined
- **THEN** it does not import `infrastructure`

### Requirement: Dependency inversion
`application` SHALL depend on abstract ports, and `infrastructure` SHALL
implement these ports. Concrete infrastructure implementations SHALL NOT leak
into the application layer.

#### Scenario: Application depends on port
- **WHEN** a use case calls the LLM
- **THEN** it depends on `application.ports.outbound.llm_client`, not on
  `infrastructure.llm`

#### Scenario: Infrastructure implements port
- **WHEN** `infrastructure/llm/` implements `llm_client`
- **THEN** it imports the port and provides a concrete implementation

### Requirement: Sibling dependencies
Direct dependencies between modules of the same layer SHALL NOT be forbidden per
se. Cyclic dependencies and dependencies violating established architectural
boundaries are forbidden. Module granularity and decomposition SHALL be driven
by cohesion and responsibility, not by a formal ban on sibling imports.

#### Scenario: Sibling import allowed
- **WHEN** `infrastructure/llm/` imports `infrastructure/storage/`
- **THEN** the import is allowed if it does not violate other boundaries

#### Scenario: Cyclic sibling dependency forbidden
- **WHEN** `infrastructure/llm/` imports `infrastructure/storage/` and
  `infrastructure/storage/` imports `infrastructure/llm/`
- **THEN** the cycle is forbidden by the linter

### Requirement: Public module contracts
The public API of a package MAY be explicitly defined via `__init__.py` and
`__all__`. `__init__.py` SHALL be treated as a means of exposing public API, not
as a mandatory placement for implementations. `__all__` SHALL be required only
for packages that provide an explicit public API.

#### Scenario: Public API declared
- **WHEN** a package exposes a public API
- **THEN** `__init__.py` contains `__all__`

#### Scenario: Internal package
- **WHEN** a package is an internal implementation
- **THEN** `__all__` is not required

### Requirement: Internal implementation
Internal modules SHALL be considered implementation details. Code outside the
corresponding package boundary SHALL use its public API if one is defined.
Internal modules MAY import other internal modules within the allowed
architectural boundary.

#### Scenario: External code uses public API
- **WHEN** code outside a package accesses it
- **THEN** it uses the public API

#### Scenario: Internal imports allowed
- **WHEN** a module inside a package imports a sibling module of the same package
- **THEN** the import is allowed

### Requirement: No immutable files
Existing modules and `__init__.py` MAY be changed, fixed, and refactored.
Architectural immutability SHALL apply to dependency boundaries and public
contracts, not to physical files.

#### Scenario: File modified
- **WHEN** an edit is needed in an existing module
- **THEN** the edit is allowed if it does not violate architectural boundaries

#### Scenario: Contract preserved
- **WHEN** a module is modified
- **THEN** its public contract is preserved or explicitly evolved per the
  contract evolution policy

### Requirement: Refactoring policy
Adding functionality SHALL NOT require creating a new version of an existing
module. Existing implementations MAY be extended and refactored while preserving
architectural constraints. Unused modules MAY be removed.

#### Scenario: Feature added in place
- **WHEN** a new function is needed in a module
- **THEN** it MAY be added to the existing module without creating a new version

#### Scenario: Unused module removed
- **WHEN** a module is no longer used
- **THEN** it MAY be removed

### Requirement: Contract evolution
Backward-compatible changes to a public API MAY be made without creating a new
version. Breaking changes SHALL be accompanied by an explicit migration or
versioning policy. Versioning SHALL be applied to public contracts only when
truly necessary, not to every implementation change.

#### Scenario: Backward-compatible change
- **WHEN** a new function is added to a public API
- **THEN** no new version is created

#### Scenario: Breaking change
- **WHEN** the signature of an existing public function is changed
- **THEN** an explicit migration or versioning policy is required

### Requirement: Granular modules
Modules SHOULD have a single cohesive responsibility. This SHALL NOT be
interpreted as requiring exactly one function or one class per module. Artificial
splitting of modules SHALL be avoided.

#### Scenario: Cohesive module
- **WHEN** a module performs one logical function
- **THEN** it satisfies the requirement even if it contains multiple classes

#### Scenario: Artificial split avoided
- **WHEN** a module can remain whole without losing cohesion
- **THEN** it is not split

### Requirement: Composition roots
Each executable entrypoint SHALL have its own composition root.
`interfaces/cli/main.py` SHALL be the composition root for the CLI runtime.
`interfaces/server/main.py` SHALL be the composition root for the server runtime.
A composition root MAY use separate composition functions or modules when there
is shared or complex wiring logic.

#### Scenario: CLI composition root
- **WHEN** the CLI is started
- **THEN** `interfaces/cli/main.py` assembles dependencies

#### Scenario: Server composition root
- **WHEN** the server is started
- **THEN** `interfaces/server/main.py` assembles dependencies

### Requirement: CLI entrypoint
`interfaces/cli/main.py` SHALL parse arguments, load configuration, assemble
dependencies, manage runtime lifecycle, and invoke the application use case. The
entry point SHALL NOT contain business logic.

#### Scenario: CLI invocation
- **WHEN** the CLI is started
- **THEN** argument parsing, config loading, dependency assembly, and use-case
  invocation occur in order

#### Scenario: No business logic in CLI
- **WHEN** `interfaces/cli/main.py` is read
- **THEN** it contains no business rules

### Requirement: Server entrypoint
`interfaces/server/main.py` SHALL parse arguments, load configuration, assemble
server dependencies, start `ServerLauncher`, manage runtime lifecycle, and
handle signals. The entry point SHALL NOT contain business logic.

#### Scenario: Server invocation
- **WHEN** the server is started
- **THEN** argument parsing, dependency assembly, `ServerLauncher` start, and
  signal waiting occur in order

#### Scenario: No business logic in server
- **WHEN** `interfaces/server/main.py` is read
- **THEN** it contains no business rules

### Requirement: Thin entrypoints
Entry points SHALL contain only configuration, dependency composition, lifecycle
orchestration, and invocation. Business rules, application logic, persistence
logic, and infrastructure implementation logic SHALL reside outside the
entrypoint.

#### Scenario: Only orchestration
- **WHEN** an entrypoint is inspected
- **THEN** it contains only configuration, assembly, lifecycle, and invocation

#### Scenario: Business logic elsewhere
- **WHEN** a business rule is needed
- **THEN** it is implemented in `application` or `domain`

### Requirement: Graceful shutdown
Entry points SHALL handle SIGINT and SIGTERM via controlled cancellation or
stopping of the relevant runtime. Graceful shutdown SHALL wait for mandatory
cleanup operations to complete. State preservation policy, cleanup order, and
exit codes SHALL be defined by the application runtime contract.

#### Scenario: SIGINT handled
- **WHEN** SIGINT is received
- **THEN** the runtime stops via controlled cancellation

#### Scenario: Cleanup awaited
- **WHEN** graceful shutdown is performed
- **THEN** mandatory cleanup operations complete before exit

### Requirement: Use case contract
`run_cycle` SHALL reside in `application/use_cases/run_cycle/`. The use case
SHALL receive all necessary dependencies from outside and SHALL NOT create
concrete dependencies itself. The dependency contract SHALL be explicit and
typed. The use case SHALL return a defined execution result or terminate with an
exception.

#### Scenario: Use case receives dependencies
- **WHEN** `run_cycle` is called
- **THEN** all dependencies are passed from outside

#### Scenario: Use case returns result
- **WHEN** the use case completes successfully
- **THEN** a defined typed result is returned

### Requirement: Dependency object
When a use case has multiple dependencies, an explicit typed dependency object
SHALL be used instead of an unstructured service locator. The set of dependencies
SHALL be part of the use case contract.

#### Scenario: Typed dependency object
- **WHEN** a use case has multiple dependencies
- **THEN** they are passed via a typed object

#### Scenario: No service locator
- **WHEN** a use case receives dependencies
- **THEN** it does not use a service locator

### Requirement: Pydantic policy
`domain` SHOULD remain independent of framework-specific libraries. Pydantic
SHOULD be used at application or external boundaries for configuration,
validation, and serialization. Use of Pydantic inside `domain` SHALL be a
separate, deliberate architectural decision.

#### Scenario: Pydantic on boundaries
- **WHEN** input from the external world is validated
- **THEN** Pydantic is used at the application boundary

#### Scenario: Domain free of Pydantic
- **WHEN** a domain model is described
- **THEN** by default it does not use Pydantic

### Requirement: Testing policy
Integration and contract tests SHOULD use public APIs. Unit tests MAY import
internal implementation modules for isolated testing of specific components. The
test structure SHALL reflect architectural boundaries but SHALL NOT artificially
expand the production code's public API.

#### Scenario: Integration test uses public API
- **WHEN** integration with the LLM is tested
- **THEN** the test uses the public API

#### Scenario: Unit test uses internal module
- **WHEN** an isolated component is tested
- **THEN** the test MAY import an internal module

### Requirement: Import-linter scope
import-linter SHALL check only dependency and architectural boundaries: layer
direction, domain isolation, infrastructure restrictions, and the absence of
forbidden cyclic dependencies. It SHALL NOT be the sole mechanism for checking
all architectural requirements.

#### Scenario: Import-linter checks boundaries
- **WHEN** `lint-imports` is run
- **THEN** dependency boundaries are checked

#### Scenario: Other checks use other tools
- **WHEN** type checking is required
- **THEN** a type checker is used, not import-linter

### Requirement: Additional static checks
Other architectural invariants MAY be checked by specialized tools, including a
type checker (mypy), a formatter/linter (ruff), AST checks, tests, and CI
policies. Each type of constraint SHALL be checked by a tool suited to that
property.

#### Scenario: Types checked by mypy
- **WHEN** type checking is performed
- **THEN** mypy is used

#### Scenario: AST checks for custom rules
- **WHEN** an invariant unreachable by the linter is needed
- **THEN** an AST script is used

### Requirement: CI enforcement
CI SHALL run architectural checks before tests. A violation of mandatory
dependency boundaries SHALL cause failure. The specific CI pipeline SHALL remain
a separate task if CI already exists.

#### Scenario: Architecture checks first
- **WHEN** CI runs
- **THEN** architectural checks execute before tests

#### Scenario: Boundary violation fails CI
- **WHEN** a dependency boundary violation is detected
- **THEN** CI fails

### Requirement: Logging
Runtime logging configuration SHALL reside in `interfaces` or the composition
root. Concrete external logging handlers and integrations MAY reside in
`infrastructure`. `application` and `domain` SHALL use an abstract logging API
without configuration responsibility.

#### Scenario: Logging configured in entrypoint
- **WHEN** the runtime starts
- **THEN** logging configuration is performed in the entrypoint or composition
  root

#### Scenario: Domain uses abstract API
- **WHEN** `domain` writes a log
- **THEN** it uses an abstract logging API

### Requirement: Configuration
Configuration of architectural tools SHALL reside in `pyproject.toml`.
Application configuration SHALL reside in a separate application config file
(TOML format). Paths to the application config MAY be set via CLI arguments.
Application configuration SHALL NOT define architectural dependency rules.

#### Scenario: Tool config in pyproject
- **WHEN** import-linter is configured
- **THEN** the configuration is in `pyproject.toml`

#### Scenario: App config separate
- **WHEN** the application is configured
- **THEN** a separate config file is used

### Requirement: Architecture documentation
Architectural boundaries, dependency rules, public API policy, port policy,
composition roots, and contract evolution policy SHALL be unambiguously defined
in the specification. Fundamental architectural decisions SHALL NOT remain in
Open Questions.

#### Scenario: Boundaries defined
- **WHEN** the specification is read
- **THEN** all architectural boundaries are defined

#### Scenario: No open architectural questions
- **WHEN** the specification is finalized
- **THEN** fundamental decisions are not in Open Questions

### Requirement: Migration
Migration of existing code into the new architecture SHALL be performed as a
separate task. The architectural specification SHALL define the target structure
and rules, but SHALL NOT include migration details unless necessary to define
architectural contracts.

#### Scenario: Migration separate
- **WHEN** existing code needs to be moved
- **THEN** a separate task is created

#### Scenario: Architecture spec stays focused
- **WHEN** the architecture specification is read
- **THEN** it describes the target structure, not migration steps

### Requirement: Non-goals
Implementation of concrete use cases, beyond defining the `run_cycle` contract
and its invocation point, SHALL remain out of scope. Implementation of concrete
adapters SHALL remain out of scope, except for explicitly existing dependencies
(`ServerLauncher`, `AsyncTaskClient`). Migration of existing code, full test
implementation, and changes to existing CI MAY be performed as separate tasks.

#### Scenario: Use case implementation out of scope
- **WHEN** the architecture spec is created
- **THEN** use case implementations are not included

#### Scenario: Adapters out of scope
- **WHEN** the architecture spec is created
- **THEN** adapter implementations are not included

### Requirement: Architecture principle
Architectural stability SHALL be ensured by the stability of dependency
boundaries and public contracts, not by forbidding changes to files, modules, or
implementations. Internal implementation SHALL remain free for fixes, extension,
and refactoring while preserving established architectural contracts.

#### Scenario: Internal refactoring allowed
- **WHEN** internal implementation is refactored
- **THEN** it is allowed if contracts are preserved

#### Scenario: Boundary preserved
- **WHEN** an implementation is changed
- **THEN** dependency boundaries are preserved
