# Spec Delta

## Purpose

Defines the one-time structural migration of all root-level Python modules into
the layered `src/dynamic_prompt_core/` packages: concrete placement of each split
module, the port inventory, the composition roots, tooling contracts, and the
atomic removal of the old flat files.

## ADDED Requirements

### Requirement: Target structure
After migration the project SHALL have the following structure under
`src/dynamic_prompt_core/`: `domain/` with `models/`, `prompts/`, `errors/`;
`application/` with `ports/`, `schemas/`, `services/`, `use_cases/`;
`infrastructure/` with `llm/`, `server/`, `config/`, `storage/`, `nlp/`,
`embeddings/`, `data/`; `interfaces/` with `cli/`, `server/`. A `tests/` tree
SHALL contain `unit/`, `integration/`, `contract/`. All old root-level `.py`
files SHALL be removed.

#### Scenario: Structure created
- **WHEN** migration is complete
- **THEN** the directory tree matches the target structure

#### Scenario: Old files removed
- **WHEN** the project root is inspected
- **THEN** none of the old root-level `.py` files remain

### Requirement: Split and place asyncTask
`asyncTask.py` SHALL be split into `infrastructure/llm/`: enums (`Backend`,
`StructuredMode`) in `enums.py`; adapters (`_VLLMAdapter`, `_LlamaCppAdapter`,
`_OpenAIAdapter`) in `adapters.py`; `AsyncTask` in `client.py`; `RawResponse` in
`response.py`; errors (`HttpError`, `UnexpectedShapeError`) in `errors.py`.
Truncation, retry, and logging SHALL reside with `AsyncTask` in `client.py` or
in cohesive companion modules. A public API SHALL be exposed via
`infrastructure/llm/__init__.py` with `__all__`; adapters SHALL NOT be exported.

#### Scenario: AsyncTask available
- **WHEN** `AsyncTask` is imported
- **THEN** it is available through `infrastructure.llm`

#### Scenario: Adapters internal
- **WHEN** `infrastructure.llm.__all__` is read
- **THEN** the adapter classes are not listed

### Requirement: Split and place server_launcher
`server_launcher.py` SHALL be split: `load_config` and `ConfigError` into
`infrastructure/config/loader.py`; `build_argv`, `_build_llamacpp_argv`,
`_build_vllm_argv` into `infrastructure/server/argv.py`; `wait_for_ready` into
`infrastructure/server/readiness.py`; `ServerLauncher` and
`VllmNotSupportedError` into `infrastructure/server/launcher.py`; the CLI wrapper
`main` and signal handlers into `interfaces/server/main.py`. Public APIs SHALL be
exposed via `infrastructure/server/__init__.py` and
`infrastructure/config/__init__.py`.

#### Scenario: ServerLauncher available
- **WHEN** `ServerLauncher` is imported
- **THEN** it is available through `infrastructure.server`

#### Scenario: Config loader available
- **WHEN** `load_config` is imported
- **THEN** it is available through `infrastructure.config`

### Requirement: Place dataset
`dataset.py` SHALL be split: domain models (`Example`, `Split`) into
`domain/models/`; dev/holdout/ambiguous splitting logic into
`domain/services/dataset_split.py`; file loading (jsonl, parquet) into
`infrastructure/data/loader.py`; artifact saving into
`infrastructure/data/writer.py`; loading ports into
`application/ports/dataset_repository/`.

#### Scenario: Dataset loaded via infrastructure
- **WHEN** the application needs a dataset
- **THEN** loading goes through `infrastructure.data`

#### Scenario: Splitting is domain logic
- **WHEN** splitting is performed
- **THEN** the logic resides in `domain.services`

### Requirement: Place prompts
`prompts.py` SHALL be split: base prompt layers (role, task, output contract,
fallback) into `domain/prompts/base.py`; Pydantic schemas for structured output
into `application/schemas/`; prompt rendering to text into
`application/services/prompt_render.py`; fixed prompts (thesis extraction, rule
formulation) into `domain/prompts/fixed.py`.

#### Scenario: Base layers in domain
- **WHEN** base layers are read
- **THEN** they reside in `domain.prompts`

#### Scenario: Rendering in application
- **WHEN** a prompt is rendered
- **THEN** rendering resides in `application.services`

### Requirement: Place schemas
`schemas.py` SHALL be placed in `application/schemas/`. Pydantic models for
structured output SHALL be grouped by purpose: `classification.py`, `thesis.py`,
`refinement.py`.

#### Scenario: Schemas available
- **WHEN** the runner needs a classification schema
- **THEN** it is imported from `application.schemas.classification`

### Requirement: Place metrics
`metrics.py` SHALL be placed in `application/services/metrics/` because it uses
numpy and/or scikit-learn, which are forbidden in `domain`. It SHALL be split:
accuracy, F1, confusion matrix into `classification.py`; grouped metrics (by
text length) into `grouped.py`; version comparison into `comparison.py`.

#### Scenario: Metrics computed in application
- **WHEN** metrics are computed
- **THEN** they reside in `application.services.metrics`

#### Scenario: Domain stays free
- **WHEN** `domain` is imported
- **THEN** it does not pull in numpy or scikit-learn

### Requirement: Place normalize
`normalize.py` SHALL be split: a normalizer port into
`application/ports/normalizer/`; the pymorphy3 / spaCy implementation into
`infrastructure/nlp/normalizer.py`.

#### Scenario: Normalizer via port
- **WHEN** a use case needs normalization
- **THEN** it depends on `application.ports.normalizer`

#### Scenario: Implementation in infrastructure
- **WHEN** normalization is performed
- **THEN** `infrastructure.nlp` is used

### Requirement: Place runner
`runner.py` SHALL be placed in `application/use_cases/run_baseline/`. The use
case SHALL obtain `AsyncTask` through the `llm_client` port, not construct it
directly. Normalization SHALL go through the `normalizer` port. Result saving
SHALL go through the `run_repository` port.

#### Scenario: Runner as use case
- **WHEN** a baseline run is started
- **THEN** `application.use_cases.run_baseline` is invoked

#### Scenario: Dependencies via ports
- **WHEN** the runner executes
- **THEN** it uses ports, not concrete implementations

### Requirement: Place thesis_analyzer
`thesis_analyzer.py` SHALL be placed in `application/use_cases/analyze_theses/`.
Embeddings SHALL go through the `embedding_client` port. Clustering (cosine,
manhattan) SHALL reside in `application/services/clustering/` since it requires
numpy.

#### Scenario: Analyzer as use case
- **WHEN** theses are analyzed
- **THEN** `application.use_cases.analyze_theses` is invoked

#### Scenario: Clustering location
- **WHEN** clustering is performed
- **THEN** it resides in `application.services.clustering`

### Requirement: Place rule_candidate_selector
`rule_candidate_selector.py` SHALL be split: selection logic (frequency,
precision, top-N) into `domain/services/rule_selection.py`; the use case wrapper
into `application/use_cases/select_candidates/`.

#### Scenario: Selection logic in domain
- **WHEN** a frequency filter is applied
- **THEN** the logic resides in `domain.services.rule_selection`

#### Scenario: Use case wraps selection
- **WHEN** candidate selection is started
- **THEN** `application.use_cases.select_candidates` is invoked

### Requirement: Place prompt_composer
`prompt_composer.py` SHALL be placed in `application/use_cases/compose_prompt/`.
Rule formulation via LLM SHALL go through the `llm_client` port. Distortion
checking (cosine with centroid) SHALL use `application.services.clustering` or
`domain.services.similarity`.

#### Scenario: Composer as use case
- **WHEN** a new prompt version is composed
- **THEN** `application.use_cases.compose_prompt` is invoked

#### Scenario: Distortion check uses service
- **WHEN** distortion is checked
- **THEN** a similarity service is used

### Requirement: Place version_comparator
`version_comparator.py` SHALL be placed in
`application/use_cases/compare_versions/`. The accept/rollback decision rule
SHALL reside in `domain/services/decision.py`. The rollback counter SHALL be
part of the use case state.

#### Scenario: Comparator as use case
- **WHEN** versions are compared
- **THEN** `application.use_cases.compare_versions` is invoked

#### Scenario: Decision rule in domain
- **WHEN** a decision is taken
- **THEN** `domain.services.decision` is used

### Requirement: Place prompt_store
`prompt_store.py` SHALL be split: the repository port into
`application/ports/prompt_repository/`; the JSON store implementation into
`infrastructure/storage/prompt_store.py`.

#### Scenario: Store via port
- **WHEN** a use case accesses the prompt store
- **THEN** it uses `application.ports.prompt_repository`

#### Scenario: Implementation in infrastructure
- **WHEN** the store reads or writes a file
- **THEN** `infrastructure.storage.prompt_store` does it

### Requirement: Place thesis_refiner
`thesis_refiner.py` SHALL be placed in
`application/use_cases/refine_theses/`. The teacher model SHALL be called through
a dedicated `teacher_llm_client` port. Filtering and reformulation SHALL reside
in the use case.

#### Scenario: Refiner as use case
- **WHEN** refinement is performed
- **THEN** `application.use_cases.refine_theses` is invoked

#### Scenario: Teacher via port
- **WHEN** the teacher model is called
- **THEN** `application.ports.teacher_llm_client` is used

### Requirement: Place cross_task_transfer
`cross_task_transfer.py` SHALL be placed in
`application/use_cases/cross_task_transfer/`. Base layer and thesis transfer
logic SHALL reside in the use case. Comparison with cold start SHALL go through
`application/use_cases/run_cycle/`.

#### Scenario: Transfer as use case
- **WHEN** transfer is performed
- **THEN** `application.use_cases.cross_task_transfer` is invoked

### Requirement: Place cycle_orchestrator
`cycle_orchestrator.py` SHALL be placed in `application/use_cases/run_cycle/`.
This is the central use case that invokes the others. Dependencies SHALL be
passed through a typed dependency object.

#### Scenario: Orchestrator as use case
- **WHEN** a cycle is started
- **THEN** `application.use_cases.run_cycle` is invoked

#### Scenario: Dependency object
- **WHEN** the orchestrator executes
- **THEN** it receives dependencies through a typed object

### Requirement: Place cycle_observer
`cycle_observer.py` SHALL be split: artifact reading and aggregation logic into
`application/use_cases/observe_cycle/`; output formatting (markdown, json) into
`interfaces/cli/observer.py`; the observer CLI command into
`interfaces/cli/main.py`.

#### Scenario: Observer reads artifacts
- **WHEN** the observer collects data
- **THEN** a use case in `application.use_cases` does it

#### Scenario: Formatting in interface
- **WHEN** a report is formatted as markdown
- **THEN** `interfaces.cli.observer` does it

### Requirement: Place convert_csv_to_jsonl
`convert_csv_to_jsonl.py` SHALL be split: conversion logic into
`infrastructure/data/csv_to_jsonl.py`; the CLI command into
`interfaces/cli/convert.py`.

#### Scenario: Conversion in infrastructure
- **WHEN** conversion is performed
- **THEN** the logic resides in `infrastructure.data`

#### Scenario: CLI for conversion
- **WHEN** conversion is started from the CLI
- **THEN** `interfaces.cli.convert` is invoked

### Requirement: Place smoke_test
`smoke_test.py` SHALL be moved to `tests/integration/test_smoke.py`. The test
SHALL use the public APIs of `infrastructure.llm` and `infrastructure.server`
only; direct imports of internal modules SHALL NOT appear.

#### Scenario: Smoke test migrated
- **WHEN** `pytest tests/integration` is run
- **THEN** the smoke test executes

#### Scenario: Public API only
- **WHEN** the smoke test is read
- **THEN** it imports only through `infrastructure.llm` and `infrastructure.server`

### Requirement: Application ports
Ports SHALL be created in `application/ports/` for all external dependencies:
`llm_client` (LLM calls: extraction, classification, batch),
`teacher_llm_client` (teacher model calls), `embedding_client` (embeddings),
`normalizer` (thesis normalization), `prompt_repository` (prompt version store),
`run_repository` (run result store), `dataset_repository` (dataset load/save).
Each port SHALL use `typing.Protocol`.

#### Scenario: All ports defined
- **WHEN** `application.ports` is read
- **THEN** all listed ports are present

#### Scenario: Protocols used
- **WHEN** a port is created
- **THEN** `typing.Protocol` is used

### Requirement: Composition roots
`interfaces/cli/main.py` SHALL be the composition root for the CLI runtime:
assembles `AsyncTask`, `JsonPromptStore`, `Normalizer`, `EmbeddingClient`,
`RunRepository` and invokes `run_cycle`. `interfaces/server/main.py` SHALL be the
composition root for the server runtime: assembles `ServerLauncher` and handles
signals.

#### Scenario: CLI wires dependencies
- **WHEN** the CLI starts
- **THEN** all dependencies are assembled in `main.py`

#### Scenario: Server wires dependencies
- **WHEN** the server starts
- **THEN** `ServerLauncher` is assembled in `interfaces/server/main.py`

### Requirement: Logging
`logging.basicConfig` SHALL be called only in `interfaces/cli/main.py` and
`interfaces/server/main.py`. Modules in `domain`, `application`, and
`infrastructure` SHALL use `logging.getLogger(__name__)` without configuration.

#### Scenario: Logging at entrypoint
- **WHEN** a runtime starts
- **THEN** `basicConfig` is called in the entrypoint

#### Scenario: No logging config in library
- **WHEN** `infrastructure.llm` is imported
- **THEN** no logging configuration is performed

### Requirement: Public API of each module
Each package exposing a public API SHALL have `__init__.py` with `__all__`.
Internal implementation modules MAY omit `__all__`. Existing public API signatures
(`AsyncTask`, `ServerLauncher`, `load_config`, `run_cycle`, and others) SHALL NOT
change.

#### Scenario: Public API preserved
- **WHEN** `AsyncTask.analyze` is called
- **THEN** the signature matches the original

#### Scenario: all present
- **WHEN** `infrastructure/llm/__init__.py` is read
- **THEN** it contains `__all__`

### Requirement: Config and pyproject
`config.toml` SHALL remain at the project root. `pyproject.toml` SHALL contain
package metadata, dependencies (aiohttp, pydantic, transformers, numpy,
pymorphy3, spacy, scikit-learn), dev dependencies (pytest, pytest-asyncio,
import-linter, mypy, ruff), and configuration for import-linter, ruff, mypy, and
pytest.

#### Scenario: Config in root
- **WHEN** the application starts
- **THEN** `config.toml` is read from the root or a specified path

#### Scenario: Tool config in pyproject
- **WHEN** `lint-imports` is run
- **THEN** the configuration is read from `pyproject.toml`

### Requirement: Import-linter contracts
`pyproject.toml` SHALL contain import-linter contracts: layered
(interfaces → infrastructure → application → domain); domain isolation (domain
does not import application, infrastructure, interfaces); infrastructure
restriction (infrastructure does not import `application.use_cases` or
interfaces); no cyclic imports.

#### Scenario: Layer contract enforced
- **WHEN** `application` imports `infrastructure`
- **THEN** `lint-imports` fails

#### Scenario: Domain contract enforced
- **WHEN** `domain` imports `numpy`
- **THEN** `lint-imports` fails unless numpy is explicitly permitted

### Requirement: Test migration
All existing tests SHALL be moved into `tests/` mirroring the `src/` structure.
Unit tests MAY import internal modules. Integration and contract tests SHALL use
only public APIs.

#### Scenario: Tests mirror structure
- **WHEN** `infrastructure/llm/client.py` exists
- **THEN** `tests/unit/infrastructure/llm/test_client.py` exists

#### Scenario: Integration uses public API
- **WHEN** integration is tested
- **THEN** public APIs are used

### Requirement: Migration is atomic
Migration SHALL be performed in a separate git branch. Old files SHALL be removed
only after the new modules are ready and pass all checks. Merge into the main
branch SHALL happen after acceptance criteria pass.

#### Scenario: Migration in branch
- **WHEN** migration is performed
- **THEN** changes are in a separate branch

#### Scenario: Atomic removal
- **WHEN** old files are removed
- **THEN** the new modules already work

### Requirement: Documentation update
`README.md` SHALL be updated: run examples SHALL use the new paths
(`python -m dynamic_prompt_core.interfaces.cli.main`,
`python -m dynamic_prompt_core.interfaces.server.main`). References to old files
SHALL be removed.

#### Scenario: README uses new paths
- **WHEN** the README is read
- **THEN** examples use the new entrypoints

#### Scenario: No stale references
- **WHEN** the project is checked
- **THEN** no mentions of old files remain except historical notes
