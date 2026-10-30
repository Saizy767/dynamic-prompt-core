# Design

## Context

The repository currently holds 19 root-level `.py` modules implementing the full
pipeline (LLM transport, server launcher, dataset, prompts, schemas, metrics,
normalization, runner, thesis analyzer, rule selector, version comparator,
prompt composer, prompt store, cycle orchestrator, cycle observer, thesis
refiner, cross-task transfer, CSV converter, smoke test). A partial
`src/dynamic_prompt_core/` scaffold already exists with the four layer packages
and a few placeholders (`application/ports/outbound/llm_client.py`,
`application/use_cases/run_cycle/`, `interfaces/cli/main.py`,
`interfaces/server/main.py`), but no real logic has been migrated. The
`architecture` spec fixes the target contracts; `pyproject.toml` already
configures import-linter, mypy, and ruff against the layered package. The
migration is structural: logic is preserved, only placement, granularity, and
entrypoints change.

## Goals / Non-Goals

**Goals:**
- Move every root-level module into the correct layer with the granularity
  specified in the spec delta.
- Introduce the full port inventory (`typing.Protocol`) so use cases depend on
  abstractions, not infrastructure.
- Make `interfaces/cli/main.py` and `interfaces/server/main.py` the only
  composition roots; confine `logging.basicConfig` to them.
- Preserve all existing public signatures via `__init__.py` + `__all__`.
- Get `lint-imports`, `mypy`, `ruff`, and `pytest` green against the layered
  tree.
- Remove the old flat files atomically once the new tree passes.

**Non-Goals:**
- Changing module logic — this is a move/split, not a rewrite.
- Implementing new use cases, adapters, or features beyond what exists.
- Setting up CI (separate task).
- Rewriting tests beyond import/path adaptation.

## Decisions

### Decision: Clustering and metrics go to `application/services/`
Clustering (`thesis_analyzer`) and metrics both require numpy/sklearn, which the
`architecture` spec forbids in `domain` unless explicitly permitted. Rather than
carve an exception that weakens the domain isolation contract, both live in
`application/services/clustering/` and `application/services/metrics/`. Pure
decision rules that need no third-party libs (e.g. accept/rollback) still go to
`domain/services/decision.py`.
- **Alternative considered**: permit numpy in `domain` via an import-linter
  whitelist. Rejected because it erodes the domain isolation contract and makes
  the linter config a per-module exception list.

### Decision: Domain models use dataclasses, not Pydantic
Per the `architecture` Pydantic policy, `domain/models/` uses stdlib
`dataclasses`. Pydantic stays at the application boundary in
`application/schemas/` for structured-output validation and serialization.
- **Alternative considered**: Pydantic everywhere for uniform validation.
  Rejected to keep `domain` framework-free.

### Decision: Fixed prompts live in `domain/prompts/fixed.py`
Base layers and fixed prompts are domain knowledge (role, task, output contract,
fallback, thesis-extraction template, rule-formulation template). They live in
`domain/prompts/`. Only rendering (which depends on schemas/templating) lives in
`application/services/prompt_render.py`.
- **Alternative considered**: fixed prompts in `application/prompts/`. Rejected
  because they are stable domain facts, not orchestration.

### Decision: Separate `embedding_client` port
Embeddings are a distinct external dependency with a different lifecycle and
provider than the LLM client. A dedicated `embedding_client` port keeps the
`llm_client` contract focused on extraction/classification/batch calls.
- **Alternative considered**: fold embeddings into `llm_client`. Rejected to
  avoid a bloated port and to allow swapping the embedding provider
  independently.

### Decision: `convert_csv_to_jsonl` splits into infrastructure + CLI
Conversion logic is infrastructure (`infrastructure/data/csv_to_jsonl.py`); the
CLI wrapper is `interfaces/cli/convert.py`. This keeps the CLI thin and the logic
reusable from other use cases.
- **Alternative considered**: CLI-only script. Rejected because it duplicates
  logic if conversion is ever called from a use case.

### Decision: Smoke test becomes a pytest integration test
`smoke_test.py` moves to `tests/integration/test_smoke.py` and imports only
`infrastructure.llm` and `infrastructure.server` public APIs. This makes it part
of the standard `pytest` run instead of a standalone script.
- **Alternative considered**: keep as a standalone script. Rejected because it
  bypasses the test runner and import-linter coverage.

### Decision: Migration branch `migration/architecture`
All work happens in a dedicated branch. Old files are deleted in a single commit
after the new tree passes `lint-imports`, `mypy`, `ruff`, and `pytest`. Merge to
main only after acceptance criteria pass.

### Decision: Port subpackage layout
New ports are placed under `application/ports/outbound/` (for dependencies the
application calls out to: `llm_client`, `teacher_llm_client`,
`embedding_client`, `normalizer`, `prompt_repository`, `run_repository`,
`dataset_repository`), consistent with the existing
`application/ports/outbound/llm_client.py` and the `architecture` spec's
inbound/outbound split. The spec delta's `application/ports/<name>/` shorthand
resolves to `application/ports/outbound/<name>/` for these outbound
dependencies.

## Risks / Trade-offs

- **[Import churn breaks tests]** → Migrate test imports in the same commit as
  each module move; run `pytest` per milestone.
- **[Hidden cross-module coupling in monoliths]** → When splitting a file, trace
  its internal references first; if a split would create a layer violation,
  introduce a port rather than forcing the import.
- **[Public API drift]** → Re-export the original names through `__init__.py`
  `__all__` and add a signature-preservation check (smoke test covers
  `AsyncTask.analyze`, `ServerLauncher`, `load_config`, `run_cycle`).
- **[import-linter false positives during transition]** → Keep the old flat files
  out of the layered package during migration; only delete after the layered tree
  is green so the linter never sees a half-migrated state.
- **[Large diff hinders review]** → Group commits by source file (one commit per
  module migration + its tests); the atomic deletion is the final commit.

## Migration Plan

1. Create the full target directory tree (empty `__init__.py` files).
2. Migrate modules in dependency order: `infrastructure` first (llm, server,
   config, nlp, storage, data, embeddings), then `domain` (models, prompts,
   errors, services), then `application` (ports, schemas, services, use_cases),
   then `interfaces` (cli, server).
3. After each module group, run `ruff` + `mypy` on the new files.
4. Migrate tests into `tests/` mirroring `src/`, adapting imports.
5. Wire composition roots; confine `logging.basicConfig` to entrypoints.
6. Run `lint-imports`, `mypy`, `ruff`, `pytest` end-to-end.
7. Delete all 19 root-level `.py` files in one commit.
8. Update `README.md` entrypoint examples.
9. Merge `migration/architecture` to main after acceptance criteria pass.

**Rollback**: Until step 7, the old files still exist and the branch can be
discarded with no impact on main. After deletion, revert the deletion commit.
