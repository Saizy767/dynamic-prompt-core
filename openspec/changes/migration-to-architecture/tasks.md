# Tasks

## 1. Branch and scaffold

- [x] 1.1 Create git branch `migration/architecture` from main and verify `git branch --show-current` prints it
- [x] 1.2 Create the full target directory tree under `src/dynamic_prompt_core/` with empty `__init__.py` for each package (`domain/{models,prompts,errors}`, `application/{ports/{inbound,outbound},schemas,services/{metrics,clustering},use_cases}`, `infrastructure/{llm,server,config,storage,nlp,embeddings,data}`, `interfaces/{cli,server}`) and verify the tree matches the spec's Requirement: Target structure
- [x] 1.3 Create `tests/{unit,integration,contract}/` with `__init__.py` and verify `pytest --collect-only` finds the roots

## 2. Infrastructure — LLM transport

- [x] 2.1 Split `asyncTask.py` into `infrastructure/llm/enums.py` (`Backend`, `StructuredMode`), `adapters.py` (`_VLLMAdapter`, `_LlamaCppAdapter`, `_OpenAIAdapter`), `client.py` (`AsyncTask` + truncation/retry/logging), `response.py` (`RawResponse`), `errors.py` (`HttpError`, `UnexpectedShapeError`) and verify `ruff check` + `mypy` pass on the new files
- [x] 2.2 Write `infrastructure/llm/__init__.py` with `__all__` exporting `AsyncTask`, `RawResponse`, `Backend`, `StructuredMode` (not adapters) and verify `from dynamic_prompt_core.infrastructure.llm import AsyncTask` succeeds and adapters are absent from `__all__`

## 3. Infrastructure — server and config

- [x] 3.1 Split `server_launcher.py` config parts into `infrastructure/config/loader.py` (`load_config`, `ConfigError`) and verify `from dynamic_prompt_core.infrastructure.config import load_config` succeeds
- [x] 3.2 Split `server_launcher.py` server parts into `infrastructure/server/argv.py` (`build_argv`, `_build_llamacpp_argv`, `_build_vllm_argv`), `readiness.py` (`wait_for_ready`), `launcher.py` (`ServerLauncher`, `VllmNotSupportedError`) and verify `from dynamic_prompt_core.infrastructure.server import ServerLauncher` succeeds
- [x] 3.3 Write `infrastructure/server/__init__.py` and `infrastructure/config/__init__.py` with `__all__` and verify both imports from 3.1 and 3.2 work through the package

## 4. Infrastructure — NLP, storage, data, embeddings

- [x] 4.1 Move `normalize.py` implementation to `infrastructure/nlp/normalizer.py` (pymorphy3/spaCy) and verify `ruff check` + `mypy` pass
- [x] 4.2 Move `prompt_store.py` JSON implementation to `infrastructure/storage/prompt_store.py` and verify it imports the `prompt_repository` port (created in group 6) rather than being imported by application
- [x] 4.3 Split `dataset.py` file loading into `infrastructure/data/loader.py` (jsonl, parquet) and artifact saving into `infrastructure/data/writer.py` and verify `ruff check` + `mypy` pass
- [x] 4.4 Move `convert_csv_to_jsonl.py` conversion logic to `infrastructure/data/csv_to_jsonl.py` and verify a round-trip read of a converted jsonl matches the source csv

## 5. Domain — models, prompts, errors, services

- [x] 5.1 Move `dataset.py` domain models (`Example`, `Split`) to `domain/models/` as stdlib dataclasses and verify they import no third-party libraries
- [x] 5.2 Split `prompts.py` base layers into `domain/prompts/base.py` and fixed prompts into `domain/prompts/fixed.py` and verify both import only stdlib
- [x] 5.3 Move `dataset.py` splitting logic to `domain/services/dataset_split.py` and verify it imports no third-party libraries
- [x] 5.4 Move `rule_candidate_selector.py` selection logic (frequency, precision, top-N) to `domain/services/rule_selection.py` and verify unit tests for the filters pass
- [x] 5.5 Move `version_comparator.py` accept/rollback decision rule to `domain/services/decision.py` and verify it imports no third-party libraries
- [x] 5.6 Add `domain/errors/` for any domain-specific errors split out of the migrated modules and verify `domain` imports no application/infrastructure/interfaces modules (`lint-imports` domain-isolated contract)

## 6. Application — ports

- [x] 6.1 Define `application/ports/outbound/llm_client.py` as `typing.Protocol` (extraction, classification, batch) and verify `mypy` accepts it
- [x] 6.2 Define `application/ports/outbound/teacher_llm_client.py` as `typing.Protocol` and verify it is distinct from `llm_client`
- [x] 6.3 Define `application/ports/outbound/embedding_client.py` as `typing.Protocol` and verify it is distinct from `llm_client`
- [x] 6.4 Define `application/ports/outbound/normalizer.py` as `typing.Protocol` and verify `infrastructure/nlp/normalizer.py` satisfies it structurally
- [x] 6.5 Define `application/ports/outbound/prompt_repository.py` as `typing.Protocol` and verify `infrastructure/storage/prompt_store.py` satisfies it structurally
- [x] 6.6 Define `application/ports/outbound/run_repository.py` as `typing.Protocol` for run result storage and verify `mypy` accepts it
- [x] 6.7 Define `application/ports/outbound/dataset_repository.py` as `typing.Protocol` for dataset load/save and verify `mypy` accepts it

## 7. Application — schemas and services

- [x] 7.1 Move `schemas.py` to `application/schemas/` split by purpose: `classification.py`, `thesis.py`, `refinement.py` and verify `from dynamic_prompt_core.application.schemas.classification import *` works
- [x] 7.2 Move `prompts.py` rendering to `application/services/prompt_render.py` and verify a rendered prompt matches the original output for a sample input
- [x] 7.3 Split `metrics.py` into `application/services/metrics/classification.py` (accuracy, F1, confusion), `grouped.py` (by text length), `comparison.py` (version comparison) and verify `domain` does not import numpy/sklearn (`lint-imports` domain contract)
- [x] 7.4 Move clustering logic from `thesis_analyzer.py` to `application/services/clustering/` (cosine, manhattan) and verify it uses numpy but resides outside `domain`

## 8. Application — use cases

- [x] 8.1 Move `runner.py` to `application/use_cases/run_baseline/` depending on `llm_client`, `normalizer`, `run_repository` ports and verify it imports no infrastructure directly
- [x] 8.2 Move `thesis_analyzer.py` to `application/use_cases/analyze_theses/` depending on `embedding_client` and the clustering service and verify it imports no infrastructure directly
- [x] 8.3 Wrap `rule_candidate_selector.py` as `application/use_cases/select_candidates/` calling `domain/services/rule_selection.py` and verify the use case imports no infrastructure
- [x] 8.4 Move `prompt_composer.py` to `application/use_cases/compose_prompt/` using `llm_client` and the similarity/clustering service and verify it imports no infrastructure directly
- [x] 8.5 Move `version_comparator.py` to `application/use_cases/compare_versions/` with rollback counter state, calling `domain/services/decision.py` and verify it imports no infrastructure directly
- [x] 8.6 Move `thesis_refiner.py` to `application/use_cases/refine_theses/` using `teacher_llm_client` and verify it imports no infrastructure directly
- [x] 8.7 Move `cross_task_transfer.py` to `application/use_cases/cross_task_transfer/` and verify it calls `run_cycle` for cold-start comparison
- [x] 8.8 Move `cycle_orchestrator.py` to `application/use_cases/run_cycle/` with a typed dependency object (`run_cycle_deps.py`) and verify `run_cycle` receives all dependencies via the object, not a service locator
- [x] 8.9 Move `cycle_observer.py` aggregation to `application/use_cases/observe_cycle/` and verify it imports no infrastructure directly

## 9. Interfaces — composition roots

- [x] 9.1 Write `interfaces/cli/main.py` as CLI composition root: parse args, `load_config`, assemble `AsyncTask`, `JsonPromptStore`, `Normalizer`, `EmbeddingClient`, `RunRepository`, call `run_cycle`; verify it contains no business logic and `python -m dynamic_prompt_core.interfaces.cli.main --help` runs
- [x] 9.2 Write `interfaces/server/main.py` as server composition root: parse args, assemble `ServerLauncher`, handle SIGINT/SIGTERM, wait; verify `python -m dynamic_prompt_core.interfaces.server.main --help` runs
- [x] 9.3 Move `cycle_observer.py` formatting to `interfaces/cli/observer.py` (markdown, json) and observer CLI command into `interfaces/cli/main.py` and verify a sample observer invocation produces markdown output
- [x] 9.4 Move `convert_csv_to_jsonl.py` CLI wrapper to `interfaces/cli/convert.py` and verify `python -m dynamic_prompt_core.interfaces.cli.convert --help` runs
- [x] 9.5 Ensure `logging.basicConfig` is called only in `interfaces/cli/main.py` and `interfaces/server/main.py`; verify `rg "basicConfig" src/` returns only those two files

## 10. Tooling configuration

- [x] 10.1 Update `pyproject.toml` dependencies (aiohttp, pydantic, transformers, numpy, pymorphy3, spacy, scikit-learn) and dev dependencies (pytest, pytest-asyncio, import-linter, mypy, ruff) and verify `pip install -e ".[dev]"` succeeds
- [x] 10.2 Add/confirm import-linter contracts in `pyproject.toml` (layer-direction, domain-isolated, application-isolated, infrastructure-isolated, no-cyclic-imports) and verify `lint-imports` passes
- [x] 10.3 Add pytest configuration to `pyproject.toml` (testpaths, asyncio_mode) and verify `pytest --collect-only` collects the migrated tests

## 11. Test migration

- [x] 11.1 Migrate existing unit tests into `tests/unit/` mirroring `src/` structure, adapting imports, and verify `pytest tests/unit` passes
- [x] 11.2 Migrate existing integration tests into `tests/integration/` using public APIs only and verify `pytest tests/integration` passes
- [x] 11.3 Move `smoke_test.py` to `tests/integration/test_smoke.py` importing only `infrastructure.llm` and `infrastructure.server` public APIs and verify `pytest tests/integration/test_smoke.py` runs (or skips gracefully if no server is available)
- [x] 11.4 Migrate contract tests into `tests/contract/` using public APIs only and verify `pytest tests/contract` passes

## 12. Documentation and cleanup

- [x] 12.1 Update `README.md` run examples to `python -m dynamic_prompt_core.interfaces.cli.main` and `python -m dynamic_prompt_core.interfaces.server.main` and verify no old filenames are referenced
- [x] 12.2 Verify `config.toml` remains at the project root and is still read correctly by `load_config`

## 13. Atomic removal and integration

- [x] 13.1 Run `lint-imports` and verify it passes against the full layered tree
- [x] 13.2 Run `mypy` and verify it passes
- [x] 13.3 Run `ruff check` and verify it passes
- [x] 13.4 Run `pytest` and verify all tests pass
- [x] 13.5 Verify both entrypoints run: `python -m dynamic_prompt_core.interfaces.cli.main --help` and `python -m dynamic_prompt_core.interfaces.server.main --help`
- [x] 13.6 Delete all 19 root-level `.py` files in one commit and verify `git status` shows only deletions and the tree from 1.2 remains
- [x] 13.7 Re-run `lint-imports`, `mypy`, `ruff check`, `pytest` after deletion and verify all still pass
- [ ] 13.8 Merge `migration/architecture` to main after acceptance criteria pass and verify `git log --oneline` shows the merge (needs explicit user request)
