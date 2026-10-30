"""CLI composition root: parse args, load config, assemble dependencies, invoke use cases."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import tomllib
from datetime import UTC, datetime
from typing import Any

from dynamic_prompt_core.application.ports.inbound.run_cycle_input import RunCycleInput
from dynamic_prompt_core.application.use_cases.refine_theses.refine_theses_deps import (
    RefineThesesDeps,
)
from dynamic_prompt_core.application.use_cases.refine_theses.refiner import (
    parse_artifact_name,
    print_summary,
    refine_theses,
)
from dynamic_prompt_core.application.use_cases.refine_theses.result import (
    RefineThesesInput,
)
from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import RunCycleResult, run_cycle
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import RunCycleDeps
from dynamic_prompt_core.infrastructure.config import ConfigError, load_config

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic prompt optimization CLI.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    sub = parser.add_subparsers(dest="command")

    cycle = sub.add_parser("cycle", help="Run the optimization cycle")
    cycle.add_argument(
        "--split",
        default=None,
        help="Dataset split to use (default: dev_split from config)",
    )
    cycle.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Number of optimization cycles (default: max_cycles from config)",
    )
    cycle.add_argument("--resume", default=None, help="Resume from a state dump JSON file")

    refine = sub.add_parser("refine", help="Run teacher thesis refinement")
    refine.add_argument(
        "--results",
        required=True,
        help="Path to results_*.jsonl artifact from stage1-baseline-runner",
    )
    refine.add_argument("--run-id", default=None, help="Override run_id")
    refine.add_argument("--prompt-version", default=None, help="Override prompt_version")

    args = parser.parse_args(argv)
    # Artifact-name parsers (metrics, refiner, observer) split on "_".
    if getattr(args, "run_id", None) and "_" in args.run_id:
        parser.error("--run-id must not contain '_'")
    return args


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class _NormalizerAdapter:
    """Thin adapter wrapping infrastructure.nlp module-level functions."""

    def normalize_thesis(self, text: str, lang: str = "ru") -> str:
        from dynamic_prompt_core.infrastructure.nlp import normalize_thesis

        return normalize_thesis(text, lang)

    def normalize_theses(self, theses: list[str], lang: str = "ru") -> list[str]:
        from dynamic_prompt_core.infrastructure.nlp import normalize_theses

        return normalize_theses(theses, lang)


def build_cli_deps(
    config: dict[str, Any], config_path: str, run_id: str | None = None
) -> RunCycleDeps:
    """Assemble all dependencies for the run_cycle use case.

    This is the composition root: it creates concrete infrastructure
    implementations and passes them through the typed dependency object.
    When *run_id* is given, every transport request is logged to
    ``<output_dir>/log_cycle/requests_{run_id}.jsonl``.
    """
    from dynamic_prompt_core.application.use_cases.run_baseline.runner import (
        RunnerConfig,
    )
    from dynamic_prompt_core.domain.models.candidate import Candidate
    from dynamic_prompt_core.infrastructure.data.writer import FileDatasetRepository
    from dynamic_prompt_core.infrastructure.llm import AsyncTask
    from dynamic_prompt_core.infrastructure.llm.classifier_factory import (
        build_classifier,
    )
    from dynamic_prompt_core.infrastructure.llm.extractor import Extractor
    from dynamic_prompt_core.infrastructure.llm.scoring.config import (
        scorer_backend_config_from_toml,
    )
    from dynamic_prompt_core.infrastructure.storage import (
        PromptStoreConfig,
        init_store,
    )
    from dynamic_prompt_core.infrastructure.storage.run_repository import (
        FileRunRepository,
    )

    with open(config_path, "rb") as f:
        toml_config = tomllib.load(f)

    request_log_path: str | None = None
    if run_id:
        output_dir = config.get("cycle_orchestrator", {}).get("output_dir", "data/results")
        request_log_path = os.path.join(output_dir, "log_cycle", f"requests_{run_id}.jsonl")
    llm = AsyncTask(config_path=config_path, log_path=request_log_path, run_id=run_id or "cycle")
    store_config = PromptStoreConfig.from_config(config_path)
    store = init_store(store_config)
    run_repo = FileRunRepository()
    dataset_repo = FileDatasetRepository()
    normalizer = _NormalizerAdapter()

    candidates = tuple(
        Candidate(v) for v in toml_config.get("classification", {}).get("candidates", ["0", "1"])
    )
    classifier = build_classifier(
        scorer_backend_config_from_toml(toml_config),
        list(candidates),
    )
    extractor = Extractor(task=llm, language=RunnerConfig.from_config(config_path).language)

    teacher_llm_client = None
    try:
        t = toml_config.get("teacher", {})
        if t.get("endpoint"):
            from dynamic_prompt_core.infrastructure.llm.teacher_client import TeacherClient

            teacher_llm_client = TeacherClient(
                endpoint=t["endpoint"],
                model_name=t.get("model_name", ""),
                temperature=float(t.get("temperature", 0.0)),
                max_tokens=int(t.get("max_tokens", 512)),
                timeout=int(t.get("timeout", 60)),
                max_retries=int(t.get("max_retries", 3)),
                config_path=config_path,
                filter_noisy=bool(t.get("filter_noisy", True)),
                filter_interpretive=bool(t.get("filter_interpretive", True)),
                allow_additions=bool(t.get("allow_additions", True)),
                tokenizer_path=t.get("model_path"),
            )
    except ValueError:
        # Config typos (bad timeout, missing backend, ConfigError) must fail loudly.
        raise
    except Exception as exc:
        log.warning("teacher client build failed; refinement disabled: %s", exc, exc_info=True)

    stop_criteria_deps = None
    try:
        if "stop_criteria" in toml_config:
            from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.deps import (
                EvaluateStopCriteriaDeps,
            )

            stop_criteria_deps = EvaluateStopCriteriaDeps(run_repository=run_repo)
    except Exception as exc:
        log.debug("stop criteria not configured: %s", exc)

    return RunCycleDeps(
        llm_client=llm,
        prompt_repository=store,
        run_repository=run_repo,
        normalizer=normalizer,
        classifier=classifier,
        extractor=extractor,
        dataset_repository=dataset_repo,
        candidates=candidates,
        teacher_llm_client=teacher_llm_client,
        stop_criteria_deps=stop_criteria_deps,
    )


def build_refine_deps(config_path: str) -> RefineThesesDeps:
    """Assemble all dependencies for the refine_theses use case.

    Creates the TeacherClient and FileRunRepository in the composition root
    and passes them through the typed dependency object.
    """
    from dynamic_prompt_core.infrastructure.llm.teacher_client import TeacherClient
    from dynamic_prompt_core.infrastructure.storage.run_repository import (
        FileRunRepository,
    )

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    t = config.get("teacher", {})
    endpoint = t.get("endpoint", "")
    if not endpoint:
        raise ConfigError("Missing required config key: teacher.endpoint")
    model_name = t.get("model_name", "")
    if not model_name:
        raise ConfigError("Missing required config key: teacher.model_name")

    teacher_client = TeacherClient(
        endpoint=endpoint,
        model_name=model_name,
        temperature=float(t.get("temperature", 0.0)),
        max_tokens=int(t.get("max_tokens", 512)),
        timeout=int(t.get("timeout", 60)),
        max_retries=int(t.get("max_retries", 3)),
        config_path=config_path,
        filter_noisy=bool(t.get("filter_noisy", True)),
        filter_interpretive=bool(t.get("filter_interpretive", True)),
        allow_additions=bool(t.get("allow_additions", True)),
        tokenizer_path=t.get("model_path"),
    )
    run_repo = FileRunRepository()

    return RefineThesesDeps(
        teacher_llm_client=teacher_client,
        run_repository=run_repo,
    )


async def run_refine(args: argparse.Namespace, config_path: str) -> int:
    deps = build_refine_deps(config_path)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    t = config.get("teacher", {})

    default_run_id, default_prompt_version = parse_artifact_name(args.results)
    run_id = args.run_id or default_run_id
    prompt_version = args.prompt_version or default_prompt_version

    inp = RefineThesesInput(
        results_path=args.results,
        run_id=run_id,
        prompt_version=prompt_version,
        output_dir=t.get("output_dir", "data/results"),
        log_path=t.get("log_path", "data/thesis_refiner.jsonl"),
        filter_noisy=bool(t.get("filter_noisy", True)),
        filter_interpretive=bool(t.get("filter_interpretive", True)),
        allow_additions=bool(t.get("allow_additions", True)),
        concurrency=int(t.get("concurrency", 2)),
    )

    try:
        result = await refine_theses(deps, inp)
        print_summary(result)
        print(f"Artifact: {result.artifact_path}")
        return 0
    except Exception:
        log.exception("Unrecoverable error during refinement.")
        return 1


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    config_path = args.config

    if args.command == "refine":
        return await run_refine(args, config_path)

    try:
        cfg = load_config(config_path)
        cfg["config_path"] = config_path
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cycles = getattr(args, "cycles", None)
    split = getattr(args, "split", None)
    resume_from = getattr(args, "resume", None)

    cycle_config = CycleConfig.from_toml(config_path)
    if not cycle_config.run_id:
        cycle_config = type(cycle_config)(
            **{
                **cycle_config.__dict__,
                "run_id": f"cycle-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
            }
        )
    deps = build_cli_deps(cfg, config_path, run_id=cycle_config.run_id)
    overrides: dict[str, Any] = {}
    if cycles is not None and cycles != cycle_config.max_cycles:
        # A list duration would re-override max_cycles in __post_init__;
        # collapse it so the user's --cycles survives.
        d = cycle_config.duration
        overrides["max_cycles"] = cycles
        overrides["duration"] = d[0] if isinstance(d, list) and d else d
    if split is not None and split != cycle_config.dev_split:
        overrides["dev_split"] = split
    if overrides:
        cycle_config = type(cycle_config)(**{**cycle_config.__dict__, **overrides})

    try:
        from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
            StopCriteriaConfig,
        )

        with open(config_path, "rb") as f:
            toml_config = tomllib.load(f)
        if "stop_criteria" in toml_config:
            stop_criteria_config = StopCriteriaConfig.from_toml(config_path)
            cycle_config = type(cycle_config)(
                **{
                    **cycle_config.__dict__,
                    "stop_criteria_config": stop_criteria_config,
                }
            )
    except Exception as exc:
        log.debug("stop criteria config not loaded: %s", exc)

    run_input = RunCycleInput(
        config_path=config_path,
        resume_from=resume_from,
        config=cycle_config,
    )

    def _request_shutdown() -> None:
        log.info("shutdown signal received; cancelling cycle for clean cleanup")
        raise asyncio.CancelledError()

    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
            installed_signals.append(sig)
        except NotImplementedError:
            # signal handlers require a running event loop with signal support
            pass

    try:
        result: RunCycleResult = await run_cycle(deps, run_input)
        log.info(
            "Cycle complete: %d cycles, stop=%s, final=%s",
            result.total_cycles,
            result.stop_reason,
            result.final_active_version,
        )
        return 0
    except asyncio.CancelledError:
        log.warning("Cycle cancelled by signal; state dumps from completed cycles remain on disk.")
        return 130
    except Exception:
        log.exception("Unrecoverable error during cycle execution.")
        return 1
    finally:
        for sig in installed_signals:
            loop.remove_signal_handler(sig)
        close_fn = getattr(deps.classifier, "aclose", None)
        if close_fn is not None:
            await close_fn()
        await deps.llm_client.aclose()


def cli_entry() -> None:
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli_entry()
