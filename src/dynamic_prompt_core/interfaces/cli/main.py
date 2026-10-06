"""CLI composition root: parse args, load config, assemble dependencies, invoke use cases."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tomllib
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
    cycle.add_argument("--split", default="dev", help="Dataset split to use")
    cycle.add_argument("--rounds", type=int, default=3, help="Number of optimization rounds")

    refine = sub.add_parser("refine", help="Run teacher thesis refinement")
    refine.add_argument(
        "--results",
        required=True,
        help="Path to results_*.jsonl artifact from stage1-baseline-runner",
    )
    refine.add_argument("--run-id", default=None, help="Override run_id")
    refine.add_argument("--prompt-version", default=None, help="Override prompt_version")

    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def build_cli_deps(config: dict[str, Any]) -> RunCycleDeps:
    """Assemble all dependencies for the run_cycle use case.

    This is the composition root: it creates concrete infrastructure
    implementations and passes them through the typed dependency object.
    """
    from dynamic_prompt_core.infrastructure.llm import AsyncTask
    from dynamic_prompt_core.infrastructure.storage import PromptStore, PromptStoreConfig

    llm = AsyncTask(config_path=config.get("config_path", DEFAULT_CONFIG_PATH))
    store_config = PromptStoreConfig.from_config(config.get("config_path", DEFAULT_CONFIG_PATH))
    store = PromptStore(store_config)

    return RunCycleDeps(llm_client=llm, task_store=store)  # type: ignore[arg-type]


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

    deps = build_cli_deps(cfg)
    rounds = getattr(args, "rounds", None) or 3
    split = getattr(args, "split", None) or "dev"
    run_input = RunCycleInput(config_path=config_path, max_rounds=rounds, dev_split=split)

    try:
        result: RunCycleResult = await run_cycle(deps, run_input)
        log.info(
            "Cycle complete: %d rounds, stop=%s, final=%s",
            result.total_rounds, result.stop_reason, result.final_active_version,
        )
        return 0
    except NotImplementedError:
        log.error("run_cycle not yet implemented. Wire dependencies in build_cli_deps.")
        return 1
    except Exception:
        log.exception("Unrecoverable error during cycle execution.")
        return 1


def cli_entry() -> None:
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli_entry()
