"""CLI composition root: parse args, load config, assemble dependencies, invoke run_cycle."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from dynamic_prompt_core.application.ports.inbound.run_cycle_input import RunCycleInput
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import RunCycleResult, run_cycle
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import RunCycleDeps
from dynamic_prompt_core.infrastructure.config import ConfigError, load_config

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic prompt optimization CLI.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    parser.add_argument("--split", default="dev", help="Dataset split to use")
    parser.add_argument("--rounds", type=int, default=3, help="Number of optimization rounds")
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


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    try:
        cfg = load_config(args.config)
        cfg["config_path"] = args.config
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    deps = build_cli_deps(cfg)
    run_input = RunCycleInput(config_path=args.config, max_rounds=args.rounds, dev_split=args.split)

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
