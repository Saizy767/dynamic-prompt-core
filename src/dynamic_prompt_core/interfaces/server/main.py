"""Server composition root: assemble ServerLauncher and manage lifecycle."""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from dynamic_prompt_core.infrastructure.config import ConfigError, load_config
from dynamic_prompt_core.infrastructure.server import ServerLauncher, VllmNotSupportedError

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the LLM inference server.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def build_server_deps(config: dict[str, Any]) -> ServerLauncher:
    return ServerLauncher(config)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    launcher = build_server_deps(cfg)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    exit_code = 0

    def _signal_handler(signum: int) -> None:
        nonlocal exit_code
        log.info("Received signal %d, shutting down.", signum)
        exit_code = 130 if signum == signal.SIGINT else 0
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler, sig)

    try:
        endpoint = launcher.start()
        log.info("Server ready at %s (PID %d). Press Ctrl+C to stop.", endpoint, launcher.pid)
        await stop_event.wait()
    except VllmNotSupportedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        log.exception("Unrecoverable error during server lifecycle.")
        return 1
    finally:
        launcher.stop()
        log.info("Cleanup complete.")

    return exit_code


def server_entry() -> None:
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    server_entry()
