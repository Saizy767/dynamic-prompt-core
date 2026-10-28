"""Tests for the CLI composition root's config override wiring."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import (
    RunCycleResult,
)

MAIN = "dynamic_prompt_core.interfaces.cli.main"

_CONFIG = """
[llm]
backend = "llamacpp"
model_path = "model"
served_model_name = "test-model"
host = "127.0.0.1"
port = 8080

[cycle_orchestrator]
max_cycles = 10
"""


def _write_config(tmp_path, body: str) -> str:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _run_main(argv: list[str]) -> MagicMock:
    deps = MagicMock()
    deps.classifier.aclose = AsyncMock()
    result = RunCycleResult(
        total_cycles=0,
        stop_reason="max_cycles_reached",
        final_active_version="classify-v0",
        accepted_history=[],
        rollback_history=[],
    )
    with (
        patch(f"{MAIN}.build_cli_deps", return_value=deps),
        patch(f"{MAIN}.run_cycle", new_callable=AsyncMock, return_value=result) as run,
    ):
        code = asyncio.run(_main_main(argv))
    assert code == 0
    return run


def _main_main(argv: list[str]):
    from dynamic_prompt_core.interfaces.cli.main import main

    return main(argv)


def _received_config(run: MagicMock):
    _deps, run_input = run.call_args[0]
    return run_input.config


def test_cycle_without_flags_uses_config_max_cycles(tmp_path):
    """Regression: no --cycles must NOT force 3 cycles over config's 10."""
    config_path = _write_config(tmp_path, _CONFIG)

    run = _run_main(["--config", config_path, "cycle"])

    config = _received_config(run)
    assert config.max_cycles == 10
    assert config.dev_split == "dev"


def test_cycle_flags_override_config(tmp_path):
    """--cycles/--split override only when passed; list duration collapses."""
    config_path = _write_config(
        tmp_path,
        _CONFIG.replace("max_cycles = 10", "max_cycles = 3\nduration = [10, 20, 30]"),
    )

    run = _run_main(
        ["--config", config_path, "cycle", "--cycles", "5", "--split", "holdout"]
    )

    config = _received_config(run)
    assert config.max_cycles == 5
    assert config.duration == 10
    assert config.dev_split == "holdout"


def test_parse_args_cycle_flags_default_to_none():
    """Both flags must default to None so the config value survives."""
    from dynamic_prompt_core.interfaces.cli.main import parse_args

    args = parse_args(["cycle"])
    assert args.cycles is None
    assert args.split is None

    args = parse_args(["cycle", "--cycles", "7", "--split", "holdout"])
    assert args.cycles == 7
    assert args.split == "holdout"
