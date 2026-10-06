"""CLI observer: format cycle observation reports as markdown or JSON."""
from __future__ import annotations

import argparse
import sys

from dynamic_prompt_core.application.use_cases.observe_cycle.observer import (
    CycleObserverConfig,
    run_observer,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe and report on optimization cycles")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--run-id", default="latest", help="Run ID to observe")
    args = parser.parse_args()

    config = CycleObserverConfig.from_config(args.config)
    report_path = run_observer(args.run_id, config)

    with open(report_path, encoding="utf-8") as f:
        content = f.read()

    if args.format == "json":
        print(content)
    else:
        print(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
