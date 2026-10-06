"""CLI wrapper for CSV to JSONL conversion."""
from __future__ import annotations

import argparse
import sys

from dynamic_prompt_core.infrastructure.data.csv_to_jsonl import convert


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert train.csv to jsonl")
    parser.add_argument("--input", default="data/train.csv")
    parser.add_argument("--output", default="data/train.jsonl")
    args = parser.parse_args()
    n = convert(args.input, args.output)
    print(f"Wrote {n} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
