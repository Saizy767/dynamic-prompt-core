"""
One-time conversion: data/train.csv -> data/train.jsonl.

Maps CSV columns: id -> id, text -> text, target -> label.
Skips rows with empty text. Run once before the first Stage 1 run:

    python convert_csv_to_jsonl.py
    python convert_csv_to_jsonl.py --input data/train.csv --output data/train.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys


def convert(input_path: str, output_path: str) -> int:
    count = 0
    with open(input_path, "r", encoding="utf-8", newline="") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            label_raw = (row.get("target") or "").strip()
            if label_raw not in ("0", "1"):
                continue
            record = {
                "id": int(row["id"]),
                "text": text,
                "label": int(label_raw),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


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
