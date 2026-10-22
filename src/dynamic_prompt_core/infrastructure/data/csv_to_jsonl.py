"""CSV to JSONL conversion logic for dataset preparation."""
from __future__ import annotations

import csv
import json


def convert(input_path: str, output_path: str) -> int:
    count = 0
    with open(input_path, encoding="utf-8", newline="") as fin, \
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
