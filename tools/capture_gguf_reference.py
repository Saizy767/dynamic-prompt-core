"""Capture GGUF reference scores for backend-equivalence verification.

Classifies a fixed slice of the prepared dataset with the current GGUF
backend and writes per-candidate scores to a JSON artifact. Run BEFORE the
GGUF backend is deleted; ``tools/verify_server_equivalence.py`` replays the
same rows through the llama_server backend and compares.

Usage::

    python tools/capture_gguf_reference.py [N_ROWS]
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification_rules import ClassificationRules
from dynamic_prompt_core.infrastructure.llm.classifier_factory import build_classifier
from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    GgufParams,
    ScorerBackendConfig,
)

MODEL_PATH = "model/qwen2.5-3b-instruct-q4_k_m.gguf"
DATASET = "data/prepared_42_20261002T092615Z.jsonl"
OUT = "data/results/gguf_reference_scores.json"

RULES = ClassificationRules(
    rules=(
        "The text reports a real disaster, accident or emergency event",
        "The text is personal opinion or everyday chatter with no event",
        "The text mentions casualties, damage or rescue operations",
    )
)


async def main(n_rows: int) -> None:
    rows = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") == "dev":
                rows.append(row)
            if len(rows) >= n_rows:
                break

    config = ScorerBackendConfig(
        backend="gguf",
        model_path=MODEL_PATH,
        gguf=GgufParams(n_ctx=2048, n_batch=512, n_gpu_layers=16, seed=42),
    )
    classifier = build_classifier(config, [Candidate("0"), Candidate("1")])

    results = []
    t_start = time.monotonic()
    for row in rows:
        t0 = time.monotonic()
        classification = await classifier.classify(row["text"], RULES)
        elapsed = time.monotonic() - t0
        scores = {j.candidate.value: j.score for j in classification.judgments}
        results.append(
            {
                "id": row["id"],
                "text": row["text"],
                "selected": classification.selected.value,
                "scores": scores,
                "latency_s": round(elapsed, 3),
            }
        )
        print(
            f"id={row['id']} selected={classification.selected.value} "
            f"scores={scores} latency={elapsed:.1f}s",
            flush=True,
        )

    artifact = {
        "backend": "gguf",
        "model_path": MODEL_PATH,
        "gguf_params": {"n_ctx": 2048, "n_batch": 512, "n_gpu_layers": 16, "seed": 42},
        "rules": list(RULES.rules),
        "rows": results,
    }
    Path(OUT).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"\nWrote {OUT}: {len(results)} rows, "
        f"total {time.monotonic() - t_start:.0f}s, "
        f"mean latency {sum(r['latency_s'] for r in results) / len(results):.1f}s"
    )


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
