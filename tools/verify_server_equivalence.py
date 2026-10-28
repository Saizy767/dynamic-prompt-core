"""Verify llama_server scoring equivalence against the GGUF reference capture.

Replays the rows stored in ``data/results/gguf_reference_scores.json``
through the llama_server backend (same rules, read from the artifact) and
compares per-candidate probabilities with tolerance.  Run against a live
llama-server serving the same model file the GGUF capture used.

Usage::

    python tools/verify_server_equivalence.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification_rules import ClassificationRules
from dynamic_prompt_core.infrastructure.llm.classifier_factory import build_classifier
from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    scorer_backend_config_from_toml,
)

REFERENCE = "data/results/gguf_reference_scores.json"
# Probability tolerance: the GGUF reference was captured with the model split
# 16 GPU / 20 CPU layers; the server runs all-Metal. Different llama.cpp
# compute kernels accumulate float32 differently through 36 layers, so
# probabilities drift by up to ~0.06 across backends. Decisions are the
# hard criterion; probabilities are compared with this bounded tolerance.
PROBABILITY_TOLERANCE = 0.1


async def main() -> None:
    artifact = json.loads(Path(REFERENCE).read_text(encoding="utf-8"))
    rules = ClassificationRules(rules=tuple(artifact["rules"]))

    with open("config.toml", "rb") as f:
        config = scorer_backend_config_from_toml(tomllib.load(f))
    classifier = build_classifier(config, [Candidate("0"), Candidate("1")])

    max_delta = 0.0
    mismatches = 0
    total_latency = 0.0
    for row in artifact["rows"]:
        t0 = time.monotonic()
        classification = await classifier.classify(row["text"], rules)
        latency = time.monotonic() - t0
        total_latency += latency
        server_scores = {j.candidate.value: j.score for j in classification.judgments}
        ref_scores = row["scores"]

        deltas = {
            c: abs(server_scores[c] - ref_scores[c]) for c in ref_scores
        }
        row_max = max(deltas.values())
        max_delta = max(max_delta, row_max)
        selected_match = classification.selected.value == row["selected"]
        if row_max > PROBABILITY_TOLERANCE or not selected_match:
            mismatches += 1
            reason = "decision" if not selected_match else "probability"
            print(
                f"MISMATCH ({reason}) id={row['id']}: ref={ref_scores} "
                f"server={server_scores} deltas={deltas} "
                f"selected ref={row['selected']} "
                f"server={classification.selected.value}"
            )
        else:
            print(
                f"id={row['id']} OK max_delta={row_max:.5f} "
                f"latency={latency:.2f}s",
                flush=True,
            )

    n = len(artifact["rows"])
    print(
        f"\n{n} rows, mismatches={mismatches}, max|delta|={max_delta:.5f} "
        f"(tolerance {PROBABILITY_TOLERANCE}), "
        f"mean server latency={total_latency / n:.2f}s "
        f"(GGUF reference mean: "
        f"{sum(r['latency_s'] for r in artifact['rows']) / n:.2f}s)"
    )
    await classifier.aclose()
    if mismatches:
        raise SystemExit(f"FAILED: {mismatches} rows out of tolerance")
    print("EQUIVALENCE VERIFIED")


if __name__ == "__main__":
    asyncio.run(main())
