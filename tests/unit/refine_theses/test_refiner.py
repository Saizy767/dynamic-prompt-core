"""Concurrency and ordering tests for the refine_theses use case."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from dynamic_prompt_core.application.use_cases.refine_theses.refine_theses_deps import (
    RefineThesesDeps,
)
from dynamic_prompt_core.application.use_cases.refine_theses.refiner import refine_theses
from dynamic_prompt_core.application.use_cases.refine_theses.result import RefineThesesInput
from dynamic_prompt_core.domain.services.thesis_refinement import RefinementReview


class _TrackingClient:
    """Fake TeacherLLMClient that tracks in-flight calls and can fail per id."""

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.in_flight = 0
        self.max_in_flight = 0

    async def review_theses(self, text: str, theses: list[str]) -> RefinementReview:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        if text in self.fail_ids:
            self.in_flight -= 1
            raise RuntimeError("teacher down")
        review = RefinementReview(
            keep=list(theses),
            latency_ms=1.0,
            usage={"prompt_tokens": 2, "completion_tokens": 3},
        )
        self.in_flight -= 1
        return review


def _write_results(path: Any, ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, row_id in enumerate(ids):
            f.write(
                json.dumps(
                    {
                        "id": row_id,
                        "text": f"text-{i}",
                        "theses_raw": [f"t{i}"],
                        "theses_norm": [f"t{i}"],
                        "extract_status": "ok",
                    }
                )
                + "\n"
            )


def _make_deps(client: _TrackingClient) -> RefineThesesDeps:
    return RefineThesesDeps(teacher_llm_client=client, run_repository=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_refine_runs_concurrently_and_preserves_order(tmp_path: Any) -> None:
    results = tmp_path / "results.jsonl"
    ids = [f"r{i}" for i in range(6)]
    _write_results(results, ids)
    client = _TrackingClient()

    inp = RefineThesesInput(
        results_path=str(results),
        run_id="run",
        prompt_version="v1",
        output_dir=str(tmp_path),
        log_path=str(tmp_path / "log.jsonl"),
        concurrency=3,
    )
    result = await refine_theses(_make_deps(client), inp)

    assert client.max_in_flight == 3
    assert result.processed == 6
    assert result.teacher_errors == 0
    with open(result.artifact_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert [r["id"] for r in records] == ids


@pytest.mark.asyncio
async def test_refine_isolates_teacher_errors_under_concurrency(tmp_path: Any) -> None:
    results = tmp_path / "results.jsonl"
    ids = [f"r{i}" for i in range(4)]
    _write_results(results, ids)
    client = _TrackingClient(fail_ids={"text-1", "text-2"})

    inp = RefineThesesInput(
        results_path=str(results),
        run_id="run",
        prompt_version="v1",
        output_dir=str(tmp_path),
        log_path=str(tmp_path / "log.jsonl"),
        concurrency=4,
    )
    result = await refine_theses(_make_deps(client), inp)

    assert result.processed == 2
    assert result.teacher_errors == 2
    with open(result.artifact_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert [r["id"] for r in records] == ids
    for rec in records:
        if rec["id"] in ("r1", "r2"):
            assert rec["theses_refined"] == rec["theses_raw_tiny"]
