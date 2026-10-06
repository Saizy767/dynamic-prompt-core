"""Typed input and result for the refine_theses use case."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RefineThesesInput:
    """Input for the refine_theses use case."""

    results_path: str
    run_id: str
    prompt_version: str
    output_dir: str = "data/results"
    log_path: str = "data/thesis_refiner.jsonl"
    filter_noisy: bool = True
    filter_interpretive: bool = True
    allow_additions: bool = True


@dataclass(frozen=True)
class RefineThesesResult:
    """Typed result of the refine_theses use case."""

    processed: int = 0
    teacher_errors: int = 0
    filtered: int = 0
    reformulated: int = 0
    added: int = 0
    skipped: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    artifact_path: str = ""
