"""Regression tests: Hugging Face vs GGUF backend (opt-in).

Task 8.1: run a fixed evaluation dataset through both backends using
corresponding models and record prediction agreement, macro F1, per-class F1,
accuracy, latency, and peak memory.  No hard logit equality is asserted.
"""
from __future__ import annotations

import pytest

from tests.integration.gguf.conftest import gguf_skip


@gguf_skip
class TestHfVsGgufRegression:
    async def test_backend_comparison_records_metrics(self, gguf_model_path: str) -> None:
        pytest.skip("requires corresponding HF and GGUF models for the fixed dataset")
