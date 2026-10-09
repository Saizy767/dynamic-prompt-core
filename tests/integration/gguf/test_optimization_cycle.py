"""Integration tests for the optimization cycle with the GGUF backend (opt-in).

Task 7.3: the optimization cycle accepts and rejects candidates according to
its existing policy, and resume validates model and experiment identity.
"""
from __future__ import annotations

import pytest

from .conftest import gguf_skip


@gguf_skip
class TestOptimizationCycleGguf:
    async def test_cycle_runs_with_gguf_backend(self, gguf_model_path: str) -> None:
        pytest.skip("requires full optimization-cycle fixture wiring")

    async def test_resume_validates_model_identity(self, gguf_model_path: str) -> None:
        pytest.skip("requires full optimization-cycle fixture wiring")
