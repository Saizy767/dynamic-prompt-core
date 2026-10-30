"""Tests for _load_best_prompt version selection."""

from __future__ import annotations

import json

import pytest

from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import (
    _load_best_prompt,
)


def _write_prompt(path, version: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": version, "text": f"text-{version}", "hash": ""}, f)


def test_load_best_prompt_picks_highest_version_number(tmp_path):
    """v10 must win over v2 despite lexicographic order."""
    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()
    _write_prompt(prompt_dir / "prompt_v2_run1_20260101T000000Z.json", "v2")
    _write_prompt(prompt_dir / "prompt_v10_run1_20260102T000000Z.json", "v10")

    artifact, path = _load_best_prompt(str(prompt_dir))
    assert artifact.version == "v10"
    assert "prompt_v10" in path


def test_load_best_prompt_empty_dir_raises(tmp_path):
    """No artifacts means an explicit error, not a silent v0 fallback."""
    from dynamic_prompt_core.application.use_cases.run_cycle.report import (
        CycleOrchestratorError,
    )

    with pytest.raises(CycleOrchestratorError):
        _load_best_prompt(str(tmp_path))
