"""Tests for cross-cycle in_prompt carry-over (duplicate-rule prevention)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer import (
    ThesisAnalyzerConfig,
    ThesisBank,
    ThesisEntry,
    carry_over_in_prompt_flags,
)

STEPS = "dynamic_prompt_core.application.use_cases.run_cycle.steps"


def _bank_with_norms(*norms: str) -> ThesisBank:
    bank = ThesisBank(ThesisAnalyzerConfig())
    for n in norms:
        bank.theses[n] = ThesisEntry(text_raw=n, text_norm=n)
    return bank


def _write_previous_bank(tmp_path, theses: dict) -> str:
    path = tmp_path / "thesis_bank_prev.json"
    path.write_text(json.dumps({"theses": theses, "clusters": {}}), encoding="utf-8")
    return str(path)


def test_carry_over_flags_matching_norms(tmp_path):
    previous = _write_previous_bank(
        tmp_path,
        {
            "flood water": {"in_prompt": True, "cluster_id": 0},
            "storm damage": {"in_prompt": False, "cluster_id": 1},
        },
    )
    bank = _bank_with_norms("flood water", "storm damage", "new thesis")

    carried = carry_over_in_prompt_flags(bank, previous)

    assert carried == 1
    assert bank.theses["flood water"].in_prompt is True
    assert bank.theses["storm damage"].in_prompt is False
    assert bank.theses["new thesis"].in_prompt is False


def test_carry_over_on_missing_file_raises_nothing_but_steps_guard(tmp_path):
    """A missing previous bank is guarded by the caller (os.path.exists)."""
    bank = _bank_with_norms("a")
    try:
        carry_over_in_prompt_flags(bank, str(tmp_path / "missing.json"))
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised  # the steps wiring checks existence before calling


def _empty_results(tmp_path) -> str:
    path = tmp_path / "results_test-run_v1_dev_20260101T000000Z.jsonl"
    path.write_text("", encoding="utf-8")
    return str(path)


def test_update_thesis_collection_carries_flags_from_previous_bank(tmp_path):
    """Regression: the new bank must inherit in_prompt flags by text_norm,
    so the selector excludes clusters whose rules are already in the prompt."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        update_thesis_collection,
    )

    previous = _write_previous_bank(
        tmp_path,
        {"flood water": {"in_prompt": True, "cluster_id": 0}},
    )
    bank = _bank_with_norms("flood water", "new thesis")

    with (
        patch(f"{STEPS}.ThesisAnalyzerConfig.from_config", return_value=MagicMock()),
        patch(
            f"{STEPS}.analyze",
            return_value=(bank, {"run_id": "test-run", "prompt_version": "v1"}),
        ),
        patch(f"{STEPS}.write_dump", return_value="thesis_bank.json") as mock_dump,
    ):
        update_thesis_collection(
            _empty_results(tmp_path),
            "config.toml",
            previous_bank_path=previous,
        )

    written_bank = mock_dump.call_args[0][0]
    assert written_bank.theses["flood water"].in_prompt is True
    assert written_bank.theses["new thesis"].in_prompt is False


def test_update_thesis_collection_ignores_missing_previous_bank(tmp_path):
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        update_thesis_collection,
    )

    bank = _bank_with_norms("flood water")

    with (
        patch(f"{STEPS}.ThesisAnalyzerConfig.from_config", return_value=MagicMock()),
        patch(
            f"{STEPS}.analyze",
            return_value=(bank, {"run_id": "test-run", "prompt_version": "v1"}),
        ),
        patch(f"{STEPS}.write_dump", return_value="thesis_bank.json"),
        patch(f"{STEPS}.carry_over_in_prompt_flags") as mock_carry,
    ):
        update_thesis_collection(
            _empty_results(tmp_path),
            "config.toml",
            previous_bank_path=str(tmp_path / "missing.json"),
        )

    assert not mock_carry.called
