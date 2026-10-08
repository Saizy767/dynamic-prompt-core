"""Forbidden-reference tests: verify generative classification is fully removed.

These tests target specific deleted symbols, not arbitrary text. The following
symbols are explicitly ALLOWED and not flagged:
  - classify_input (the new candidate-scoring use case)
  - ClassificationPolicy / ArgmaxClassificationPolicy (the new policy)
  - Classification (the new domain model)
  - CLASSIFICATION_PROMPT_V0 (retained as prompt versioning seed)
  - classification_policy.py (the new service module)
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "dynamic_prompt_core"

FORBIDDEN_SYMBOLS = [
    "classify_detailed",
    "ClassificationResult",
    "classify_many_detailed",
]

FORBIDDEN_PORT_METHODS = ["classify", "classify_many"]


def _all_python_files(root: Path) -> list[Path]:
    return list(root.rglob("*.py"))


def _file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("symbol", FORBIDDEN_SYMBOLS)  # noqa
def test_no_production_code_references_forbidden_symbol(symbol: str) -> None:
    """No production file under src/ may reference a deleted generative symbol."""
    for py_file in _all_python_files(SRC_ROOT):
        text = _file_text(py_file)
        assert symbol not in text, (
            f"{py_file} references forbidden symbol '{symbol}'"
        )


def test_llmclient_port_has_no_classify_methods() -> None:
    """The LLMClient port must not define classify or classify_many."""
    port_path = SRC_ROOT / "application" / "ports" / "outbound" / "llm_client.py"
    text = _file_text(port_path)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert node.name not in FORBIDDEN_PORT_METHODS, (
                f"LLMClient port defines forbidden method '{node.name}'"
            )


def test_no_runtime_fallback_to_generative_classification() -> None:
    """No production file may contain a try/except fallback pattern that
    falls back from candidate scoring to generative classification."""
    fallback_pattern = re.compile(
        r"classify_detailed|ClassificationResult",
        re.IGNORECASE,
    )
    for py_file in _all_python_files(SRC_ROOT):
        text = _file_text(py_file)
        matches = fallback_pattern.findall(text)
        assert not matches, (
            f"{py_file} contains possible fallback to generative classification: {matches}"
        )
