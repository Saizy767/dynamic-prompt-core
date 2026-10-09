"""Shared fixtures and skip logic for opt-in GGUF integration tests.

These tests require the optional ``llama-cpp-python`` runtime and a local GGUF
test model.  They are skipped by default and collected only when
``--run-gguf`` is passed and the ``GGUF_TEST_MODEL`` environment variable
points to a readable GGUF file.
"""
from __future__ import annotations

import os

import pytest


def _gguf_available() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("GGUF_TEST_MODEL"))


gguf_required = pytest.mark.gguf
gguf_skip = pytest.mark.skipif(
    not _gguf_available(),
    reason="requires llama-cpp-python and GGUF_TEST_MODEL env var (use --run-gguf)",
)


@pytest.fixture
def gguf_model_path() -> str:
    path = os.environ.get("GGUF_TEST_MODEL", "")
    if not path or not os.path.isfile(path):
        pytest.skip("GGUF_TEST_MODEL does not point to a readable file")
    return path
