"""Unit tests for the classifier composition-root factory."""

from __future__ import annotations

import sys

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.llm.classifier import LLMLogitClassifier
from dynamic_prompt_core.infrastructure.llm.classifier_factory import build_classifier
from dynamic_prompt_core.infrastructure.llm.scoring.config import ScorerBackendConfig


def test_build_classifier_llama_server_single_path() -> None:
    config = ScorerBackendConfig(backend="llama_server")
    classifier = build_classifier(config, [Candidate("0"), Candidate("1")])
    assert isinstance(classifier, LLMLogitClassifier)
    describe = classifier.describe()
    assert describe["backend"] == "llama_server"
    assert describe["endpoint"] == "http://127.0.0.1:8080"


def test_build_classifier_custom_endpoint() -> None:
    from dynamic_prompt_core.infrastructure.llm.scoring.config import LlamaServerParams

    config = ScorerBackendConfig(
        backend="llama_server",
        llama_server=LlamaServerParams(endpoint="http://10.0.0.1:7070", timeout=45),
    )
    classifier = build_classifier(config, [Candidate("0"), Candidate("1")])
    describe = classifier.describe()
    assert describe["endpoint"] == "http://10.0.0.1:7070"
    assert describe["timeout"] == 45


def test_build_classifier_wires_scoring_concurrency() -> None:
    from dynamic_prompt_core.infrastructure.llm.scoring.config import LlamaServerParams

    config = ScorerBackendConfig(
        backend="llama_server",
        llama_server=LlamaServerParams(max_concurrency=3),
    )
    classifier = build_classifier(config, [Candidate("0"), Candidate("1")])
    assert classifier._model._max_concurrency == 3


def test_build_classifier_does_not_import_llama_cpp() -> None:
    config = ScorerBackendConfig(backend="llama_server")
    build_classifier(config, [Candidate("0"), Candidate("1")])
    assert "llama_cpp" not in sys.modules
