"""Mock port implementations and fixtures for run_cycle unit tests."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
    RunCycleDeps,
)
from dynamic_prompt_core.domain.models.dataset import Dataset, Record


class MockLLMClient:
    """Mock LLM client port."""

    async def classify(
        self, text, model, *, system_prompt="", max_tokens=512, truncate_tokens=None
    ):
        return {"decision": 1, "confidence": 80}

    async def classify_many(
        self, texts, model, *, system_prompt="", max_tokens=512, truncate_tokens=None
    ):
        return [await self.classify(t, model, system_prompt=system_prompt) for t in texts]

    async def extract_theses(
        self, text, model, *, system_prompt="", max_tokens=512, truncate_tokens=None
    ):
        return {"theses": ["thesis one", "thesis two"]}

    async def extract_theses_many(
        self, texts, model, *, system_prompt="", max_tokens=512, truncate_tokens=None
    ):
        return [await self.extract_theses(t, model, system_prompt=system_prompt) for t in texts]


class MockPromptRepository:
    """Mock prompt repository port."""

    def __init__(self):
        self._versions: dict[int, dict[str, Any]] = {}
        self._active: int | None = None

    def save(self, prompt_version_dict, reason="composed", base_version=None):
        vnum = len(self._versions) + 1
        record = {**prompt_version_dict, "version_number": vnum, "reason": reason}
        self._versions[vnum] = record
        return record

    def get(self, version_number):
        return self._versions.get(version_number, {})

    def get_active(self):
        if self._active is None:
            raise KeyError("no active version")
        return self._versions.get(self._active, {})

    def activate(self, version_number):
        self._active = version_number
        return self._versions.get(version_number, {})

    def list_versions(self):
        return list(self._versions.values())

    def lineage(self, version_number):
        return [self._versions.get(version_number, {})]


class MockRunRepository:
    """Mock run repository port."""

    def __init__(self):
        self.saved: list[tuple[list[dict[str, Any]], str, str]] = []

    def save_results(self, results, run_id, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.saved.append((results, run_id, path))
        return path

    def load_results(self, path):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


class MockDatasetRepository:
    """Mock dataset repository port."""

    def load_dataset(self, path):
        return [Record(id=1, text="test", label=1)]

    def load_artifact(self, path):
        return Dataset(dev=[Record(id=1, text="test", label=1)])

    def write_artifact(self, dataset, output_dir="data", fmt="jsonl", notes=""):
        return "mock_path"


class MockEmbeddingClient:
    """Mock embedding client port."""

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_one(self, text):
        return [0.1, 0.2, 0.3]


class MockNormalizer:
    """Mock normalizer port."""

    def normalize_thesis(self, text, lang="ru"):
        return text.lower().strip()

    def normalize_theses(self, theses, lang="ru"):
        return [self.normalize_thesis(t, lang) for t in theses]


class MockTeacherLLMClient:
    """Mock teacher LLM client port."""

    async def review_theses(self, text, theses):
        from dynamic_prompt_core.domain.services.thesis_refinement import (
            RefinementReview,
        )

        return RefinementReview(
            keep=list(theses),
            reformulate=[],
            drop=[],
            add=[],
            latency_ms=10.0,
            usage={"prompt_tokens": 5, "completion_tokens": 10},
        )


@pytest.fixture
def mock_llm_client():
    return MockLLMClient()


@pytest.fixture
def mock_prompt_repository():
    return MockPromptRepository()


@pytest.fixture
def mock_run_repository():
    return MockRunRepository()


@pytest.fixture
def mock_dataset_repository():
    return MockDatasetRepository()


@pytest.fixture
def mock_embedding_client():
    return MockEmbeddingClient()


@pytest.fixture
def mock_normalizer():
    return MockNormalizer()


@pytest.fixture
def mock_teacher_llm_client():
    return MockTeacherLLMClient()


@pytest.fixture
def build_mock_deps():
    def _build(teacher=None) -> RunCycleDeps:
        return RunCycleDeps(
            llm_client=MockLLMClient(),
            prompt_repository=MockPromptRepository(),
            run_repository=MockRunRepository(),
            dataset_repository=MockDatasetRepository(),
            embedding_client=MockEmbeddingClient(),
            normalizer=MockNormalizer(),
            teacher_llm_client=teacher,
        )

    return _build


@pytest.fixture
def cycle_config():
    return CycleConfig(max_rounds=3, run_id="test-run")


@pytest.fixture
def tmp_output_dir(tmp_path):
    return str(tmp_path / "results")
