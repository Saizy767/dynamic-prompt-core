"""Tests for GPU-accelerated embedding: device resolution, batch-encode, config validation."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer import (
    ThesisAnalyzerConfig,
    ThesisAnalyzerError,
    ThesisBank,
    precompute_embeddings,
    resolve_embedding_device,
)


class TestResolveEmbeddingDevice:
    """Task 6.1: device resolution for all four inputs + fallback."""

    def test_cpu_passes_through(self) -> None:
        assert resolve_embedding_device("cpu") == "cpu"

    def test_auto_prefers_mps_when_available(self) -> None:
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = True
        fake_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            assert resolve_embedding_device("auto") == "mps"

    def test_auto_falls_to_cuda_when_no_mps(self) -> None:
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = False
        fake_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            assert resolve_embedding_device("auto") == "cuda"

    def test_auto_falls_to_cpu_when_no_gpu(self) -> None:
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = False
        fake_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            assert resolve_embedding_device("auto") == "cpu"

    def test_mps_falls_back_to_cpu_when_unavailable(self) -> None:
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            assert resolve_embedding_device("mps") == "cpu"

    def test_mps_used_when_available(self) -> None:
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            assert resolve_embedding_device("mps") == "mps"

    def test_cuda_falls_back_to_cpu_when_unavailable(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            assert resolve_embedding_device("cuda") == "cpu"

    def test_cuda_used_when_available(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            assert resolve_embedding_device("cuda") == "cuda"


class TestPrecomputeEmbeddings:
    """Task 6.2: batch-encode in a single call with correct args."""

    def test_single_batch_call_with_correct_args(self) -> None:
        norms = {"alpha", "beta", "gamma"}
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
        )
        with patch(
            "dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer._get_embedding_model",
            return_value=mock_model,
        ):
            result = precompute_embeddings(norms, "fake-model", device="cpu", batch_size=8)

        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert call_args.kwargs["batch_size"] == 8
        assert call_args.kwargs["convert_to_numpy"] is True
        encoded_texts = call_args.args[0]
        assert sorted(encoded_texts) == ["alpha", "beta", "gamma"]
        assert set(result.keys()) == norms
        for vec in result.values():
            assert vec.dtype == np.float32

    def test_empty_input_returns_empty_dict_without_loading_model(self) -> None:
        with patch(
            "dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer._get_embedding_model"
        ) as mock_get:
            result = precompute_embeddings(set(), "fake-model")
        mock_get.assert_not_called()
        assert result == {}


class TestAddOrUpdatePrecomputed:
    """Task 6.4: add_or_update uses pre-computed embedding."""

    def test_precomputed_embedding_is_used(self) -> None:
        config = ThesisAnalyzerConfig()
        bank = ThesisBank(config)
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        with patch(
            "dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer.compute_embedding"
        ) as mock_compute:
            entry = bank.add_or_update("test_norm", "test_raw", 1, precomputed_embedding=emb)

        mock_compute.assert_not_called()
        assert entry.embedding is not None
        np.testing.assert_array_equal(entry.embedding, emb)

    def test_falls_back_to_compute_when_no_precomputed(self) -> None:
        config = ThesisAnalyzerConfig()
        bank = ThesisBank(config)
        emb = np.array([0.5, 0.5, 0.0], dtype=np.float32)

        with patch(
            "dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer.compute_embedding",
            return_value=emb,
        ) as mock_compute:
            entry = bank.add_or_update("test_norm", "test_raw", 1)

        mock_compute.assert_called_once()
        assert entry.embedding is not None
        np.testing.assert_array_equal(entry.embedding, emb)


class TestAnalyzeBatchFlow:
    """Task 6.3: two-pass analyze() produces same results as per-thesis path."""

    @staticmethod
    def _make_rows() -> list[dict[str, Any]]:
        return [
            {
                "id": "r1",
                "true_label": 1,
                "theses_norm": ["alpha", "beta"],
                "theses_raw": ["Alpha", "Beta"],
                "extract_status": "ok",
                "classify_status": "ok",
            },
            {
                "id": "r2",
                "true_label": 0,
                "theses_norm": ["beta", "gamma"],
                "theses_raw": ["Beta", "Gamma"],
                "extract_status": "ok",
                "classify_status": "ok",
            },
            {
                "id": "r3",
                "true_label": 1,
                "theses_norm": ["alpha", "delta"],
                "theses_raw": ["Alpha", "Delta"],
                "extract_status": "ok",
                "classify_status": "ok",
            },
        ]

    @staticmethod
    def _make_embedding_map() -> dict[str, np.ndarray]:
        return {
            "alpha": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "beta": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "gamma": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "delta": np.array([0.5, 0.5, 0.0], dtype=np.float32),
        }

    def test_batch_flow_matches_per_thesis(self) -> None:
        from dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer import analyze

        rows = self._make_rows()
        emb_map = self._make_embedding_map()

        mock_model = MagicMock()

        def fake_encode(texts, batch_size=32, convert_to_numpy=True, **kw: Any) -> Any:
            if isinstance(texts, str):
                return emb_map[texts].copy()
            return np.stack([emb_map[t].copy() for t in texts])

        mock_model.encode.side_effect = fake_encode

        with patch(
            "dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer._get_embedding_model",
            return_value=mock_model,
        ):
            config = ThesisAnalyzerConfig()
            artifact = "results_test-run_classify-v0_dev_20260101T000000Z.jsonl"
            bank_batch, _ = analyze(rows, config, artifact)

        assert len(bank_batch.theses) == 4
        for norm, expected_emb in emb_map.items():
            entry = bank_batch.theses[norm]
            assert entry.embedding is not None
            np.testing.assert_allclose(entry.embedding, expected_emb, atol=1e-4)

        assert bank_batch.theses["alpha"].frequency == 2
        assert bank_batch.theses["alpha"].positive_hits == 2
        assert bank_batch.theses["beta"].frequency == 2
        assert bank_batch.theses["beta"].positive_hits == 1
        assert bank_batch.theses["beta"].negative_hits == 1
        assert bank_batch.theses["gamma"].frequency == 1
        assert bank_batch.theses["gamma"].negative_hits == 1


class TestInvalidDeviceConfig:
    """Task 6.5: invalid embedding_device raises config error."""

    def test_thesis_analyzer_rejects_invalid_device(self, tmp_path: Any) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[thesis_analyzer]\nembedding_device = "tpu"\n'
        )
        with pytest.raises(ThesisAnalyzerError, match="invalid embedding_device"):
            ThesisAnalyzerConfig.from_config(str(config_file))

    def test_prompt_composer_rejects_invalid_device(self, tmp_path: Any) -> None:
        from dynamic_prompt_core.application.use_cases.compose_prompt.composer import (
            PromptComposerConfig,
            PromptComposerError,
        )

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[prompt_composer]\nembedding_device = "tpu"\n'
        )
        with pytest.raises(PromptComposerError, match="invalid embedding_device"):
            PromptComposerConfig.from_config(str(config_file))
