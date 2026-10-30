"""Unit tests for the Extractor component."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from dynamic_prompt_core.infrastructure.llm import CallResult, ResponseStatus
from dynamic_prompt_core.infrastructure.llm.extractor import (
    Extractor,
)


class TestExtractor:
    def test_loads_dedicated_system_prompt(self) -> None:
        with patch("dynamic_prompt_core.infrastructure.llm.extractor.load_prompt") as mock_load:
            mock_load.return_value = "EXTRACTOR_PROMPT_TEXT"
            ext = Extractor(task=MagicMock(), language="ru")
            mock_load.assert_called_once_with("extractor")
            assert ext._system_prompt == "EXTRACTOR_PROMPT_TEXT"

    async def test_successful_extraction(self) -> None:
        mock_task = MagicMock()
        mock_parsed = MagicMock()
        mock_parsed.theses = ["thesis one", "thesis two"]
        mock_task.extract_theses_detailed = AsyncMock(
            return_value=CallResult(
                parsed=mock_parsed,
                raw_content='{"theses": ["thesis one", "thesis two"]}',
                latency_ms=42.0,
                parse_status=None,
                status=ResponseStatus.OK,
            )
        )

        with patch(
            "dynamic_prompt_core.infrastructure.llm.extractor.normalize_theses",
            return_value=["norm one", "norm two"],
        ):
            ext = Extractor(task=mock_task, language="ru")
            result = await ext.extract("some text", true_val=1)

        assert result.status == "ok"
        assert result.theses_raw == ["thesis one", "thesis two"]
        assert result.theses_norm == ["norm one", "norm two"]
        assert result.latency_ms == 42.0
        assert result.raw_extract == '{"theses": ["thesis one", "thesis two"]}'

    async def test_extraction_crash_produces_failed(self) -> None:
        mock_task = MagicMock()
        mock_task.extract_theses_detailed = AsyncMock(side_effect=RuntimeError("boom"))

        ext = Extractor(task=mock_task, language="ru")
        result = await ext.extract("text")

        assert result.status == "failed"
        assert result.theses_raw == []
        assert result.theses_norm == []

    def test_skipped_result(self) -> None:
        result = Extractor.skipped()
        assert result.status == "skipped"
        assert result.theses_raw == []
        assert result.theses_norm == []
        assert result.latency_ms == 0.0
        assert result.raw_extract is None

    async def test_does_not_receive_classification_results(self) -> None:
        mock_task = MagicMock()
        mock_parsed = MagicMock()
        mock_parsed.theses = ["t"]
        mock_task.extract_theses_detailed = AsyncMock(
            return_value=CallResult(
                parsed=mock_parsed,
                raw_content="{}",
                latency_ms=1.0,
                parse_status=None,
                status=ResponseStatus.OK,
            )
        )

        with patch(
            "dynamic_prompt_core.infrastructure.llm.extractor.normalize_theses",
            return_value=["n"],
        ):
            ext = Extractor(task=mock_task)
            await ext.extract("text", true_val=0)

        call_kwargs = mock_task.extract_theses_detailed.call_args
        assert "predicted_decision" not in call_kwargs.kwargs
        assert "classification" not in call_kwargs.kwargs
