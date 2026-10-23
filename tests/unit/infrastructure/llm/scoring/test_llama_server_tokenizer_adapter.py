"""Unit tests for the llama-server HTTP tokenizer adapter."""
from __future__ import annotations

from typing import Any

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.llama_server_tokenizer_adapter import (
    LlamaServerBatchTokenizerAdapter,
)


class _MockResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        if isinstance(self._body, str):
            return self._body
        return str(self._body)

    async def json(self) -> Any:
        return self._body

    async def __aenter__(self) -> _MockResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


class _MockSession:
    def __init__(self, responses: list[_MockResponse]) -> None:
        self._responses = list(responses)
        self.closed = False
        self.post_calls: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any] | None = None) -> _MockResponse:
        self.post_calls.append({"url": url, "json": json})
        if not self._responses:
            raise AssertionError("no more mock responses")
        return self._responses.pop(0)

    async def close(self) -> None:
        self.closed = True


class TestLlamaServerBatchTokenizerAdapter:
    def _make_adapter_with_mock(
        self, responses: list[_MockResponse]
    ) -> tuple[LlamaServerBatchTokenizerAdapter, _MockSession]:
        adapter = LlamaServerBatchTokenizerAdapter(
            endpoint="http://127.0.0.1:8080",
            timeout=5,
        )
        mock_session = _MockSession(responses)
        adapter._session = mock_session
        return adapter, mock_session

    async def test_encode_batch_single_item(self) -> None:
        full_tokens = [1, 2, 3, 4, 5]
        prefix_tokens = [1, 2, 3]
        adapter, mock = self._make_adapter_with_mock(
            [
                _MockResponse(200, {"tokens": prefix_tokens}),
                _MockResponse(200, {"tokens": full_tokens}),
            ]
        )
        result = await adapter.encode_batch(["hello "], ["world"])
        assert len(result.input_ids) == 1
        assert result.input_ids[0] == [1, 2, 3, 4, 5]
        assert result.prefix_token_counts[0] == 3
        assert result.candidate_token_ids[0] == [4, 5]
        assert result.attention_mask[0] == [1, 1, 1, 1, 1]

    async def test_encode_batch_multiple_items_with_padding(self) -> None:
        adapter, _ = self._make_adapter_with_mock(
            [
                _MockResponse(200, {"tokens": [1, 2]}),
                _MockResponse(200, {"tokens": [4, 5, 6]}),
                _MockResponse(200, {"tokens": [1, 2, 3]}),
                _MockResponse(200, {"tokens": [4, 5, 6, 7, 8]}),
            ]
        )
        result = await adapter.encode_batch(
            ["a", "b"],
            ["c", "d"],
        )
        assert len(result.input_ids) == 2
        assert result.input_ids[0] == [1, 2, 3, 0, 0]
        assert result.input_ids[1] == [4, 5, 6, 7, 8]
        assert result.attention_mask[0] == [1, 1, 1, 0, 0]
        assert result.attention_mask[1] == [1, 1, 1, 1, 1]
        assert result.prefix_token_counts == [2, 3]
        assert result.candidate_token_ids[0] == [3]
        assert result.candidate_token_ids[1] == [7, 8]

    async def test_candidate_value_sent_verbatim(self) -> None:
        adapter, mock = self._make_adapter_with_mock(
            [
                _MockResponse(200, {"tokens": [1]}),
                _MockResponse(200, {"tokens": [1, 2]}),
            ]
        )
        await adapter.encode_batch(["prefix "], ["  spaced  "])
        full_call = mock.post_calls[1]["json"]
        assert full_call["content"] == "prefix   spaced  "

    async def test_add_special_false_in_request(self) -> None:
        adapter, mock = self._make_adapter_with_mock(
            [
                _MockResponse(200, {"tokens": [1]}),
                _MockResponse(200, {"tokens": [1, 2]}),
            ]
        )
        await adapter.encode_batch(["a"], ["b"])
        for call in mock.post_calls:
            assert call["json"]["add_special"] is False

    async def test_empty_batch(self) -> None:
        adapter = LlamaServerBatchTokenizerAdapter(endpoint="http://127.0.0.1:8080")
        result = await adapter.encode_batch([], [])
        assert result.input_ids == []
        assert result.attention_mask == []

    async def test_boundary_mismatch_raises(self) -> None:
        """BPE merge across prefix/candidate boundary must be rejected."""
        adapter, _ = self._make_adapter_with_mock(
            [
                _MockResponse(200, {"tokens": [1, 2, 3]}),
                _MockResponse(200, {"tokens": [1, 9]}),
            ]
        )
        with pytest.raises(LLMScoringError, match="boundary mismatch"):
            await adapter.encode_batch(["pre"], ["fix"])

    async def test_unequal_lengths_raises(self) -> None:
        adapter = LlamaServerBatchTokenizerAdapter(endpoint="http://127.0.0.1:8080")
        with pytest.raises(LLMScoringError, match="equal length"):
            await adapter.encode_batch(["a", "b"], ["c"])

    async def test_non_200_raises(self) -> None:
        adapter, _ = self._make_adapter_with_mock(
            [_MockResponse(400, "bad request")]
        )
        with pytest.raises(LLMScoringError, match="HTTP 400"):
            await adapter.encode_batch(["a"], ["b"])

    async def test_missing_tokens_field_raises(self) -> None:
        adapter, _ = self._make_adapter_with_mock(
            [_MockResponse(200, {"content": "no tokens here"})]
        )
        with pytest.raises(LLMScoringError, match="missing 'tokens'"):
            await adapter.encode_batch(["a"], ["b"])

    async def test_aclose_closes_session(self) -> None:
        adapter, mock = self._make_adapter_with_mock([])
        await adapter.aclose()
        assert mock.closed is True
