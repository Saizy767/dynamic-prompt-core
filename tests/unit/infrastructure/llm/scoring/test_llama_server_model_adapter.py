"""Unit tests for the llama-server HTTP model adapter."""
from __future__ import annotations

from typing import Any

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.llama_server_model_adapter import (
    LlamaServerCausalLMAdapter,
    _build_logits_from_logprobs,
    _parse_v1_logprobs,
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


class TestBuildLogitsFromLogprobs:
    def test_basic_conversion(self) -> None:
        token_logprobs = [None, -0.5, -1.2]
        token_ids = [1, 2, 3]
        result = _build_logits_from_logprobs(token_logprobs, token_ids, vocab_size=10)
        assert len(result) == 3
        assert len(result[0]) == 10
        assert result[1][2] == pytest.approx(-0.5)
        assert result[2][3] == pytest.approx(-1.2)

    def test_none_logprob_first_position(self) -> None:
        result = _build_logits_from_logprobs([None], [5], vocab_size=10)
        assert all(v == -100.0 for v in result[0])

    def test_nan_rejected(self) -> None:
        with pytest.raises(LLMScoringError, match="NaN"):
            _build_logits_from_logprobs([float("nan")], [0], vocab_size=10)


class TestParseV1Logprobs:
    def test_valid_response(self) -> None:
        body = {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [None, -0.5, -1.2],
                        "tokens": ["a", "b", "c"],
                    }
                }
            ]
        }
        result = _parse_v1_logprobs(body, token_ids=[1, 2, 3])
        assert len(result) == 3
        assert result[1][2] == pytest.approx(-0.5)

    def test_content_format(self) -> None:
        body = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"id": 1, "token": "a", "logprob": None},
                            {"id": 2, "token": "b", "logprob": -0.5},
                            {"id": 3, "token": "c", "logprob": -1.2},
                        ]
                    }
                }
            ]
        }
        result = _parse_v1_logprobs(body, token_ids=[1, 2, 3])
        assert len(result) == 3
        assert result[1][2] == pytest.approx(-0.5)
        assert result[2][3] == pytest.approx(-1.2)

    def test_missing_choices(self) -> None:
        with pytest.raises(LLMScoringError, match="missing choices"):
            _parse_v1_logprobs({}, token_ids=[1])

    def test_missing_logprobs(self) -> None:
        with pytest.raises(LLMScoringError, match="missing logprobs"):
            _parse_v1_logprobs({"choices": [{}]}, token_ids=[1])

    def test_missing_token_logprobs(self) -> None:
        with pytest.raises(LLMScoringError, match="missing token_logprobs or content"):
            _parse_v1_logprobs(
                {"choices": [{"logprobs": {}}]}, token_ids=[1]
            )


class TestLlamaServerCausalLMAdapter:
    def _make_adapter_with_mock(
        self, responses: list[_MockResponse]
    ) -> tuple[LlamaServerCausalLMAdapter, _MockSession]:
        adapter = LlamaServerCausalLMAdapter(
            endpoint="http://127.0.0.1:8080",
            timeout=5,
        )
        mock_session = _MockSession(responses)
        adapter._session = mock_session
        return adapter, mock_session

    def _make_v1_response(self, token_logprobs: list[float | None]) -> _MockResponse:
        return _MockResponse(
            200,
            {
                "choices": [
                    {
                        "logprobs": {
                            "token_logprobs": token_logprobs,
                            "tokens": ["x"] * len(token_logprobs),
                        }
                    }
                ]
            },
        )

    def _make_detok_response(self, text: str = "dummy") -> _MockResponse:
        return _MockResponse(200, {"content": text})

    async def test_forward_batch_single_item(self) -> None:
        adapter, mock = self._make_adapter_with_mock(
            [self._make_detok_response(), self._make_v1_response([None, -0.5, -1.2])]
        )
        result = await adapter.forward_batch([[1, 2]], [[1, 1]])
        assert len(result.items) == 1
        assert len(result.items[0].logits) == 2
        assert mock.post_calls[1]["json"]["echo"] is True
        assert mock.post_calls[1]["json"]["logprobs"] == 1

    async def test_forward_batch_multiple_items(self) -> None:
        adapter, _ = self._make_adapter_with_mock(
            [
                self._make_detok_response(),
                self._make_v1_response([None, -0.3]),
                self._make_detok_response(),
                self._make_v1_response([None, -0.7, -0.2]),
            ]
        )
        result = await adapter.forward_batch([[1], [2, 3]], [[1], [1, 1]])
        assert len(result.items) == 2

    async def test_forward_batch_strips_padding(self) -> None:
        adapter, mock = self._make_adapter_with_mock(
            [self._make_detok_response(), self._make_v1_response([None, -0.5, -1.0])]
        )
        await adapter.forward_batch([[1, 2, 0, 0]], [[1, 1, 0, 0]])
        sent_tokens = mock.post_calls[0]["json"]["tokens"]
        assert sent_tokens == [1, 2]

    async def test_forward_batch_empty(self) -> None:
        adapter = LlamaServerCausalLMAdapter(endpoint="http://127.0.0.1:8080")
        result = await adapter.forward_batch([], [])
        assert result.items == ()

    async def test_non_200_raises(self) -> None:
        adapter, _ = self._make_adapter_with_mock(
            [_MockResponse(500, "internal server error")]
        )
        with pytest.raises(LLMScoringError, match="HTTP 500"):
            await adapter.forward_batch([[1]], [[1]])

    async def test_connection_error_raises(self) -> None:
        adapter = LlamaServerCausalLMAdapter(endpoint="http://127.0.0.1:9999")

        class FailingSession:
            closed = False

            def post(self, url: str, json: dict | None = None) -> Any:
                raise ConnectionRefusedError("connection refused")

            async def close(self) -> None:
                pass

        adapter._session = FailingSession()
        with pytest.raises(LLMScoringError, match="detokenize failed"):
            await adapter.forward_batch([[1]], [[1]])

    async def test_describe_returns_metadata(self) -> None:
        adapter = LlamaServerCausalLMAdapter(
            endpoint="http://10.0.0.1:7070",
            timeout=45,
            served_model_name="qwen2.5-3b",
        )
        meta = adapter.describe()
        assert meta["backend"] == "llama_server"
        assert meta["endpoint"] == "http://10.0.0.1:7070"
        assert meta["served_model_name"] == "qwen2.5-3b"
        assert meta["timeout"] == 45

    async def test_aclose_closes_session(self) -> None:
        adapter, mock = self._make_adapter_with_mock([])
        await adapter.aclose()
        assert mock.closed is True
