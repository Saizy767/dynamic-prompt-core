"""Unit tests for the llama-server HTTP model adapter (logit_bias forcing)."""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.llama_server_model_adapter import (
    LlamaServerCausalLMAdapter,
)
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer


def _completion_response(token_id: int, logprob: float) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "logprobs": {
                    "content": [
                        {
                            "id": token_id,
                            "token": "x",
                            "bytes": [120],
                            "logprob": logprob,
                            "top_logprobs": [],
                        }
                    ]
                },
                "finish_reason": "length",
            }
        ]
    }


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
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.closed = False
        self.post_calls: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any] | None = None) -> _MockResponse:
        self.post_calls.append({"url": url, "json": json})
        if not self._responses:
            raise AssertionError("no more mock responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def close(self) -> None:
        self.closed = True


def _make_adapter(
    responses: list[Any],
) -> tuple[LlamaServerCausalLMAdapter, _MockSession]:
    adapter = LlamaServerCausalLMAdapter(
        endpoint="http://127.0.0.1:8080", timeout=5, max_concurrency=2
    )
    session = _MockSession(responses)
    adapter._session = session
    return adapter, session


class TestForcedTokenScoring:
    async def test_single_token_candidate(self) -> None:
        adapter, session = _make_adapter([_MockResponse(200, _completion_response(16, -12.17))])
        result = await adapter.forward_batch(
            [[1, 2, 3, 16]],
            [[1, 1, 1, 1]],
            prefix_token_counts=[3],
            candidate_token_ids=[[16]],
        )
        assert len(session.post_calls) == 1
        call = session.post_calls[0]
        assert call["url"] == "http://127.0.0.1:8080/v1/completions"
        payload = call["json"]
        assert payload["prompt"] == [1, 2, 3]
        assert payload["logit_bias"] == {"16": 100}
        assert payload["max_tokens"] == 1
        assert payload["logprobs"] == 1
        assert payload["temperature"] == 0

        item = result.items[0]
        assert len(item.logits) == 4
        score = LogitScorer().score(
            prefix_token_count=3, candidate_token_ids=[16], sequence_logits=item
        )
        assert score == pytest.approx(-12.17, abs=1e-9)

    async def test_two_candidates_one_request_each(self) -> None:
        adapter, session = _make_adapter(
            [
                _MockResponse(200, _completion_response(15, -14.62)),
                _MockResponse(200, _completion_response(16, -12.17)),
            ]
        )
        result = await adapter.forward_batch(
            [[1, 2, 3, 15], [1, 2, 3, 16]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            prefix_token_counts=[3, 3],
            candidate_token_ids=[[15], [16]],
        )
        assert len(session.post_calls) == 2
        scorer = LogitScorer()
        score0 = scorer.score(3, [15], result.items[0])
        score1 = scorer.score(3, [16], result.items[1])
        assert score0 == pytest.approx(-14.62, abs=1e-9)
        assert score1 == pytest.approx(-12.17, abs=1e-9)

    async def test_multi_token_candidate_prompt_grows(self) -> None:
        adapter, session = _make_adapter(
            [
                _MockResponse(200, _completion_response(7, -0.5)),
                _MockResponse(200, _completion_response(8, -1.5)),
            ]
        )
        result = await adapter.forward_batch(
            [[1, 2, 3, 7, 8]],
            [[1, 1, 1, 1, 1]],
            prefix_token_counts=[3],
            candidate_token_ids=[[7, 8]],
        )
        prompts = [c["json"]["prompt"] for c in session.post_calls]
        biases = [c["json"]["logit_bias"] for c in session.post_calls]
        assert prompts == [[1, 2, 3], [1, 2, 3, 7]]
        assert biases == [{"7": 100}, {"8": 100}]
        score = LogitScorer().score(3, [7, 8], result.items[0])
        assert score == pytest.approx((-0.5 + -1.5) / 2, abs=1e-9)

    async def test_padding_stripped_from_prompt(self) -> None:
        adapter, session = _make_adapter([_MockResponse(200, _completion_response(16, -1.0))])
        await adapter.forward_batch(
            [[1, 2, 3, 16, 0, 0]],
            [[1, 1, 1, 1, 0, 0]],
            prefix_token_counts=[3],
            candidate_token_ids=[[16]],
        )
        assert session.post_calls[0]["json"]["prompt"] == [1, 2, 3]

    async def test_empty_batch(self) -> None:
        adapter, session = _make_adapter([])
        result = await adapter.forward_batch([], [], prefix_token_counts=[], candidate_token_ids=[])
        assert result.items == ()
        assert session.post_calls == []


class TestRequiredKwargs:
    async def test_missing_kwargs_rejected(self) -> None:
        adapter, _ = _make_adapter([])
        with pytest.raises(LLMScoringError, match="requires prefix_token_counts"):
            await adapter.forward_batch([[1, 2]], [[1, 1]])

    async def test_batch_size_mismatch_rejected(self) -> None:
        adapter, _ = _make_adapter([])
        with pytest.raises(LLMScoringError, match="must.*match the batch size"):
            await adapter.forward_batch(
                [[1, 2]], [[1, 1]], prefix_token_counts=[1, 1], candidate_token_ids=[[5]]
            )


class TestResponseValidation:
    async def test_identity_mismatch_rejected(self) -> None:
        adapter, _ = _make_adapter([_MockResponse(200, _completion_response(99, -1.0))])
        with pytest.raises(LLMScoringError, match="generated token id 99"):
            await adapter.forward_batch(
                [[1, 2, 3, 16]],
                [[1, 1, 1, 1]],
                prefix_token_counts=[3],
                candidate_token_ids=[[16]],
            )

    async def test_nan_logprob_rejected(self) -> None:
        adapter, _ = _make_adapter([_MockResponse(200, _completion_response(16, float("nan")))])
        with pytest.raises(LLMScoringError, match="NaN"):
            await adapter.forward_batch(
                [[1, 2, 3, 16]],
                [[1, 1, 1, 1]],
                prefix_token_counts=[3],
                candidate_token_ids=[[16]],
            )

    async def test_missing_choices_rejected(self) -> None:
        adapter, _ = _make_adapter([_MockResponse(200, {"unexpected": True})])
        with pytest.raises(LLMScoringError, match="missing choices"):
            await adapter.forward_batch(
                [[1, 2, 3, 16]],
                [[1, 1, 1, 1]],
                prefix_token_counts=[3],
                candidate_token_ids=[[16]],
            )

    async def test_missing_logprobs_rejected(self) -> None:
        adapter, _ = _make_adapter([_MockResponse(200, {"choices": [{"index": 0}]})])
        with pytest.raises(LLMScoringError, match="missing logprobs"):
            await adapter.forward_batch(
                [[1, 2, 3, 16]],
                [[1, 1, 1, 1]],
                prefix_token_counts=[3],
                candidate_token_ids=[[16]],
            )

    async def test_missing_content_rejected(self) -> None:
        adapter, _ = _make_adapter([_MockResponse(200, {"choices": [{"logprobs": {}}]})])
        with pytest.raises(LLMScoringError, match="missing logprobs content"):
            await adapter.forward_batch(
                [[1, 2, 3, 16]],
                [[1, 1, 1, 1]],
                prefix_token_counts=[3],
                candidate_token_ids=[[16]],
            )

    async def test_http_error_surfaces_with_status(self) -> None:
        adapter, _ = _make_adapter([_MockResponse(500, {"error": "boom"})])
        with pytest.raises(LLMScoringError, match="HTTP 500"):
            await adapter.forward_batch(
                [[1, 2, 3, 16]],
                [[1, 1, 1, 1]],
                prefix_token_counts=[3],
                candidate_token_ids=[[16]],
            )


class TestRetries:
    async def test_transient_disconnect_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, session = _make_adapter(
            [
                aiohttp.ServerDisconnectedError("boom"),
                _MockResponse(200, _completion_response(16, -2.0)),
            ]
        )
        monkeypatch.setattr(adapter, "_get_session", lambda: session)
        result = await adapter.forward_batch(
            [[1, 2, 3, 16]],
            [[1, 1, 1, 1]],
            prefix_token_counts=[3],
            candidate_token_ids=[[16]],
        )
        assert len(session.post_calls) == 2
        score = LogitScorer().score(3, [16], result.items[0])
        assert score == pytest.approx(-2.0, abs=1e-9)

    async def test_retries_exhausted_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, session = _make_adapter([aiohttp.ServerDisconnectedError("boom")] * 3)
        monkeypatch.setattr(adapter, "_get_session", lambda: session)
        with pytest.raises(LLMScoringError, match="after 3 retries"):
            await adapter.forward_batch(
                [[1, 2, 3, 16]],
                [[1, 1, 1, 1]],
                prefix_token_counts=[3],
                candidate_token_ids=[[16]],
            )


class TestLifecycle:
    async def test_describe_metadata(self) -> None:
        adapter, _ = _make_adapter([])
        meta = adapter.describe()
        assert meta["backend"] == "llama_server"
        assert meta["endpoint"] == "http://127.0.0.1:8080"
        assert meta["forcing_bias"] == 100
        assert meta["timeout"] == 5
        assert set(meta) == {
            "backend",
            "endpoint",
            "served_model_name",
            "forcing_bias",
            "timeout",
            "model_path",
            "model_checksum",
        }

    async def test_describe_checksums_gguf_file(self, tmp_path) -> None:
        """describe() must identify the model file, not just the endpoint."""
        import hashlib

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"model-bytes")
        adapter = LlamaServerCausalLMAdapter(
            endpoint="http://127.0.0.1:8080",
            timeout=5,
            gguf_path=str(gguf),
        )
        meta = adapter.describe()
        expected = hashlib.sha256(b"model-bytes").hexdigest()
        assert meta["model_path"] == str(gguf)
        assert meta["model_checksum"] == expected
        assert adapter.describe()["model_checksum"] == expected

    async def test_describe_missing_gguf_gives_none_checksum(self, tmp_path) -> None:
        adapter = LlamaServerCausalLMAdapter(
            endpoint="http://127.0.0.1:8080",
            timeout=5,
            gguf_path=str(tmp_path / "absent.gguf"),
        )
        meta = adapter.describe()
        assert meta["model_checksum"] is None

    async def test_aclose_closes_session(self) -> None:
        adapter, session = _make_adapter([])
        await adapter.aclose()
        assert session.closed is True
