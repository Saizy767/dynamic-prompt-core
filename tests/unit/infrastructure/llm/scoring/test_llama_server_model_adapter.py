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
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import _log_probability


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
        # Position 0 predicts token_ids[1]=2 with logprob token_logprobs[1]=-0.5
        assert result[0][2] == pytest.approx(-0.5)
        # Position 1 predicts token_ids[2]=3 with logprob token_logprobs[2]=-1.2
        assert result[1][3] == pytest.approx(-1.2)

    def test_log_softmax_preserves_real_logprob(self) -> None:
        """The core invariant: log_softmax(logits[i])[token_ids[i+1]] == token_logprobs[i+1].

        This is the semantic contract that was broken by the _LOGIT_FLOOR
        constant: with -100.0 for all other tokens, log_softmax collapsed
        every token's log-prob to ~0.0, producing 50/50 for all candidates.
        """
        token_logprobs = [None, -0.7, -0.3]
        token_ids = [1, 2, 3]
        result = _build_logits_from_logprobs(token_logprobs, token_ids, vocab_size=10)
        # Position 0 predicts token 2 with logprob -0.7
        assert _log_probability(result[0], 2) == pytest.approx(-0.7, abs=1e-9)
        # Position 1 predicts token 3 with logprob -0.3
        assert _log_probability(result[1], 3) == pytest.approx(-0.3, abs=1e-9)

    def test_other_tokens_not_floor(self) -> None:
        """Non-actual tokens must use the correct formula, not _LOGIT_FLOOR."""
        result = _build_logits_from_logprobs([None, -0.5], [1, 2], vocab_size=10)
        other_logits = [result[0][i] for i in range(10) if i != 2]
        assert not all(v == -100.0 for v in other_logits)
        assert all(v < -0.5 for v in other_logits)

    def test_none_logprob_first_position(self) -> None:
        result = _build_logits_from_logprobs([None], [5], vocab_size=10)
        assert all(v == -100.0 for v in result[0])

    def test_none_logprob_at_scoring_position_raises(self) -> None:
        """None at a position with a successor token must raise, not silently floor."""
        with pytest.raises(LLMScoringError, match="is None at position"):
            _build_logits_from_logprobs([None, None, -0.5], [1, 2, 3], vocab_size=10)

    def test_none_logprob_mid_sequence_raises(self) -> None:
        """None in the middle of the sequence (not just position 0) must raise."""
        with pytest.raises(LLMScoringError, match="is None at position"):
            _build_logits_from_logprobs([None, -0.3, None, -0.7], [1, 2, 3, 4], vocab_size=10)

    def test_nan_rejected(self) -> None:
        with pytest.raises(LLMScoringError, match="NaN"):
            _build_logits_from_logprobs([None, float("nan")], [0, 1], vocab_size=10)

    def test_different_logprobs_produce_different_scores(self) -> None:
        """End-to-end: different log-probs must yield different softmax probs.

        Regression test for the 50/50 bug: with the _LOGIT_FLOOR constant,
        all candidates got log_softmax ≈ 0.0, making softmax always 50/50.
        Uses causal alignment: prefix_count=2, candidate at position 1.
        """
        from dynamic_prompt_core.infrastructure.llm.scoring.candidate_scorer import (
            _softmax,
        )

        vocab = 100
        # Candidate A: strong support (lp = -0.2)
        # Sequence: [prefix_0, prefix_1, cand_a], LogitScorer reads position 1
        logits_a = _build_logits_from_logprobs(
            [None, -0.5, -0.2], [0, 1, 10], vocab_size=vocab
        )
        score_a = _log_probability(logits_a[1], 10)
        # Candidate B: weak support (lp = -2.0)
        logits_b = _build_logits_from_logprobs(
            [None, -0.5, -2.0], [0, 1, 20], vocab_size=vocab
        )
        score_b = _log_probability(logits_b[1], 20)

        probs = _softmax([score_a, score_b])
        assert probs[0] > 0.5
        assert probs[1] < 0.5
        assert probs[0] != pytest.approx(0.5, abs=1e-6)


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
        # Position 0 predicts token_ids[1]=2 with logprob -0.5
        assert result[0][2] == pytest.approx(-0.5)

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
        # Position 0 predicts token_ids[1]=2 with logprob -0.5
        assert result[0][2] == pytest.approx(-0.5)
        # Position 1 predicts token_ids[2]=3 with logprob -1.2
        assert result[1][3] == pytest.approx(-1.2)

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

    def test_token_id_mismatch_raises(self) -> None:
        """API returning different token IDs than the originals must raise."""
        body = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"id": 10, "token": "a", "logprob": None},
                            {"id": 20, "token": "b", "logprob": -0.5},
                            {"id": 30, "token": "c", "logprob": -1.2},
                        ]
                    }
                }
            ]
        }
        with pytest.raises(LLMScoringError, match="do not match"):
            _parse_v1_logprobs(body, token_ids=[1, 2, 3])

    def test_token_id_match_succeeds(self) -> None:
        """API returning the same token IDs as the originals succeeds."""
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
        assert result[0][2] == pytest.approx(-0.5)


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

    def _make_native_response(
        self, token_logprobs: list[float | None], token_ids: list[int] | None = None,
    ) -> _MockResponse:
        ids = token_ids or list(range(len(token_logprobs)))
        return _MockResponse(
            200,
            {
                "content": "",
                "prompt_tokens": ids,
                "prompt_logprobs": token_logprobs,
            },
        )

    async def test_forward_batch_single_item(self) -> None:
        adapter, mock = self._make_adapter_with_mock(
            [self._make_native_response([None, -0.5], [1, 2])]
        )
        result = await adapter.forward_batch([[1, 2]], [[1, 1]])
        assert len(result.items) == 1
        assert len(result.items[0].logits) == 2
        assert mock.post_calls[0]["json"]["n_predict"] == 0
        assert mock.post_calls[0]["json"]["prompt_logprobs"] == 1

    async def test_forward_batch_multiple_items(self) -> None:
        adapter, _ = self._make_adapter_with_mock(
            [
                self._make_native_response([None], [1]),
                self._make_native_response([None, -0.7], [2, 3]),
            ]
        )
        result = await adapter.forward_batch([[1], [2, 3]], [[1], [1, 1]])
        assert len(result.items) == 2

    async def test_forward_batch_strips_padding(self) -> None:
        adapter, mock = self._make_adapter_with_mock(
            [self._make_native_response([None, -0.5], [1, 2])]
        )
        await adapter.forward_batch([[1, 2, 0, 0]], [[1, 1, 0, 0]])
        sent_prompt = mock.post_calls[0]["json"]["prompt"]
        assert sent_prompt == [1, 2]

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
        with pytest.raises(LLMScoringError, match="request failed"):
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
