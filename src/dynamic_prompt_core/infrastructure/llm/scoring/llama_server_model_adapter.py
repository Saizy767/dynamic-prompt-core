"""HTTP-based model adapter for candidate scoring via llama-server.

``LlamaServerCausalLMAdapter`` implements the ``BatchedCausalLanguageModel`` seam
by calling the ``llama-server`` OpenAI-compatible API.  Log probabilities for
all input tokens are retrieved via ``POST /v1/completions`` with ``echo: true``
and ``logprobs: 0``, which returns ``token_logprobs`` — the log probability of
each token given its preceding context.

Since ``LogitScorer`` applies ``log_softmax`` to the logits, we reconstruct a
logit vector per position that yields the correct log probability:

    logits[i][actual_token] = log_prob[i]
    logits[i][other]        = log((1 - exp(log_prob[i])) / (V - 1))

This guarantees ``log_softmax(logits[i])[actual_token] == log_prob[i]``.

For servers that support ``--logits-all`` (older builds), the raw ``logits``
field from ``/completion`` is preferred when present.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import aiohttp

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedLogits,
    SequenceLogits,
)

log = logging.getLogger(__name__)

_LOGIT_FLOOR = -100.0
_MAX_RETRIES = 3
_RETRY_DELAY_S = 0.5
_DEFAULT_VOCAB_SIZE = 32000


def _build_logits_from_logprobs(
    token_logprobs: list[float | None],
    token_ids: list[int],
    vocab_size: int,
) -> list[list[float]]:
    """Construct a logit vector per position from token log probabilities.

    For each position ``i``, the logit vector is set so that
    ``log_softmax(logits[i])[token_ids[i+1]] == token_logprobs[i+1]``.

    ``token_logprobs[0]`` is typically ``None`` (no preceding context); it gets
    a zero logit vector.
    """
    seq_len = len(token_ids)
    result: list[list[float]] = []

    for i in range(seq_len):
        logit = [_LOGIT_FLOOR] * vocab_size

        if i < len(token_logprobs):
            raw_lp = token_logprobs[i]
            if raw_lp is not None:
                lp = float(raw_lp)
                if math.isnan(lp) or math.isinf(lp):
                    raise LLMScoringError(
                        f"token_logprobs[{i}] is NaN or infinite: {lp}"
                    )
                next_token = token_ids[i]
                if next_token >= vocab_size:
                    vocab_size = next_token + 1
                    logit.extend([_LOGIT_FLOOR] * (next_token + 1 - len(logit)))
                if 0 <= next_token < len(logit):
                    logit[next_token] = lp

        result.append(logit)

    return result


def _parse_v1_logprobs(
    body: dict[str, Any],
    token_ids: list[int],
) -> list[list[float]]:
    """Parse the /v1/completions response and build logits from token_logprobs."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMScoringError(
            f"v1/completions response missing choices; excerpt: {str(body)[:200]}"
        )

    logprobs_data = choices[0].get("logprobs")
    if not isinstance(logprobs_data, dict):
        raise LLMScoringError(
            f"v1/completions response missing logprobs; excerpt: {str(body)[:200]}"
        )

    token_logprobs = logprobs_data.get("token_logprobs")
    if not isinstance(token_logprobs, list):
        content = logprobs_data.get("content")
        if isinstance(content, list):
            token_logprobs = [
                entry.get("logprob") if isinstance(entry, dict) else None
                for entry in content
            ]
        else:
            raise LLMScoringError(
                f"v1/completions response missing token_logprobs or content; "
                f"excerpt: {str(body)[:200]}"
            )

    vocab_size = _DEFAULT_VOCAB_SIZE
    return _build_logits_from_logprobs(token_logprobs, token_ids, vocab_size)


def _extract_logits_from_completion(
    body: dict[str, Any],
    expected_seq_len: int,
) -> list[list[float]]:
    """Parse raw ``logits`` from /completion response (for --logits-all servers)."""
    raw_logits = body.get("logits")
    if raw_logits is None:
        return None  # type: ignore[return-value]

    if not isinstance(raw_logits, list):
        raise LLMScoringError(
            f"logits field is not a list (got {type(raw_logits).__name__})"
        )

    if len(raw_logits) == expected_seq_len + 1:
        raw_logits = raw_logits[:expected_seq_len]
    elif len(raw_logits) != expected_seq_len:
        raise LLMScoringError(
            f"logit sequence length {len(raw_logits)} does not match "
            f"input token count {expected_seq_len}"
        )

    result: list[list[float]] = []
    for i, position in enumerate(raw_logits):
        if not isinstance(position, (list, tuple)):
            raise LLMScoringError(
                f"logit position {i} is not a list "
                f"(got {type(position).__name__})"
            )
        values = [float(x) for x in position]
        if not values:
            raise LLMScoringError(f"logit position {i} is empty")
        for v in values:
            if math.isnan(v) or math.isinf(v):
                raise LLMScoringError(
                    f"logit position {i} contains NaN or infinite values"
                )
        result.append(values)

    return result


class LlamaServerCausalLMAdapter:
    """Concrete ``BatchedCausalLanguageModel`` backed by llama-server HTTP API.

    Uses the OpenAI-compatible ``/v1/completions`` endpoint with ``echo: true``
    and ``logprobs: 0`` to get ``token_logprobs`` for all input tokens.  The
    log probabilities are converted to logits that produce the correct scores
    when ``LogitScorer`` applies ``log_softmax``.

    Concurrency is limited by a semaphore (default 1).  Transient errors are
    retried up to 3 times with backoff.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: int = 30,
        served_model_name: str | None = None,
        max_concurrency: int = 1,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._v1_completions_url = f"{self._endpoint}/v1/completions"
        self._completion_url = f"{self._endpoint}/completion"
        self._timeout = timeout
        self._served_model_name = served_model_name
        self._session: aiohttp.ClientSession | None = None
        self._vocab_size: int = 0
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self._semaphore._value,
                force_close=True,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
        return self._session

    async def _detokenize(self, token_ids: list[int]) -> str:
        """Convert token IDs to text via POST /detokenize."""
        session = self._get_session()
        payload = {"tokens": token_ids}
        try:
            async with session.post(
                f"{self._endpoint}/detokenize", json=payload
            ) as resp:
                if resp.status != 200:
                    raise LLMScoringError(
                        f"detokenize returned HTTP {resp.status}"
                    )
                body = await resp.json()
        except LLMScoringError:
            raise
        except Exception as exc:
            raise LLMScoringError(f"detokenize failed: {exc}") from exc

        text = body.get("content", "")
        return str(text)

    async def _forward_single(
        self,
        token_ids: list[int],
    ) -> SequenceLogits:
        """Send one prompt and return per-position logits.

        Detokenizes token IDs to text first, then calls ``/v1/completions``
        with ``echo: true`` and ``logprobs: 1`` to get ``token_logprobs``
        for all input tokens.  Retries on transient errors.
        """
        text = await self._detokenize(token_ids)

        payload: dict[str, Any] = {
            "prompt": text,
            "max_tokens": 1,
            "logprobs": 1,
            "echo": True,
            "temperature": 0,
        }
        if self._served_model_name is not None:
            payload["model"] = self._served_model_name

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._do_request(token_ids, payload)
            except aiohttp.ServerDisconnectedError as exc:
                last_exc = exc
                log.warning(
                    "llama-server disconnected (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, _RETRY_DELAY_S,
                )
                await self._safe_close_session()
                await asyncio.sleep(_RETRY_DELAY_S)
            except aiohttp.ClientConnectorError as exc:
                last_exc = exc
                log.warning(
                    "llama-server connection error (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, _RETRY_DELAY_S,
                )
                await self._safe_close_session()
                await asyncio.sleep(_RETRY_DELAY_S)

        raise LLMScoringError(
            f"llama-server request failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc

    async def _do_request(
        self,
        token_ids: list[int],
        payload: dict[str, Any],
    ) -> SequenceLogits:
        """Execute a single HTTP request and parse the response."""
        session = self._get_session()
        try:
            async with session.post(self._v1_completions_url, json=payload) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    excerpt = body_text[:300]
                    raise LLMScoringError(
                        f"llama-server returned HTTP {resp.status} "
                        f"from {self._v1_completions_url}: {excerpt}"
                    )
                body = await resp.json()
        except aiohttp.ServerDisconnectedError:
            raise
        except aiohttp.ClientConnectorError:
            raise
        except aiohttp.ServerTimeoutError as exc:
            raise LLMScoringError(
                f"request to llama-server at {self._endpoint} timed out "
                f"after {self._timeout}s: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise LLMScoringError(
                f"request to llama-server at {self._endpoint} timed out "
                f"after {self._timeout}s"
            ) from exc
        except LLMScoringError:
            raise
        except Exception as exc:
            raise LLMScoringError(
                f"llama-server request failed: {exc}"
            ) from exc

        per_position = _parse_v1_logprobs(body, token_ids)

        if self._vocab_size == 0 and per_position:
            self._vocab_size = len(per_position[0])

        return SequenceLogits(logits=tuple(tuple(p) for p in per_position))

    async def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        """Evaluate each batch item and return per-item logits."""
        if not input_ids:
            return BatchedLogits(items=())

        unpadded_items: list[list[int]] = []
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            unpadded = [
                token_id
                for token_id, m in zip(ids, mask, strict=True)
                if m == 1
            ]
            unpadded_items.append(unpadded)

        log.debug(
            "llama-server forward_batch: %d items, sizes=%s",
            len(unpadded_items),
            [len(x) for x in unpadded_items],
        )

        async def _gated_forward(tokens: list[int]) -> SequenceLogits:
            async with self._semaphore:
                return await self._forward_single(tokens)

        try:
            results = await asyncio.gather(
                *(_gated_forward(tokens) for tokens in unpadded_items)
            )
        except LLMScoringError:
            await self._safe_close_session()
            raise

        return BatchedLogits(items=tuple(results))

    async def _safe_close_session(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def aclose(self) -> None:
        """Close the underlying aiohttp.ClientSession."""
        await self._safe_close_session()

    def describe(self) -> dict[str, Any]:
        """Return backend metadata for run identity and resume."""
        return {
            "backend": "llama_server",
            "endpoint": self._endpoint,
            "served_model_name": self._served_model_name,
            "vocab_size": self._vocab_size,
            "timeout": self._timeout,
        }
