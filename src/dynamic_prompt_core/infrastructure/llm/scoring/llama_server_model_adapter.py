"""HTTP-based model adapter for candidate scoring via llama-server.

``LlamaServerCausalLMAdapter`` implements the ``BatchedCausalLanguageModel``
seam by calling the llama-server OpenAI-compatible ``/v1/completions``
endpoint.  Each candidate token is scored by one request that forces the
server to generate exactly that token via ``logit_bias`` (bias +100 on the
token ID) and reads back the token's log-probability from the response.

Validated live against llama.cpp 0.5.0: the reported log-prob is the token's
true full-vocabulary log-prob under the unmodified model distribution — the
forcing bias constrains only the sampling choice, not the reported value
(forced ``0`` -> -14.6238, ``1`` -> -12.167518, byte-identical to
grammar-forced values).  There is no top-N window, no floor for missing
tokens, and no renormalization arithmetic.

Because the prompt for candidate token ``k`` is fully determined upfront
(``prefix + candidate_tokens[:k]``), every request in a batch is issued
concurrently, bounded by the adapter semaphore and the server's prompt
cache (shared prefixes are evaluated once server-side).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
from typing import Any

import aiohttp

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedLogits,
    SequenceLogits,
    SparseLogitVector,
)

log = logging.getLogger(__name__)

_LOGIT_FLOOR = -100.0
_FORCING_BIAS = 100.0
_MAX_RETRIES = 3
_RETRY_DELAY_S = 0.5


class LlamaServerCausalLMAdapter:
    """Concrete ``BatchedCausalLanguageModel`` backed by llama-server HTTP API.

    Scores each candidate token with one ``POST /v1/completions`` request:
    ``prompt`` is the token-ID prefix (plus already-forced candidate tokens),
    ``max_tokens: 1``, ``logprobs: 1``, greedy sampling, and
    ``logit_bias {<token_id>: 100}`` so the server must generate exactly the
    candidate token.  The reported log-prob is the token's true
    full-vocabulary log-prob.

    ``forward_batch`` requires the ``prefix_token_counts`` and
    ``candidate_token_ids`` kwargs; without them the mechanism cannot know
    which tokens to force, so the call is rejected.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: int = 30,
        served_model_name: str | None = None,
        max_concurrency: int = 1,
        gguf_path: str | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._v1_completions_url = f"{self._endpoint}/v1/completions"
        self._timeout = timeout
        self._served_model_name = served_model_name
        self._gguf_path = gguf_path
        self._gguf_checksum: str | None = None
        self._gguf_checksum_done = False
        self._max_concurrency = max_concurrency
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=self._max_concurrency)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
        return self._session

    async def _forced_token_logprob(
        self,
        prompt_token_ids: list[int],
        forced_token_id: int,
    ) -> float:
        """One forced-generation request; return the token's true log-prob.

        Retries transient transport failures (server disconnect, connection
        error) up to ``_MAX_RETRIES`` times with backoff.
        """
        payload: dict[str, Any] = {
            "prompt": prompt_token_ids,
            "max_tokens": 1,
            "logprobs": 1,
            "temperature": 0,
            "logit_bias": {str(forced_token_id): _FORCING_BIAS},
        }
        if self._served_model_name is not None:
            payload["model"] = self._served_model_name

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._do_request(payload, forced_token_id)
            except aiohttp.ServerDisconnectedError as exc:
                last_exc = exc
                log.warning(
                    "llama-server disconnected (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    _RETRY_DELAY_S,
                )
                await self._safe_close_session()
                await asyncio.sleep(_RETRY_DELAY_S)
            except aiohttp.ClientConnectorError as exc:
                last_exc = exc
                log.warning(
                    "llama-server connection error (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    _RETRY_DELAY_S,
                )
                await self._safe_close_session()
                await asyncio.sleep(_RETRY_DELAY_S)

        raise LLMScoringError(
            f"llama-server request failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc

    async def _do_request(
        self,
        payload: dict[str, Any],
        forced_token_id: int,
    ) -> float:
        """Execute one request and extract the forced token's log-prob."""
        session = self._get_session()
        try:
            async with session.post(self._v1_completions_url, json=payload) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    raise LLMScoringError(
                        f"llama-server returned HTTP {resp.status} "
                        f"from {self._v1_completions_url}: {body_text[:300]}"
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
                f"request to llama-server at {self._endpoint} timed out after {self._timeout}s"
            ) from exc
        except LLMScoringError:
            raise
        except Exception as exc:
            raise LLMScoringError(f"llama-server request failed: {exc}") from exc

        return _parse_forced_logprob(body, forced_token_id)

    async def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        prefix_token_counts: list[int] | None = None,
        candidate_token_ids: list[list[int]] | None = None,
    ) -> BatchedLogits:
        """Score every candidate token of every batch item.

        ``prefix_token_counts`` and ``candidate_token_ids`` are required:
        they define which token to force at each step and how the prompt
        grows.  All requests are issued concurrently (prompts are fully
        determined upfront), bounded by the adapter semaphore.
        """
        if prefix_token_counts is None or candidate_token_ids is None:
            raise LLMScoringError(
                "forward_batch requires prefix_token_counts and "
                "candidate_token_ids; the logit_bias forcing mechanism "
                "cannot determine which tokens to force without them"
            )
        if len(prefix_token_counts) != len(input_ids) or len(candidate_token_ids) != len(input_ids):
            raise LLMScoringError(
                f"prefix_token_counts ({len(prefix_token_counts)}) and "
                f"candidate_token_ids ({len(candidate_token_ids)}) must "
                f"match the batch size ({len(input_ids)})"
            )
        if not input_ids:
            return BatchedLogits(items=())

        async def _gated_logprob(prompt_ids: list[int], forced_id: int) -> float:
            async with self._semaphore:
                return await self._forced_token_logprob(prompt_ids, forced_id)

        async def _score_item(
            unpadded: list[int],
            prefix_count: int,
            cand_ids: list[int],
        ) -> SequenceLogits:
            vocab_size = max(unpadded + cand_ids) + 1 if unpadded else 1
            per_position: list[SparseLogitVector] = [
                SparseLogitVector(-1, _LOGIT_FLOOR, _LOGIT_FLOOR, vocab_size) for _ in unpadded
            ]
            requests = await asyncio.gather(
                *(
                    _gated_logprob(unpadded[: prefix_count + k], token_id)
                    for k, token_id in enumerate(cand_ids)
                )
            )
            for k, (token_id, lp) in enumerate(zip(cand_ids, requests, strict=True)):
                position = prefix_count + k - 1
                if 0 <= position < len(per_position):
                    per_position[position] = SparseLogitVector(
                        token_id, lp, _LOGIT_FLOOR, vocab_size
                    )
            return SequenceLogits(logits=tuple(per_position))

        unpadded_items: list[list[int]] = []
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            unpadded_items.append(
                [token_id for token_id, m in zip(ids, mask, strict=True) if m == 1]
            )

        log.debug(
            "llama-server forward_batch: %d items, sizes=%s",
            len(unpadded_items),
            [len(x) for x in unpadded_items],
        )

        try:
            results = await asyncio.gather(
                *(
                    _score_item(unpadded, prefix_count, cand_ids)
                    for unpadded, prefix_count, cand_ids in zip(
                        unpadded_items,
                        prefix_token_counts,
                        candidate_token_ids,
                        strict=True,
                    )
                )
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

    def _model_checksum(self) -> str | None:
        """sha256 of the GGUF file, computed once; None when unavailable.

        The checksum is what resume identity checks compare — the endpoint
        says nothing about which model the server loaded.
        """
        if not self._gguf_checksum_done:
            self._gguf_checksum_done = True
            if self._gguf_path and os.path.isfile(self._gguf_path):
                try:
                    with open(self._gguf_path, "rb") as f:
                        self._gguf_checksum = hashlib.file_digest(f, "sha256").hexdigest()
                except OSError:
                    log.warning(
                        "could not read gguf %s for model checksum",
                        self._gguf_path,
                    )
        return self._gguf_checksum

    def describe(self) -> dict[str, Any]:
        """Return backend metadata for run identity and resume."""
        return {
            "backend": "llama_server",
            "endpoint": self._endpoint,
            "served_model_name": self._served_model_name,
            "forcing_bias": _FORCING_BIAS,
            "timeout": self._timeout,
            "model_path": self._gguf_path,
            "model_checksum": self._model_checksum(),
        }


def _parse_forced_logprob(body: dict[str, Any], forced_token_id: int) -> float:
    """Extract and verify the forced token's log-prob from a response body.

    Verifies the generated token ID matches the forced candidate token ID;
    a mismatch means the server generated a different token (for example an
    older build without ``logit_bias`` support) and the score would belong
    to the wrong token — raise rather than score silently wrong.
    """
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
    content = logprobs_data.get("content")
    if not isinstance(content, list) or not content:
        raise LLMScoringError(
            f"v1/completions response missing logprobs content; excerpt: {str(body)[:200]}"
        )
    entry = content[0]
    if not isinstance(entry, dict) or "logprob" not in entry or "id" not in entry:
        raise LLMScoringError(
            f"v1/completions logprobs entry missing id/logprob; excerpt: {str(body)[:200]}"
        )
    generated_id = int(entry["id"])
    if generated_id != forced_token_id:
        raise LLMScoringError(
            f"server generated token id {generated_id} but token id "
            f"{forced_token_id} was forced; the log-prob belongs to a "
            f"different token"
        )
    lp = float(entry["logprob"])
    if math.isnan(lp) or math.isinf(lp):
        raise LLMScoringError(f"forced token logprob is NaN or infinite: {lp}")
    return lp
