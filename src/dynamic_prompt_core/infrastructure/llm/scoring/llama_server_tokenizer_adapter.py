"""HTTP-based tokenizer adapter for candidate scoring via llama-server.

``LlamaServerBatchTokenizerAdapter`` implements the ``BatchTokenizerAdapter``
seam by calling the ``llama-server`` ``POST /tokenize`` endpoint.  The prefix
and candidate are encoded separately and composed into the full sequence,
preserving the separate-encoding boundary-determination strategy.

Using the server's tokenizer ensures token IDs match exactly what the model
uses for inference, avoiding potential BPE/SentencePiece mismatch between a
local HuggingFace tokenizer and the GGUF model's internal tokenizer.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    BatchTokenization,
)

log = logging.getLogger(__name__)

_PAD_TOKEN_ID = 0
_MAX_RETRIES = 3
_RETRY_DELAY_S = 0.5


class LlamaServerBatchTokenizerAdapter:
    """Concrete ``BatchTokenizerAdapter`` backed by llama-server HTTP API.

    Calls ``POST /tokenize`` for each prefix and candidate separately, composes
    ``prefix_tokens + candidate_tokens``, and records ``prefix_token_count``
    and ``candidate_token_ids`` per item.  Produces ``BatchTokenization`` with
    right padding and per-item boundary metadata.

    Transient errors (``ServerDisconnectedError``) are retried up to 3 times.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: int = 30,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._tokenize_url = f"{self._endpoint}/tokenize"
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=4, force_close=True)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
        return self._session

    async def _tokenize(self, text: str) -> list[int]:
        """Encode text via POST /tokenize and return token IDs.

        Retries on ``ServerDisconnectedError`` up to ``_MAX_RETRIES`` times.
        """
        payload: dict[str, Any] = {"content": text, "add_special": False}

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._do_tokenize_request(payload)
            except aiohttp.ServerDisconnectedError as exc:
                last_exc = exc
                log.warning(
                    "llama-server tokenize disconnected (attempt %d/%d), retrying",
                    attempt + 1, _MAX_RETRIES,
                )
                await self._safe_close_session()
                await asyncio.sleep(_RETRY_DELAY_S)
            except aiohttp.ClientConnectorError as exc:
                last_exc = exc
                log.warning(
                    "llama-server tokenize connection error (attempt %d/%d), retrying",
                    attempt + 1, _MAX_RETRIES,
                )
                await self._safe_close_session()
                await asyncio.sleep(_RETRY_DELAY_S)

        raise LLMScoringError(
            f"llama-server tokenize failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc

    async def _do_tokenize_request(
        self, payload: dict[str, Any]
    ) -> list[int]:
        """Execute a single tokenize HTTP request."""
        session = self._get_session()
        try:
            async with session.post(self._tokenize_url, json=payload) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    excerpt = body_text[:300]
                    raise LLMScoringError(
                        f"llama-server tokenize returned HTTP {resp.status} "
                        f"from {self._tokenize_url}: {excerpt}"
                    )
                body = await resp.json()
        except aiohttp.ServerDisconnectedError:
            raise
        except aiohttp.ClientConnectorError:
            raise
        except aiohttp.ServerTimeoutError as exc:
            raise LLMScoringError(
                f"tokenize request to llama-server at {self._endpoint} "
                f"timed out after {self._timeout}s: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise LLMScoringError(
                f"tokenize request to llama-server at {self._endpoint} "
                f"timed out after {self._timeout}s"
            ) from exc
        except LLMScoringError:
            raise
        except Exception as exc:
            raise LLMScoringError(
                f"llama-server tokenize request failed: {exc}"
            ) from exc

        tokens = body.get("tokens")
        if not isinstance(tokens, list):
            raise LLMScoringError(
                f"llama-server tokenize response missing 'tokens' field "
                f"or not a list (got {type(tokens).__name__})"
            )
        return [int(t) for t in tokens]

    async def encode_batch(
        self,
        prefixes: list[str],
        candidates: list[str],
    ) -> BatchTokenization:
        """Encode ``(prefix, candidate)`` pairs into a padded batch.

        For each pair, encodes ``prefix + candidate`` as one string via
        ``POST /tokenize``, then verifies the prefix token sequence is an
        exact prefix of the full sequence.  On a cross-boundary merge (BPE
        merges tokens across the prefix/candidate boundary), raises
        ``LLMScoringError`` rather than silently scoring a different token
        sequence.  Right-pads all sequences to the same length.
        """
        if len(prefixes) != len(candidates):
            raise LLMScoringError(
                f"prefixes ({len(prefixes)}) and candidates "
                f"({len(candidates)}) must have equal length"
            )

        if not prefixes:
            return BatchTokenization(
                input_ids=[],
                attention_mask=[],
                prefix_token_counts=[],
                candidate_token_ids=[],
            )

        async def _encode_pair(
            prefix: str, candidate: str
        ) -> tuple[list[int], int, list[int]]:
            full_ids = await self._tokenize(prefix + candidate)
            prefix_ids = await self._tokenize(prefix)

            if full_ids[: len(prefix_ids)] == prefix_ids:
                prefix_count = len(prefix_ids)
                candidate_ids = full_ids[prefix_count:]
            else:
                raise LLMScoringError(
                    f"tokenization boundary mismatch: the prefix token "
                    f"sequence is not an exact prefix of the full token "
                    f"sequence (prompt length {len(prefix + candidate)}); "
                    f"the server tokenizer merges tokens across the "
                    f"prefix/candidate boundary"
                )
            return full_ids, prefix_count, candidate_ids

        try:
            results = await asyncio.gather(
                *(
                    _encode_pair(prefix, candidate)
                    for prefix, candidate in zip(prefixes, candidates, strict=True)
                )
            )
        except LLMScoringError:
            await self._safe_close_session()
            raise

        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []
        all_prefix_counts: list[int] = []
        all_candidate_ids: list[list[int]] = []

        for full_ids, prefix_count, candidate_ids in results:
            all_input_ids.append(full_ids)
            all_attention_masks.append([1] * len(full_ids))
            all_prefix_counts.append(prefix_count)
            all_candidate_ids.append(candidate_ids)

        max_len = max(len(ids) for ids in all_input_ids)
        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            all_input_ids[i] = all_input_ids[i] + [_PAD_TOKEN_ID] * pad_len
            all_attention_masks[i] = all_attention_masks[i] + [0] * pad_len

        return BatchTokenization(
            input_ids=all_input_ids,
            attention_mask=all_attention_masks,
            prefix_token_counts=all_prefix_counts,
            candidate_token_ids=all_candidate_ids,
        )

    async def _safe_close_session(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def aclose(self) -> None:
        """Close the underlying aiohttp.ClientSession."""
        await self._safe_close_session()
