"""HTTP-based model adapter for candidate scoring via llama-server.

``LlamaServerCausalLMAdapter`` implements the ``BatchedCausalLanguageModel`` seam
by calling the ``llama-server`` OpenAI-compatible API.  Log probabilities for
all input tokens are retrieved via ``POST /v1/completions`` with ``echo: true``
and ``logprobs: 1``, which returns ``token_logprobs`` — the log probability of
each token given its preceding context.

Since ``LogitScorer`` applies ``log_softmax`` to the logits, we reconstruct a
logit vector per position that yields the correct log probability.  In a causal
LM, logits at position ``i`` predict the token at position ``i + 1``, so we use
``token_logprobs[i + 1]`` and ``token_ids[i + 1]``::

    logits[i][actual_token] = log_prob[i + 1]
    logits[i][other]        = log((1 - exp(log_prob[i + 1])) / (V - 1))

This guarantees ``log_softmax(logits[i])[token_ids[i + 1]] == log_prob[i + 1]``.
The ``other`` logit is computed per-position from the actual log-prob, NOT
set to a constant floor: a constant floor makes every token's ``log_softmax``
≈ 0, collapsing all candidates to equal probability (50/50).
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

    In a causal LM, logits at position ``i`` predict the token at position
    ``i + 1``.  ``token_logprobs[i + 1]`` is the log-probability of
    ``token_ids[i + 1]`` given ``token_ids[0 : i + 1]``.  For each position
    ``i``, the logit vector is set so that::

        log_softmax(logits[i])[token_ids[i + 1]] == token_logprobs[i + 1]

    via::

        logits[i][actual_token] = lp
        logits[i][other]        = log((1 - exp(lp)) / (V - 1))

    This guarantees the real model log-probability is preserved through the
    LogitScorer's ``log_softmax`` extraction with causal alignment.

    ``token_logprobs[0]`` is typically ``None`` (no preceding context); the
    last position has no successor token to predict and gets a floor logit
    vector.  Neither is used for candidate scoring.

    If ``token_logprobs[next_idx]`` is ``None`` at any position that has a
    valid successor token (``next_idx < len``), this is an API error: the
    server did not return a log-probability for a position that should have
    one.  In that case ``LLMScoringError`` is raised rather than silently
    filling the logit vector with a constant floor, which would collapse all
    candidates to equal probability.
    """
    seq_len = len(token_ids)
    result: list[list[float]] = []
    missing_count = 0

    for i in range(seq_len):
        logit = [_LOGIT_FLOOR] * vocab_size

        next_idx = i + 1
        if next_idx < len(token_logprobs) and next_idx < len(token_ids):
            raw_lp = token_logprobs[next_idx]
            if raw_lp is None:
                missing_count += 1
                raise LLMScoringError(
                    f"token_logprobs[{next_idx}] is None at position {i} "
                    f"that has a successor token (token_ids[{next_idx}]="
                    f"{token_ids[next_idx]}); the API did not return a "
                    f"valid log-probability for a scoring position"
                )
            lp = float(raw_lp)
            if math.isnan(lp) or math.isinf(lp):
                raise LLMScoringError(
                    f"token_logprobs[{next_idx}] is NaN or infinite: {lp}"
                )
            next_token = token_ids[next_idx]
            if next_token >= vocab_size:
                vocab_size = next_token + 1
                logit.extend([_LOGIT_FLOOR] * (next_token + 1 - len(logit)))
            if 0 <= next_token < len(logit):
                effective_vocab = len(logit)
                if effective_vocab > 1:
                    prob = math.exp(lp) if lp < 0.0 else 1.0
                    remaining = 1.0 - prob
                    if remaining > 0.0:
                        other_logit = (
                            math.log(remaining)
                            - math.log(effective_vocab - 1)
                        )
                    else:
                        other_logit = float("-inf")
                    logit = [other_logit] * effective_vocab
                logit[next_token] = lp

        result.append(logit)

    if missing_count > 0:
        log.warning(
            "_build_logits_from_logprobs: %d missing token_logprobs out of %d positions",
            missing_count,
            seq_len,
        )

    return result


def _parse_v1_logprobs(
    body: dict[str, Any],
    token_ids: list[int],
) -> list[list[float]]:
    """Parse the /v1/completions response and build logits from token_logprobs.

    Verifies that the token IDs returned by the API match the original
    ``token_ids``.  If the detokenize→tokenize round-trip changed the
    tokenization (BPE merges/splits), the ``token_logprobs`` would correspond
    to different tokens than ``token_ids``, producing incorrect scores.
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

    prompt_logprobs = logprobs_data.get("prompt_logprobs")
    if isinstance(prompt_logprobs, list) and len(prompt_logprobs) == len(token_ids):
        log.debug(
            "v1/completions: using prompt_logprobs (%d entries)",
            len(prompt_logprobs),
        )
        vocab_size = _DEFAULT_VOCAB_SIZE
        return _build_logits_from_logprobs(prompt_logprobs, token_ids, vocab_size)

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

    if len(token_logprobs) < len(token_ids):
        raise LLMScoringError(
            f"v1/completions returned {len(token_logprobs)} token_logprobs "
            f"but {len(token_ids)} were needed; the server does not support "
            f"echo logprobs or prompt_logprobs — consider upgrading "
            f"llama-server or using a different backend"
        )

    api_token_ids = _extract_api_token_ids(logprobs_data)
    if (
        api_token_ids is not None
        and len(api_token_ids) == len(token_ids)
        and api_token_ids != token_ids
    ):
        raise LLMScoringError(
            f"token IDs from API response do not match the original token IDs; "
            f"this indicates a detokenize->tokenize round-trip mismatch "
            f"(BPE merge/split). "
            f"api={api_token_ids[:10]}... orig={token_ids[:10]}... "
            f"(api_len={len(api_token_ids)}, orig_len={len(token_ids)})"
        )

    vocab_size = _DEFAULT_VOCAB_SIZE
    return _build_logits_from_logprobs(token_logprobs, token_ids, vocab_size)


def _extract_api_token_ids(logprobs_data: dict[str, Any]) -> list[int] | None:
    """Extract token IDs from the API logprobs response, if available.

    The ``content`` format includes ``id`` fields per token.  The
    ``token_logprobs`` format does not include IDs, so ``None`` is returned
    and the check is skipped.
    """
    content = logprobs_data.get("content")
    if isinstance(content, list):
        ids = [
            entry.get("id")
            for entry in content
            if isinstance(entry, dict) and entry.get("id") is not None
        ]
        if ids and len(ids) == len(content):
            return ids

    tokens = logprobs_data.get("tokens")
    if isinstance(tokens, list) and "token_ids" in logprobs_data:
        api_ids = logprobs_data.get("token_ids")
        if isinstance(api_ids, list):
            return [int(t) for t in api_ids]

    return None


def _parse_native_logprobs(
    body: dict[str, Any],
    token_ids: list[int],
) -> list[list[float]]:
    """Parse the native /completion response with prompt_logprobs.

    The native endpoint returns:
    - ``prompt_tokens``: the token IDs the server used (for verification)
    - ``prompt_logprobs``: log-probability of each prompt token given its
      preceding context (``prompt_logprobs[0]`` is typically ``None``)

    If ``prompt_tokens`` is present and differs from ``token_ids``, the
    server re-tokenized the input — raise ``LLMScoringError``.
    """
    prompt_logprobs = body.get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list):
        raise LLMScoringError(
            f"native /completion response missing prompt_logprobs; "
            f"excerpt: {str(body)[:200]}"
        )

    prompt_tokens = body.get("prompt_tokens")
    if isinstance(prompt_tokens, list) and prompt_tokens != token_ids:
        raise LLMScoringError(
            f"prompt_tokens from native /completion do not match the "
            f"original token IDs; "
            f"api={prompt_tokens[:10]}... orig={token_ids[:10]}... "
            f"(api_len={len(prompt_tokens)}, orig_len={len(token_ids)})"
        )

    if len(prompt_logprobs) != len(token_ids):
        log.warning(
            "prompt_logprobs length (%d) != token_ids length (%d)",
            len(prompt_logprobs),
            len(token_ids),
        )

    vocab_size = _DEFAULT_VOCAB_SIZE
    return _build_logits_from_logprobs(prompt_logprobs, token_ids, vocab_size)


class LlamaServerCausalLMAdapter:
    """Concrete ``BatchedCausalLanguageModel`` backed by llama-server HTTP API.

    Uses the native ``/completion`` endpoint with ``prompt_logprobs: 1`` and
    ``n_predict: 0`` to get log-probabilities for all input token positions.
    The log probabilities are converted to logits that produce the correct
    scores when ``LogitScorer`` applies ``log_softmax``.

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

        Tries two endpoints in order:

        1. Native ``/completion`` with ``prompt_logprobs: 1`` and
           ``n_predict: 0`` — returns ``prompt_logprobs`` for every input
           position.  Supported by newer llama-server builds.

        2. OpenAI-compatible ``/v1/completions`` with ``echo: true`` and
           ``logprobs: 1`` — some versions include ``prompt_logprobs`` in
           the logprobs data even though ``token_logprobs`` only covers
           generated tokens.

        Token IDs are passed directly as the ``prompt`` parameter (a list
        of integers) to avoid a detokenize->tokenize round-trip.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._try_native(token_ids)
            except LLMScoringError as exc:
                if "missing prompt_logprobs" in str(exc):
                    log.debug(
                        "native /completion does not support prompt_logprobs; "
                        "falling back to /v1/completions"
                    )
                    return await self._try_v1(token_ids)
                last_exc = exc
                if isinstance(
                    exc.__cause__,
                    (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectorError),
                ):
                    log.warning(
                        "llama-server disconnected (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, _RETRY_DELAY_S,
                    )
                    await self._safe_close_session()
                    await asyncio.sleep(_RETRY_DELAY_S)
                    continue
                raise
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

    async def _try_native(self, token_ids: list[int]) -> SequenceLogits:
        """Try the native /completion endpoint with prompt_logprobs."""
        payload: dict[str, Any] = {
            "prompt": token_ids,
            "n_predict": 0,
            "prompt_logprobs": 1,
        }
        if self._served_model_name is not None:
            payload["model"] = self._served_model_name
        return await self._do_request(
            token_ids, payload, f"{self._endpoint}/completion", _parse_native_logprobs,
        )

    async def _try_v1(self, token_ids: list[int]) -> SequenceLogits:
        """Fall back to /v1/completions with echo and logprobs."""
        payload: dict[str, Any] = {
            "prompt": token_ids,
            "max_tokens": 1,
            "logprobs": 1,
            "echo": True,
            "temperature": 0,
        }
        if self._served_model_name is not None:
            payload["model"] = self._served_model_name
        return await self._do_request(
            token_ids, payload, self._v1_completions_url, _parse_v1_logprobs,
        )

    async def _do_request(
        self,
        token_ids: list[int],
        payload: dict[str, Any],
        url: str,
        parser: Any,
    ) -> SequenceLogits:
        """Execute a single HTTP request and parse the response."""
        session = self._get_session()
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    excerpt = body_text[:300]
                    raise LLMScoringError(
                        f"llama-server returned HTTP {resp.status} "
                        f"from {url}: {excerpt}"
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

        per_position = parser(body, token_ids)

        if self._vocab_size == 0 and per_position:
            self._vocab_size = len(per_position[0])

        return SequenceLogits(logits=tuple(tuple(p) for p in per_position))

    async def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        prefix_token_counts: list[int] | None = None,
        candidate_token_ids: list[list[int]] | None = None,
    ) -> BatchedLogits:
        """Evaluate each batch item and return per-item logits.

        When ``prefix_token_counts`` and ``candidate_token_ids`` are provided,
        uses per-position API calls to get log-probabilities for candidate
        token positions only.  This works with llama-server versions that
        do not support ``prompt_logprobs`` or echo logprobs.
        """
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

        use_per_position = (
            prefix_token_counts is not None
            and candidate_token_ids is not None
            and len(prefix_token_counts) == len(unpadded_items)
        )

        async def _gated_forward(tokens: list[int]) -> SequenceLogits:
            async with self._semaphore:
                return await self._forward_single(tokens)

        async def _gated_forward_per_position(
            tokens: list[int], prefix_count: int, cand_ids: list[int],
        ) -> SequenceLogits:
            async with self._semaphore:
                return await self._forward_single_per_position(
                    tokens, prefix_count, cand_ids,
                )

        try:
            if use_per_position:
                results = await asyncio.gather(
                    *(
                        _gated_forward_per_position(tokens, pc, cids)
                        for tokens, pc, cids in zip(
                            unpadded_items,
                            prefix_token_counts,
                            candidate_token_ids,
                            strict=True,
                        )
                    )
                )
            else:
                results = await asyncio.gather(
                    *(_gated_forward(tokens) for tokens in unpadded_items)
                )
        except LLMScoringError:
            await self._safe_close_session()
            raise

        return BatchedLogits(items=tuple(results))

    async def _forward_single_per_position(
        self,
        token_ids: list[int],
        prefix_count: int,
        candidate_ids: list[int],
    ) -> SequenceLogits:
        """Get logits via per-position API calls for candidate token positions.

        For each candidate token position j, sends
        ``token_ids[0 : prefix_count + j]`` as the prompt to ``/v1/completions``
        with ``max_tokens: 1`` and ``logprobs: 1``.  The response contains
        the log-probability of the generated (most likely) token at the next
        position.  This logprob is used as an approximation of the candidate
        token's logprob.

        Non-candidate positions get a floor logprob (-100.0) since they are
        not used for scoring.
        """
        seq_len = len(token_ids)
        token_logprobs: list[float | None] = [None] * seq_len

        for j in range(len(candidate_ids)):
            pos = prefix_count + j
            context = token_ids[:pos]

            payload: dict[str, Any] = {
                "prompt": context,
                "max_tokens": 1,
                "logprobs": 1,
                "temperature": 0,
            }
            if self._served_model_name is not None:
                payload["model"] = self._served_model_name

            url = self._v1_completions_url
            session = self._get_session()
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body_text = await resp.text()
                        raise LLMScoringError(
                            f"llama-server returned HTTP {resp.status} "
                            f"from {url}: {body_text[:300]}"
                        )
                    body = await resp.json()
            except LLMScoringError:
                raise
            except Exception as exc:
                raise LLMScoringError(
                    f"per-position request failed at pos={pos}: {exc}"
                ) from exc

            choices = body.get("choices", [])
            if not choices:
                raise LLMScoringError(
                    f"per-position response missing choices at pos={pos}"
                )
            logprobs_data = choices[0].get("logprobs", {})
            tlp = logprobs_data.get("token_logprobs", [])
            if not tlp:
                content = logprobs_data.get("content", [])
                if content and isinstance(content[0], dict):
                    lp = content[0].get("logprob")
                else:
                    lp = None
            else:
                lp = tlp[0]

            if lp is None:
                log.warning(
                    "per-position: None logprob at pos=%d for candidate token %d",
                    pos, candidate_ids[j],
                )
                token_logprobs[pos] = -100.0
            else:
                token_logprobs[pos] = float(lp)

        for i in range(1, seq_len):
            if token_logprobs[i] is None:
                token_logprobs[i] = -100.0

        vocab_size = self._vocab_size or _DEFAULT_VOCAB_SIZE
        per_position = _build_logits_from_logprobs(
            token_logprobs, token_ids, vocab_size,
        )

        if self._vocab_size == 0 and per_position:
            self._vocab_size = len(per_position[0])

        return SequenceLogits(logits=tuple(tuple(p) for p in per_position))

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
