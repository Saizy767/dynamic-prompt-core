"""Async batch LLM client for OpenAI-compatible servers."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import tomllib
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

import aiohttp
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer

from dynamic_prompt_core.infrastructure.llm.adapters import (
    _ADAPTERS,
    _DEFAULT_MODE,
    _schema_for,
    _strip_chat_suffix,
)
from dynamic_prompt_core.infrastructure.llm.enums import Backend, StructuredMode
from dynamic_prompt_core.infrastructure.llm.errors import (
    ConfigError,
    HttpError,
    UnexpectedShapeError,
)
from dynamic_prompt_core.infrastructure.llm.response import (
    CallResult,
    ParseStatus,
    RawResponse,
    ResponseStatus,
)

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"


class _LogWriter:
    def __init__(self, log_path: str, run_id: str = "smoke") -> None:
        self._path = log_path
        self._run_id = run_id
        self._lock = asyncio.Lock()

    @property
    def run_id(self) -> str:
        return self._run_id

    async def write(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        async with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


class AsyncTask:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        endpoint: str = DEFAULT_ENDPOINT,
        system_prompt: str = "",
        completion_timeout: int = 60,
        truncate_tokens: int = 300,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_thinking: bool = False,
        backend: Backend | str | None = None,
        structured_mode: StructuredMode | str | None = None,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
        retry_backoff_max: float = 30.0,
        retry_jitter: float = 0.5,
        log_path: str | None = None,
        run_id: str = "smoke",
    ) -> None:
        config_path = os.path.abspath(config_path)

        self._endpoint = _strip_chat_suffix(endpoint)
        self._system_prompt = system_prompt
        self._completion_timeout = completion_timeout
        self._truncate_tokens = truncate_tokens
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._enable_thinking = enable_thinking
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._retry_backoff_max = retry_backoff_max
        self._retry_jitter = retry_jitter

        self._log_writer = _LogWriter(log_path, run_id) if log_path else None

        with open(config_path, "rb") as f:
            self._config = tomllib.load(f)

        if backend is None:
            cfg_backend = self._config.get("llm", {}).get("backend")
            if not cfg_backend:
                raise ConfigError(
                    "No backend specified: pass backend= explicitly or set "
                    "[llm].backend in config.toml"
                )
            backend = cfg_backend

        self._backend = Backend(backend) if isinstance(backend, str) else backend
        if structured_mode is None:
            structured_mode = _DEFAULT_MODE[self._backend]
        self._structured_mode = (
            StructuredMode(structured_mode)
            if isinstance(structured_mode, str)
            else structured_mode
        )
        self._adapter = _ADAPTERS[self._backend](self._structured_mode)
        self._chat_url = self._adapter.chat_url(self._endpoint)

        model_path = self._config["llm"]["model_path"]
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tokenizer from '{model_path}'. Point "
                "config.llm.model_path at a directory containing "
                "tokenizer.json / tokenizer_config.json."
            ) from exc

        self._served_model_name = self._config["llm"]["served_model_name"]

    # ------------------------------------------------------------------ #
    #  Auto-detection
    # ------------------------------------------------------------------ #
    @classmethod
    async def create(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        endpoint: str = DEFAULT_ENDPOINT,
        **kwargs: Any,
    ) -> AsyncTask:
        """Like __init__, but detects backend by querying GET /v1/models."""
        backend = await cls.detect_backend(endpoint)
        log.info("Detected backend: %s", backend.value)
        return cls(
            config_path=config_path, endpoint=endpoint,
            backend=backend, **kwargs,
        )

    @staticmethod
    async def detect_backend(endpoint: str, timeout: float = 5.0) -> Backend:
        url = _strip_chat_suffix(endpoint) + "/models"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status != 200:
                        raise ConfigError(
                            f"Cannot determine backend: GET /v1/models "
                            f"returned HTTP {resp.status}"
                        )
                    data = await resp.json()
        except aiohttp.ClientError as exc:
            raise ConfigError(f"Cannot determine backend: {exc}") from exc

        items = data.get("data") or data.get("models") or []
        if not items:
            raise ConfigError("Cannot determine backend: model list is empty")
        owner = (items[0].get("owned_by") or "").lower()
        if "llamacpp" in owner or "llama.cpp" in owner:
            return Backend.LLAMACPP
        if "vllm" in owner:
            return Backend.VLLM
        raise ConfigError(
            f"Cannot determine backend: unrecognized owned_by '{owner}'. "
            "Pass backend= explicitly."
        )

    # ------------------------------------------------------------------ #
    #  Truncation
    # ------------------------------------------------------------------ #
    def _truncate(self, text: str, max_tokens: int) -> tuple[str, bool]:
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text, False
        return (
            cast(str, self._tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)),
            True,
        )

    # ------------------------------------------------------------------ #
    #  Retry policy
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_retryable(exc: BaseException, temperature: float = 0.0) -> bool:
        if isinstance(exc, (aiohttp.ClientConnectionError,
                            aiohttp.ServerTimeoutError,
                            asyncio.TimeoutError)):
            return True
        if isinstance(exc, HttpError):
            # 5xx and 429 are transient; 4xx are almost always permanent.
            return exc.status >= 500 or exc.status == 429
        if isinstance(exc, ValidationError):
            # At temp=0 the model is deterministic; retrying is pointless.
            return temperature > 0.0
        return False

    def _backoff(self, attempt: int) -> float:
        delay = min(self._retry_backoff_base ** attempt, self._retry_backoff_max)
        return delay + random.uniform(0, self._retry_jitter)

    # ------------------------------------------------------------------ #
    #  Public API — transport
    # ------------------------------------------------------------------ #
    async def analyze_raw(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: type[T],
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        truncate_tokens: int | None = None,
        max_retries: int | None = None,
    ) -> RawResponse:
        truncate_limit = (
            truncate_tokens if truncate_tokens is not None else self._truncate_tokens
        )
        truncated_text, was_truncated = self._truncate(text, truncate_limit)

        max_attempts = max_retries if max_retries is not None else self._max_retries
        start = time.monotonic()
        last_exc: Exception | None = None
        attempt_count = 0

        for attempt in range(max_attempts):
            attempt_count = attempt + 1
            try:
                content, finish_reason, usage = await self._post_once(
                    session, truncated_text, model,
                    system_prompt=system_prompt, max_tokens=max_tokens,
                )
                latency_ms = (time.monotonic() - start) * 1000
                status = ResponseStatus.OK if content else ResponseStatus.EMPTY
                return RawResponse(
                    content=content,
                    model=self._served_model_name,
                    latency_ms=latency_ms,
                    status=status,
                    attempts=attempt_count,
                    finish_reason=finish_reason,
                    usage=usage,
                    truncated=was_truncated,
                )
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc, self._temperature):
                    break
                log.warning("Attempt %d/%d failed: %s",
                            attempt + 1, max_attempts, exc)
                if attempt + 1 < max_attempts:
                    await asyncio.sleep(self._backoff(attempt))

        latency_ms = (time.monotonic() - start) * 1000
        return self._error_response(
            last_exc, latency_ms, attempt_count, was_truncated
        )

    # ------------------------------------------------------------------ #
    #  Public API — parsing
    # ------------------------------------------------------------------ #
    async def _analyze_detailed(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: type[T],
        *,
        call_type: str = "analyze",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        truncate_tokens: int | None = None,
        max_retries: int | None = None,
        true_val: Any | None = None,
    ) -> CallResult:
        raw = await self.analyze_raw(
            session, text, model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
            max_retries=max_retries,
        )

        parse_status: ParseStatus | None = None
        result: T | None = None

        if raw.status == ResponseStatus.OK and raw.content:
            try:
                result = model.model_validate_json(raw.content)
                parse_status = ParseStatus.OK
            except ValidationError:
                if raw.finish_reason == "length":
                    parse_status = ParseStatus.TRUNCATED
                else:
                    try:
                        json.loads(raw.content)
                        parse_status = ParseStatus.SCHEMA_MISMATCH
                    except (json.JSONDecodeError, TypeError):
                        parse_status = ParseStatus.INVALID_JSON
                log.debug(
                    "Parse failed (%s): %r",
                    parse_status.value, raw.content[:500],
                )

        if self._log_writer:
            entry = self._build_log_entry(
                raw, text, call_type, parse_status,
                system_prompt, max_tokens, truncate_tokens,
                true_val=true_val,
            )
            await self._log_writer.write(entry)

        return CallResult(
            parsed=result,
            raw_content=raw.content,
            latency_ms=raw.latency_ms,
            parse_status=parse_status,
            status=raw.status,
            finish_reason=raw.finish_reason,
            error=raw.error,
            truncated=raw.truncated,
        )

    async def analyze(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: type[T],
        *,
        call_type: str = "analyze",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        truncate_tokens: int | None = None,
        max_retries: int | None = None,
    ) -> T | None:
        result = await self._analyze_detailed(
            session, text, model,
            call_type=call_type,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
            max_retries=max_retries,
        )
        return cast("T | None", result.parsed)

    # ------------------------------------------------------------------ #
    #  Public API — call-type wrappers
    # ------------------------------------------------------------------ #
    async def extract_theses(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: type[T],
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        truncate_tokens: int = 2000,
        **kwargs: Any,
    ) -> T | None:
        return await self.analyze(
            session, text, model,
            call_type="extract_theses",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
            **kwargs,
        )

    async def extract_theses_detailed(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: type[T],
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        truncate_tokens: int = 2000,
        true_val: Any | None = None,
        **kwargs: Any,
    ) -> CallResult:
        return await self._analyze_detailed(
            session, text, model,
            call_type="extract_theses",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
            true_val=true_val,
            **kwargs,
        )

    async def analyze_many(
        self,
        texts: list[str],
        model: type[T],
        concurrency: int = 50,
        show_progress: bool = False,
        *,
        call_type: str = "analyze",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        truncate_tokens: int | None = None,
    ) -> list[T | None]:
        semaphore = asyncio.Semaphore(concurrency)
        results: list[T | None] = [None] * len(texts)

        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError as exc:
                raise RuntimeError(
                    "Install 'tqdm' to use show_progress=True"
                ) from exc
            progress_bar = tqdm(total=len(texts), desc="Analyze")

        connector = aiohttp.TCPConnector(limit=concurrency)
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async def process_one(index: int, text: str) -> None:
                    async with semaphore:
                        try:
                            results[index] = await self.analyze(
                                session, text, model,
                                call_type=call_type,
                                system_prompt=system_prompt,
                                max_tokens=max_tokens,
                                truncate_tokens=truncate_tokens,
                            )
                        except Exception as exc:
                            log.exception("Task %d crashed: %s", index, exc)
                        finally:
                            if progress_bar:
                                progress_bar.update(1)

                await asyncio.gather(
                    *(process_one(i, t) for i, t in enumerate(texts))
                )
        finally:
            if progress_bar:
                progress_bar.close()

        return results

    async def extract_theses_many(
        self,
        texts: list[str],
        model: type[T],
        concurrency: int = 50,
        show_progress: bool = False,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        truncate_tokens: int = 2000,
    ) -> list[T | None]:
        return await self.analyze_many(
            texts, model, concurrency, show_progress,
            call_type="extract_theses",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
        )

    async def _analyze_many_detailed(
        self,
        texts: list[str],
        model: type[T],
        concurrency: int = 50,
        show_progress: bool = False,
        *,
        call_type: str = "analyze",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        truncate_tokens: int | None = None,
    ) -> list[CallResult]:
        semaphore = asyncio.Semaphore(concurrency)
        results: list[CallResult | None] = [None] * len(texts)

        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError as exc:
                raise RuntimeError(
                    "Install 'tqdm' to use show_progress=True"
                ) from exc
            progress_bar = tqdm(total=len(texts), desc=call_type)

        connector = aiohttp.TCPConnector(limit=concurrency)
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async def process_one(index: int, text: str) -> None:
                    async with semaphore:
                        try:
                            results[index] = await self._analyze_detailed(
                                session, text, model,
                                call_type=call_type,
                                system_prompt=system_prompt,
                                max_tokens=max_tokens,
                                truncate_tokens=truncate_tokens,
                            )
                        except Exception as exc:
                            log.exception("Task %d crashed: %s", index, exc)
                            results[index] = CallResult(
                                parsed=None,
                                raw_content=None,
                                latency_ms=0.0,
                                parse_status=None,
                                status=ResponseStatus.UNEXPECTED_SHAPE,
                                error=str(exc),
                            )
                        finally:
                            if progress_bar:
                                progress_bar.update(1)

                await asyncio.gather(
                    *(process_one(i, t) for i, t in enumerate(texts))
                )
        finally:
            if progress_bar:
                progress_bar.close()

        return cast("list[CallResult]", results)

    async def extract_theses_many_detailed(
        self,
        texts: list[str],
        model: type[T],
        concurrency: int = 50,
        show_progress: bool = False,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        truncate_tokens: int = 2000,
    ) -> list[CallResult]:
        return await self._analyze_many_detailed(
            texts, model, concurrency, show_progress,
            call_type="extract_theses",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
        )

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #
    async def _post_once(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: type[T],
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        schema = _schema_for(model)
        payload = self._adapter.build_payload(
            model=self._served_model_name,
            system_prompt=system_prompt if system_prompt is not None else self._system_prompt,
            user_text=text,
            json_schema_obj=schema,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            enable_thinking=self._enable_thinking,
        )

        async with session.post(
            self._chat_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self._completion_timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise HttpError(resp.status, str(resp.url), body)
            data = await resp.json()

        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise UnexpectedShapeError(f"Unexpected response shape: {data!r}") from exc

        usage = data.get("usage")
        return content, finish_reason, usage

    def _error_response(
        self,
        exc: Exception | None,
        latency_ms: float,
        attempts: int,
        truncated: bool,
    ) -> RawResponse:
        if isinstance(exc, HttpError):
            status = ResponseStatus.HTTP_ERROR
            http_status = exc.status
        elif isinstance(exc, (aiohttp.ServerTimeoutError, asyncio.TimeoutError)):
            status = ResponseStatus.TIMEOUT
            http_status = None
        elif isinstance(exc, aiohttp.ClientConnectionError):
            status = ResponseStatus.NETWORK_ERROR
            http_status = None
        else:
            status = ResponseStatus.UNEXPECTED_SHAPE
            http_status = None
        return RawResponse(
            content=None,
            model=self._served_model_name,
            latency_ms=latency_ms,
            status=status,
            attempts=attempts,
            http_status=http_status,
            error=str(exc) if exc else None,
            truncated=truncated,
        )

    def _build_log_entry(
        self,
        raw: RawResponse,
        text: str,
        call_type: str,
        parse_status: ParseStatus | None,
        system_prompt: str | None,
        max_tokens: int | None,
        truncate_tokens: int | None,
        true_val: Any | None = None,
    ) -> dict[str, Any]:
        used_sp = system_prompt if system_prompt is not None else self._system_prompt
        used_mt = max_tokens if max_tokens is not None else self._max_tokens
        used_tt = (
            truncate_tokens if truncate_tokens is not None else self._truncate_tokens
        )
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self._log_writer.run_id if self._log_writer else "smoke",
            "call_type": call_type,
            "system_prompt_hash": hashlib.sha256(
                used_sp.encode("utf-8")
            ).hexdigest()[:16],
            "text_hash": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()[:16],
            "model_name": raw.model,
            "params": {
                "temperature": self._temperature,
                "top_p": self._top_p,
                "max_tokens": used_mt,
                "truncate_tokens": used_tt,
            },
            "raw_content": raw.content,
            "latency_ms": raw.latency_ms,
            "finish_reason": raw.finish_reason,
            "usage": raw.usage,
            "status": raw.status.value,
            "parse_status": parse_status.value if parse_status else None,
            "attempts": raw.attempts,
            "http_status": raw.http_status,
            "error": raw.error,
            "truncated": raw.truncated,
            "true_val": true_val,
        }
