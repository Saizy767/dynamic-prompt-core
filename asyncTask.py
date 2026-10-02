"""
Async batch client for OpenAI-compatible LLM servers.

Works with vLLM, llama.cpp (llama-server) and any generic OpenAI-compatible
backend. Structured output is produced via JSON Schema (native on vLLM) or
JSON-object mode with schema injected into the system prompt (llama.cpp).

Quick start
-----------
    # vLLM
    task = AsyncTask(backend="vllm", endpoint="http://127.0.0.1:8080/v1")

    # llama.cpp
    task = AsyncTask(backend="llamacpp", endpoint="http://127.0.0.1:8080/v1")

    # backend from config.toml [llm].backend
    task = AsyncTask(endpoint="http://127.0.0.1:8080/v1")

    # auto-detect from GET /v1/models
    task = await AsyncTask.create(endpoint="http://127.0.0.1:8080/v1")

    results = await task.classify_many(texts, MyModel, concurrency=32,
                                       show_progress=True)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, List, Optional, Type, TypeVar

import aiohttp
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"


# --------------------------------------------------------------------------- #
#  Backend enums & adapter protocol
# --------------------------------------------------------------------------- #
class Backend(str, Enum):
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OPENAI = "openai"


class StructuredMode(str, Enum):
    JSON_SCHEMA = "json_schema"   # native constrained decoding
    JSON_OBJECT = "json_object"   # valid JSON only; schema injected into prompt
    NONE = "none"                 # no constraints at all


class ResponseStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    UNEXPECTED_SHAPE = "unexpected_shape"


class ParseStatus(str, Enum):
    OK = "ok"
    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    TRUNCATED = "truncated"


@dataclass
class RawResponse:
    content: Optional[str]
    model: str
    latency_ms: float
    status: ResponseStatus
    attempts: int
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    http_status: Optional[int] = None
    error: Optional[str] = None
    truncated: bool = False


def _strip_chat_suffix(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    suffix = "/chat/completions"
    if endpoint.endswith(suffix):
        endpoint = endpoint[: -len(suffix)]
    return endpoint


class _BaseAdapter:
    structured_mode: StructuredMode

    def chat_url(self, endpoint: str) -> str:
        return _strip_chat_suffix(endpoint) + "/chat/completions"

    def build_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        json_schema_obj: dict,
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ) -> dict:
        raise NotImplementedError

    @staticmethod
    def _schema_into_prompt(system_prompt: str, schema: dict) -> str:
        return (
            f"{system_prompt}\n\n"
            "Reply with a single JSON object that strictly matches this JSON "
            "Schema (no markdown fences, no commentary):\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )


class _VLLMAdapter(_BaseAdapter):
    def __init__(self, mode: StructuredMode) -> None:
        self.structured_mode = mode

    def build_payload(
        self, *, model, system_prompt, user_text, json_schema_obj,
        max_tokens, temperature, top_p, enable_thinking,
    ) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if self.structured_mode is StructuredMode.JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema_obj,
            }
        elif self.structured_mode is StructuredMode.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
            messages[0]["content"] = self._schema_into_prompt(
                system_prompt, json_schema_obj["schema"]
            )
        return payload


class _OpenAIAdapter(_VLLMAdapter):
    """Generic OpenAI-compatible: like vLLM, but no chat_template_kwargs."""
    def build_payload(self, **kw) -> dict:
        payload = super().build_payload(**kw)
        payload.pop("chat_template_kwargs", None)
        return payload


class _LlamaCppAdapter(_BaseAdapter):
    def __init__(self, mode: StructuredMode) -> None:
        self.structured_mode = mode

    def build_payload(
        self, *, model, system_prompt, user_text, json_schema_obj,
        max_tokens, temperature, top_p, enable_thinking,
    ) -> dict:
        # llama-server does not understand chat_template_kwargs and, on older
        # builds, not json_schema either. Default to json_object + prompt.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.structured_mode is StructuredMode.JSON_SCHEMA:
            # Supported by recent llama.cpp builds; older ones return 400.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema_obj,
            }
        elif self.structured_mode is StructuredMode.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
            messages[0]["content"] = self._schema_into_prompt(
                system_prompt, json_schema_obj["schema"]
            )
        return payload


_ADAPTERS = {
    Backend.VLLM: _VLLMAdapter,
    Backend.OPENAI: _OpenAIAdapter,
    Backend.LLAMACPP: _LlamaCppAdapter,
}

_DEFAULT_MODE = {
    Backend.VLLM: StructuredMode.JSON_SCHEMA,
    Backend.OPENAI: StructuredMode.JSON_SCHEMA,
    Backend.LLAMACPP: StructuredMode.JSON_OBJECT,
}


# --------------------------------------------------------------------------- #
#  Errors & helpers
# --------------------------------------------------------------------------- #
class ConfigError(ValueError):
    """Raised when config is missing, invalid, or backend cannot be determined."""


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} on {url}: {body[:500]}")
        self.status = status
        self.url = url
        self.body = body


@lru_cache(maxsize=None)
def _schema_for(model_cls: Type[BaseModel]) -> dict:
    return {
        "name": "request_analysis",
        "description": "Analysis of requests",
        "schema": model_cls.model_json_schema(),
        "strict": True,
    }


# --------------------------------------------------------------------------- #
#  Log writer
# --------------------------------------------------------------------------- #
class _LogWriter:
    def __init__(self, log_path: str, run_id: str = "smoke") -> None:
        self._path = log_path
        self._run_id = run_id
        self._lock = asyncio.Lock()

    @property
    def run_id(self) -> str:
        return self._run_id

    async def write(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        async with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# --------------------------------------------------------------------------- #
#  AsyncTask
# --------------------------------------------------------------------------- #
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
        log_path: Optional[str] = None,
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
        **kwargs,
    ) -> "AsyncTask":
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
            self._tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True),
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
        model: Type[T],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        truncate_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> RawResponse:
        truncate_limit = (
            truncate_tokens if truncate_tokens is not None else self._truncate_tokens
        )
        truncated_text, was_truncated = self._truncate(text, truncate_limit)

        max_attempts = max_retries if max_retries is not None else self._max_retries
        start = time.monotonic()
        last_exc: Optional[Exception] = None
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
    async def analyze(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: Type[T],
        *,
        call_type: str = "analyze",
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        truncate_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[T]:
        raw = await self.analyze_raw(
            session, text, model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
            max_retries=max_retries,
        )

        parse_status: Optional[ParseStatus] = None
        result: Optional[T] = None

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
            )
            await self._log_writer.write(entry)

        return result

    # ------------------------------------------------------------------ #
    #  Public API — call-type wrappers
    # ------------------------------------------------------------------ #
    async def classify(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: Type[T],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 128,
        truncate_tokens: int = 300,
        **kwargs,
    ) -> Optional[T]:
        return await self.analyze(
            session, text, model,
            call_type="classify",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
            **kwargs,
        )

    async def extract_theses(
        self,
        session: aiohttp.ClientSession,
        text: str,
        model: Type[T],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        truncate_tokens: int = 2000,
        **kwargs,
    ) -> Optional[T]:
        return await self.analyze(
            session, text, model,
            call_type="extract_theses",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
            **kwargs,
        )

    async def analyze_many(
        self,
        texts: List[str],
        model: Type[T],
        concurrency: int = 50,
        show_progress: bool = False,
        *,
        call_type: str = "analyze",
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        truncate_tokens: Optional[int] = None,
    ) -> List[Optional[T]]:
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[T]] = [None] * len(texts)

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

    async def classify_many(
        self,
        texts: List[str],
        model: Type[T],
        concurrency: int = 50,
        show_progress: bool = False,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 128,
        truncate_tokens: int = 300,
    ) -> List[Optional[T]]:
        return await self.analyze_many(
            texts, model, concurrency, show_progress,
            call_type="classify",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            truncate_tokens=truncate_tokens,
        )

    async def extract_theses_many(
        self,
        texts: List[str],
        model: Type[T],
        concurrency: int = 50,
        show_progress: bool = False,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        truncate_tokens: int = 2000,
    ) -> List[Optional[T]]:
        return await self.analyze_many(
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
        model: Type[T],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[dict]]:
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
            raise RuntimeError(f"Unexpected response shape: {data!r}") from exc

        usage = data.get("usage")
        return content, finish_reason, usage

    def _error_response(
        self,
        exc: Optional[Exception],
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
        parse_status: Optional[ParseStatus],
        system_prompt: Optional[str],
        max_tokens: Optional[int],
        truncate_tokens: Optional[int],
    ) -> dict:
        used_sp = system_prompt if system_prompt is not None else self._system_prompt
        used_mt = max_tokens if max_tokens is not None else self._max_tokens
        used_tt = (
            truncate_tokens if truncate_tokens is not None else self._truncate_tokens
        )
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
        }
