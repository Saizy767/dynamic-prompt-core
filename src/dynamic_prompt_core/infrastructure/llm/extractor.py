"""Standalone key-phrase extraction component.

Wraps the existing extraction logic (AsyncTask.extract_theses_detailed,
repair, status mapping, normalization) with a dedicated system prompt
loaded from a resource file.  Does not receive classification results.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TypeVar, cast

import aiohttp
from pydantic import BaseModel

from dynamic_prompt_core.application.schemas import ThesisExtraction
from dynamic_prompt_core.infrastructure.llm import (
    AsyncTask,
    CallResult,
    ResponseStatus,
)
from dynamic_prompt_core.infrastructure.llm.prompts import load_prompt
from dynamic_prompt_core.infrastructure.nlp import normalize_theses

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

_STATUS_OK = "ok"
_STATUS_REPAIRED = "repaired"
_STATUS_FAILED = "failed"
_STATUS_SKIPPED = "skipped"

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _try_repair(raw_content: str | None, model_cls: type[_ModelT]) -> _ModelT | None:
    if not raw_content:
        return None
    stripped = raw_content.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        candidate = m.group(1).strip()
        try:
            return model_cls.model_validate_json(candidate)
        except Exception:
            pass
    return None


def _map_call_status(
    call_result: CallResult, model_cls: type[_ModelT]
) -> tuple[str, _ModelT | None]:
    if call_result.parsed is not None:
        return _STATUS_OK, cast(_ModelT, call_result.parsed)
    repaired = _try_repair(call_result.raw_content, model_cls)
    if repaired is not None:
        return _STATUS_REPAIRED, repaired
    return _STATUS_FAILED, None


@dataclass
class ExtractionResult:
    theses_raw: list[str]
    theses_norm: list[str]
    status: str
    latency_ms: float
    raw_extract: str | None
    call_result: CallResult


class Extractor:
    """Key-phrase extraction component with a dedicated system prompt."""

    def __init__(self, task: AsyncTask, language: str = "ru") -> None:
        self._task = task
        self._language = language
        self._system_prompt = load_prompt("extractor")

    async def extract(
        self,
        session: aiohttp.ClientSession,
        text: str,
        true_val: int | None = None,
    ) -> ExtractionResult:
        """Extract key phrases from ``text``.

        Returns an ``ExtractionResult`` with theses, status, latency, and raw
        response.  Does not receive or use classification results.
        """
        try:
            call_result = await self._task.extract_theses_detailed(
                session,
                text,
                ThesisExtraction,
                system_prompt=self._system_prompt,
                true_val=true_val,
            )
        except Exception as exc:
            log.exception("extract_theses crashed: %s", exc)
            call_result = CallResult(
                parsed=None,
                raw_content=None,
                latency_ms=0.0,
                parse_status=None,
                status=ResponseStatus.UNEXPECTED_SHAPE,
                error=str(exc),
            )

        status, parsed = _map_call_status(call_result, ThesisExtraction)
        theses_raw = list(parsed.theses) if parsed else []
        theses_norm = normalize_theses(theses_raw, self._language) if theses_raw else []

        return ExtractionResult(
            theses_raw=theses_raw,
            theses_norm=theses_norm,
            status=status,
            latency_ms=call_result.latency_ms,
            raw_extract=call_result.raw_content,
            call_result=call_result,
        )

    @staticmethod
    def skipped() -> ExtractionResult:
        """Return a skipped extraction result."""
        return ExtractionResult(
            theses_raw=[],
            theses_norm=[],
            status=_STATUS_SKIPPED,
            latency_ms=0.0,
            raw_extract=None,
            call_result=CallResult(
                parsed=None,
                raw_content=None,
                latency_ms=0.0,
                parse_status=None,
                status=ResponseStatus.SKIPPED,
            ),
        )
