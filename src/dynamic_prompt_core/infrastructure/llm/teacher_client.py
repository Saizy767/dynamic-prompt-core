"""Infrastructure adapter: teacher-model LLM client via AsyncTask.

Implements the ``TeacherLLMClient`` port by wrapping ``AsyncTask`` pointed at
the teacher endpoint. Contains no refinement business logic — only transport,
prompt assembly, and response parsing.
"""
from __future__ import annotations

import json
from typing import Any

import aiohttp
from pydantic import BaseModel

from dynamic_prompt_core.domain.services.thesis_refinement import RefinementReview
from dynamic_prompt_core.infrastructure.llm import (
    AsyncTask,
    RawResponse,
    ResponseStatus,
)


class TeacherClientError(ValueError):
    """Raised when the teacher client fails to review theses."""


class _RefinementPlanStub(BaseModel):
    """Stub model so AsyncTask.analyze_raw accepts a pydantic type.

    The actual parsing is done via raw JSON in ``_parse_refinement_json``
    to avoid importing ``application.schemas`` from infrastructure.
    """
    keep: list[str] = []
    reformulate: list[dict[str, str]] = []
    drop: list[dict[str, str]] = []
    add: list[str] = []


def build_teacher_prompt(
    *,
    filter_noisy: bool,
    filter_interpretive: bool,
    allow_additions: bool,
) -> str:
    """Build the system prompt for the teacher model.

    Conditionally includes noisy-filter, interpretive-filter, and addition
    instructions based on the config flags.
    """
    parts: list[str] = [
        "You are a thesis refinement assistant.",
        "You receive a text and a list of candidate theses extracted by a smaller model.",
        "Return a JSON object with four arrays:",
        '- "keep": theses to preserve unchanged (extractive, informative).',
        '- "reformulate": objects {"from": <original>, "to": <new>} for theses'
        " that need a more stable extractive phrasing. The reformulated text"
        " must preserve meaning and not exceed the original length.",
        '- "drop": objects {"thesis": <text>, "reason": <why>} for theses to exclude.',
    ]
    if filter_noisy:
        parts.append(
            "Drop noisy theses: generic phrases applicable to any text that"
            " carry no signal for classification."
        )
    if filter_interpretive:
        parts.append(
            "Drop or reformulate interpretive theses (conclusions about the"
            " text) into extractive form (features present in the text)."
        )
    if allow_additions:
        parts.append(
            '- "add": missed theses present in the text and relevant for'
            " classification, as plain strings."
        )
    else:
        parts.append('Do not add any new theses: "add" must be an empty array.')
    parts.append(
        "Every input thesis must appear in exactly one of keep, reformulate, or"
        " drop. Return only the JSON object."
    )
    return "\n".join(parts)


def _parse_refinement_json(content: str) -> dict[str, Any]:
    """Parse and validate the teacher JSON response into a plan dict."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TeacherClientError(f"Teacher response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise TeacherClientError("Teacher response is not a JSON object")

    for key in ("keep", "reformulate", "drop", "add"):
        if key in data and not isinstance(data[key], list):
            raise TeacherClientError(f'"{key}" must be an array')

    return data


class TeacherClient:
    """Implements ``TeacherLLMClient`` via ``AsyncTask`` at the teacher endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 60,
        max_retries: int = 3,
        config_path: str = "config.toml",
        filter_noisy: bool = True,
        filter_interpretive: bool = True,
        allow_additions: bool = True,
    ) -> None:
        self._task = AsyncTask(
            config_path=config_path,
            endpoint=endpoint,
            completion_timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            truncate_tokens=2000,
        )
        self._task._served_model_name = model_name
        self._max_tokens = max_tokens
        self._system_prompt = build_teacher_prompt(
            filter_noisy=filter_noisy,
            filter_interpretive=filter_interpretive,
            allow_additions=allow_additions,
        )

    async def review_theses(
        self,
        text: str,
        theses: list[str],
    ) -> RefinementReview:
        """Send one teacher call and parse the response into a ``RefinementReview``."""
        user_message = (
            f"Text:\n{text}\n\nCandidate theses:\n"
            + "\n".join(f"- {t}" for t in theses)
        )
        raw: RawResponse = await self._task.analyze_raw(
            _get_session(),
            user_message,
            _RefinementPlanStub,
            system_prompt=self._system_prompt,
            max_tokens=self._max_tokens,
            truncate_tokens=2000,
        )
        if raw.status != ResponseStatus.OK or not raw.content:
            raise TeacherClientError(
                f"Teacher call failed with status {raw.status.value}"
                + (f": {raw.error}" if raw.error else "")
            )

        data = _parse_refinement_json(raw.content)

        return RefinementReview(
            keep=list(data.get("keep", [])),
            reformulate=[
                {"from": item["from"], "to": item["to"]}
                for item in data.get("reformulate", [])
            ],
            drop=[
                {"thesis": item["thesis"], "reason": item["reason"]}
                for item in data.get("drop", [])
            ],
            add=list(data.get("add", [])),
            latency_ms=raw.latency_ms,
            usage=raw.usage,
        )


_session: aiohttp.ClientSession | None = None


def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(limit=4)
        _session = aiohttp.ClientSession(connector=connector)
    return _session


async def close_session() -> None:
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None
