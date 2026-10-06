"""Response dataclasses and status enums for the LLM transport layer."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class ResponseStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    UNEXPECTED_SHAPE = "unexpected_shape"


class ParseStatus(StrEnum):
    OK = "ok"
    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    TRUNCATED = "truncated"


@dataclass
class RawResponse:
    content: str | None
    model: str
    latency_ms: float
    status: ResponseStatus
    attempts: int
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    http_status: int | None = None
    error: str | None = None
    truncated: bool = False


@dataclass
class CallResult:
    """Rich result bundling parsed model, raw content, latency, and parse status."""
    parsed: BaseModel | None
    raw_content: str | None
    latency_ms: float
    parse_status: ParseStatus | None
    status: ResponseStatus
    finish_reason: str | None = None
    error: str | None = None
    truncated: bool = False
