"""Errors for the LLM transport layer."""
from __future__ import annotations


class ConfigError(ValueError):
    """Raised when config is missing, invalid, or backend cannot be determined."""


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} on {url}: {body[:500]}")
        self.status = status
        self.url = url
        self.body = body


class UnexpectedShapeError(RuntimeError):
    """Raised when the LLM response does not match the expected JSON shape."""
