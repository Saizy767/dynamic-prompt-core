"""Outbound port: contract for key-phrase extraction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ExtractionResult(Protocol):
    """Result of a key-phrase extraction call.

    Concrete implementations expose additional infrastructure-specific
    fields (e.g. ``call_result``); this interface declares only the
    application-level attributes.
    """

    theses_raw: list[str]
    theses_norm: list[str]
    status: str
    latency_ms: float
    raw_extract: str | None

    @property
    def call_status(self) -> str | None:
        """Transport-level status name of the underlying call, if any.

        Values follow the transport's status names (e.g. ``"ok"``,
        ``"network_error"``, ``"http_error"``, ``"timeout"``); used by
        callers to bucket extraction failures.
        """
        ...


@runtime_checkable
class Extractor(Protocol):
    """Outbound port: extract key phrases from input text.

    The extractor performs an independent model call with its own system
    prompt.  It does not receive classification results.  Implementations
    own their HTTP transport; callers release it with ``aclose``.
    """

    async def extract(
        self,
        text: str,
        true_val: int | None = None,
    ) -> ExtractionResult:
        """Extract key phrases from ``text``."""
        ...

    async def aclose(self) -> None:
        """Release the underlying transport resources."""
        ...

    @staticmethod
    def skipped() -> ExtractionResult:
        """Return a skipped extraction result."""
        ...
