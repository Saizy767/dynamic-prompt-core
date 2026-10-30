"""Domain error for atomic file writes."""

from __future__ import annotations


class AtomicWriteError(OSError):
    """Raised when an atomic write or its read-back validation fails."""
