"""Domain-level scoring error.

This is the application-facing error raised when candidate scoring fails.
Infrastructure translates its internal failures into ``CandidateScoringError``
so the application never imports infrastructure error classes.  The error
resides in ``domain.errors`` (rather than ``application.errors``) because
``infrastructure`` already imports ``domain`` (the intended dependency
direction), while importing ``application`` would create a cycle with the
pre-existing ``application.use_cases → infrastructure`` migration debt.
"""
from __future__ import annotations


class CandidateScoringError(Exception):
    """Raised when candidate scoring fails for any infrastructure reason."""
