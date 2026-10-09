"""Infrastructure-internal scoring error.

This error is raised inside infrastructure to wrap library-specific failures
and invalid scores.  The ``LLMLogitCandidateScorer`` translates it into the
application-facing ``CandidateScoringError`` at the port boundary.  The
application never imports or receives this error.
"""
from __future__ import annotations


class LLMScoringError(Exception):
    """Raised inside infrastructure when scoring fails or produces invalid values."""


class GGUFProviderError(LLMScoringError):
    """Raised by the GGUF provider for GGUF-specific failures.

    Wraps missing optional dependencies, invalid or unreadable GGUF model
    files, context-window overflow, invalid logits shape or values, and
    tokenization boundary incompatibilities.  Flows through the existing
    ``LLMScoringError`` → ``CandidateScoringError`` boundary so the
    application never imports or receives this error directly.
    """
