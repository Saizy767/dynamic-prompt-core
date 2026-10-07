"""Infrastructure-internal scoring error.

This error is raised inside infrastructure to wrap library-specific failures
and invalid scores.  The ``LLMLogitCandidateScorer`` translates it into the
application-facing ``CandidateScoringError`` at the port boundary.  The
application never imports or receives this error.
"""
from __future__ import annotations


class LLMScoringError(Exception):
    """Raised inside infrastructure when scoring fails or produces invalid values."""
