"""Per-stage performance instrumentation for the scoring path.

Records monotonic timings (milliseconds) and integer counters for one
scoring request so a single run can localize the dominant bottleneck.

Schema (all timings are non-negative ``float`` milliseconds measured with
``time.perf_counter``; all counters are non-negative ``int``):

    Timings: lock_wait_ms, reset_ms, eval_ms, logits_read_ms,
             logits_convert_ms, logits_validate_ms, scoring_ms
    Counters: input_tokens, scored_tokens, vocab_size, candidate_count

The profile NEVER stores prompt text, candidate text, raw logit arrays,
or any other user content.  ``eval_ms`` measures only the
``model.eval(...)`` call; ``logits_read_ms`` measures access to
``eval_logits`` including any lazy materialization triggered by that
access, so model evaluation is separable from logit access/processing.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

TIMING_FIELDS: tuple[str, ...] = (
    "lock_wait_ms",
    "reset_ms",
    "eval_ms",
    "logits_read_ms",
    "logits_convert_ms",
    "logits_validate_ms",
    "scoring_ms",
)

COUNTER_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "scored_tokens",
    "vocab_size",
    "candidate_count",
)


@dataclass
class ScoringProfile:
    """Per-stage timing and counter record for one scoring request.

    All timing fields are non-negative milliseconds measured with a
    monotonic clock (``time.perf_counter``).  Counters are non-negative
    integers.  The profile never stores prompt text, candidate text, or
    raw logit arrays.
    """

    lock_wait_ms: float = 0.0
    reset_ms: float = 0.0
    eval_ms: float = 0.0
    logits_read_ms: float = 0.0
    logits_convert_ms: float = 0.0
    logits_validate_ms: float = 0.0
    scoring_ms: float = 0.0
    input_tokens: int = 0
    scored_tokens: int = 0
    vocab_size: int = 0
    candidate_count: int = 0

    def as_dict(self) -> dict[str, float | int]:
        """Return a plain-dict view of the profile for logging or JSON."""
        result: dict[str, float | int] = {}
        for name in TIMING_FIELDS:
            result[name] = float(getattr(self, name))
        for name in COUNTER_FIELDS:
            result[name] = int(getattr(self, name))
        return result

    def timing_total_ms(self) -> float:
        """Sum of all timing fields (the measured wall-clock stages)."""
        return float(sum(getattr(self, name) for name in TIMING_FIELDS))


@contextmanager
def timed_stage(profile: ScoringProfile, attr: str) -> Iterator[None]:
    """Time a stage and accumulate it into ``profile.<attr>`` (ms, non-negative).

    The elapsed time is *added* to the field's current value so multiple
    sequential uses (e.g. one per batch item) sum naturally without
    double-counting.  The elapsed time is recorded in a ``finally`` block so
    it remains available even when the stage raises.  The exception is not
    swallowed — instrumentation does not alter exception propagation.  Stages
    are sequential and non-overlapping by construction, so timing each into
    its own field does not count a single interval twice.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        setattr(profile, attr, getattr(profile, attr) + (elapsed if elapsed > 0.0 else 0.0))
