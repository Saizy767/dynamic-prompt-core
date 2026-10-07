"""Domain model for the application-level classification decision."""
from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment


@dataclass(frozen=True)
class Classification:
    """The application-level classification decision derived from judgments.

    Associates the selected ``Candidate`` with the full judgment set the decision
    was derived from.  Judgments are stored as a ``tuple`` so the sequence is
    deeply immutable (``frozen=True`` prevents field reassignment; the tuple
    prevents ``append``/``__setitem__``).

    The constructor accepts ``judgments`` as any iterable of ``Judgment`` and
    stores ``tuple(judgments)``.

    ``Classification`` is distinct from the legacy generative
    ``ClassificationResult`` schema and does not replace it.
    """

    selected: Candidate
    judgments: tuple[Judgment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.judgments, tuple):
            object.__setattr__(self, "judgments", tuple(self.judgments))
        if not self.judgments:
            raise ValueError("judgments must not be empty")
        judgment_candidates = {j.candidate for j in self.judgments}
        if self.selected not in judgment_candidates:
            raise ValueError(
                f"selected candidate {self.selected.value!r} is not among "
                f"the candidates in judgments"
            )
