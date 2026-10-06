"""Domain service: dataset splitting into dev, holdout, and ambiguous groups."""
from __future__ import annotations

import random

from dynamic_prompt_core.domain.errors.dataset import DatasetError
from dynamic_prompt_core.domain.models.dataset import (
    DEFAULT_HOLDOUT_RATIO,
    DEFAULT_SEED,
    Dataset,
    Record,
)


def split_dataset(
    records: list[Record],
    seed: int = DEFAULT_SEED,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    source_path: str = "",
) -> Dataset:
    """Split records into dev, holdout, and ambiguous groups.

    The split is reproducible: records are sorted by id (deterministic order
    independent of file row order), then shuffled with random.Random(seed).
    Ambiguous records are separated before the shuffle.
    """
    ambiguous = [r for r in records if r.ambiguous]
    non_ambiguous = [r for r in records if not r.ambiguous]

    ordered = sorted(non_ambiguous, key=lambda r: str(r.id))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    cut = int(n * holdout_ratio)
    if n > 0 and not (0 < cut < n):
        raise DatasetError(
            f"holdout_ratio {holdout_ratio} produces an invalid split: "
            f"cut={cut} for {n} non-ambiguous records. "
            f"Choose a ratio strictly between 0 and 1."
        )

    holdout = ordered[:cut]
    dev = ordered[cut:]

    return Dataset(
        dev=dev,
        holdout=holdout,
        ambiguous=ambiguous,
        seed=seed,
        holdout_ratio=holdout_ratio,
        source_path=source_path,
    )
