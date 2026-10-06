"""Domain service: rule candidate selection logic (frequency, precision, top-N)."""
from __future__ import annotations

from typing import Any


def cluster_in_prompt(
    cluster: dict[str, Any], theses: dict[str, Any]
) -> bool:
    """Derive a cluster's in_prompt flag from its member theses."""
    member_norms = cluster.get("member_norms", [])
    if not member_norms:
        return False
    for norm in member_norms:
        entry = theses.get(norm)
        if entry is None or not entry.get("in_prompt", False):
            return False
    return True


def filter_by_frequency(
    clusters: list[dict[str, Any]], threshold: int
) -> list[dict[str, Any]]:
    """Return clusters with frequency >= threshold."""
    return [c for c in clusters if c["frequency"] >= threshold]


def exclude_in_prompt(
    clusters: list[dict[str, Any]], theses: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split clusters into (candidates, already_in_prompt)."""
    candidates: list[dict[str, Any]] = []
    already: list[dict[str, Any]] = []
    for c in clusters:
        if cluster_in_prompt(c, theses):
            already.append(c)
        else:
            candidates.append(c)
    return candidates, already


def rank_by_precision(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort clusters by precision descending, frequency descending as tie-breaker."""
    return sorted(
        clusters,
        key=lambda c: (c["precision"], c["frequency"]),
        reverse=True,
    )


def select_top_n(
    ranked_clusters: list[dict[str, Any]], top_n: int
) -> list[dict[str, Any]]:
    """Return the first top_n entries, or all when fewer remain."""
    return ranked_clusters[:top_n]


def representative_theses(
    cluster: dict[str, Any],
    theses: dict[str, Any],
    max_theses: int,
) -> list[dict[str, Any]]:
    """Return up to max_theses member theses sorted by descending frequency."""
    member_norms = cluster.get("member_norms", [])
    entries: list[dict[str, Any]] = []
    for norm in member_norms:
        entry = theses.get(norm)
        if entry is None:
            continue
        entries.append(entry)
    entries.sort(
        key=lambda e: (e.get("frequency", 0), e.get("positive_hits", 0)),
        reverse=True,
    )
    result: list[dict[str, Any]] = []
    for e in entries[:max_theses]:
        result.append(
            {
                "text_norm": e.get("text_norm", ""),
                "text_raw": e.get("text_raw", ""),
                "frequency": e.get("frequency", 0),
            }
        )
    return result
