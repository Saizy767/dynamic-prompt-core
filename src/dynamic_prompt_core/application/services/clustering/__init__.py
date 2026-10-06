"""Clustering service: cosine similarity, embeddings, and cluster assignment.

This service uses numpy and sentence-transformers, so it resides in the
application layer rather than domain. Functions are implemented here directly
to avoid a cyclic dependency with application.use_cases.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_embedding(text: str, model: Any = None) -> list[float]:
    """Compute embedding for a text using a sentence-transformers model."""
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError as exc:
            raise ImportError(
                "Embeddings require 'sentence-transformers'; "
                "install with: pip install sentence-transformers"
            ) from exc
    embedding = model.encode(text)
    return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


__all__ = ["compute_embedding", "cosine_similarity"]
