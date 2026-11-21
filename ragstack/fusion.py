"""Reciprocal rank fusion (RRF) of several rankings into one.

Each retriever returns its own ranking on its own score scale, so the raw
scores cannot be mixed. RRF (Cormack, Clarke & Buettcher, 2009) fuses them
by rank position only: a document earns ``1 / (k + rank)`` from each
ranking it appears in, and the sums decide the fused order. ``k`` (default
60, the paper's value) damps how much a top-1 finish dominates a top-5.
"""

from __future__ import annotations


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse rankings of (position, score) into fused (position, rrf_score)."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (position, _score) in enumerate(ranking):
            scores[position] = scores.get(position, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
