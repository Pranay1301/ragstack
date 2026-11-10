"""Retrieval: top-k search plus MMR diversification.

Maximal Marginal Relevance (Carbonell & Goldstein, 1998) trades a little
relevance for coverage, so a top-k list is not five near-duplicates of
the same paragraph. ``lambda_rel`` = 1.0 is pure relevance, 0.0 is pure
diversity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import VectorIndex


@dataclass(frozen=True)
class Retrieved:
    position: int
    score: float  # similarity to the query


def top_k(index: VectorIndex, query: np.ndarray, k: int = 5) -> list[Retrieved]:
    return [Retrieved(position=h.position, score=h.score) for h in index.search(query, k)]


def mmr(
    index: VectorIndex,
    query: np.ndarray,
    k: int = 5,
    *,
    candidates: int = 20,
    lambda_rel: float = 0.7,
) -> list[Retrieved]:
    if not 0.0 <= lambda_rel <= 1.0:
        raise ValueError("lambda_rel must be in [0, 1]")
    pool = index.search(query, max(k, candidates))
    if not pool:
        return []
    selected: list[Retrieved] = []
    remaining = {h.position: h.score for h in pool}
    # First pick: the most relevant candidate.
    first = max(remaining, key=lambda p: remaining[p])
    selected.append(Retrieved(position=first, score=remaining.pop(first)))
    while remaining and len(selected) < k:
        best_pos = -1
        best_val = -np.inf
        for pos, rel in remaining.items():
            redundancy = max(
                float(index.vector_at(pos) @ index.vector_at(s.position))
                for s in selected
            )
            val = lambda_rel * rel - (1.0 - lambda_rel) * redundancy
            if val > best_val:
                best_val = val
                best_pos = pos
        selected.append(Retrieved(position=best_pos, score=remaining.pop(best_pos)))
    return selected
