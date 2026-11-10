"""A flat cosine-similarity vector index.

Vectors must be L2-normalized (``TfidfEmbedder.transform`` already does
this), so ranking is a single matrix-vector product. Exact search, no
approximation: at this project's scale brute force is both simpler and
faster to verify than an ANN structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hit:
    position: int  # row in the index
    score: float  # cosine similarity in [-1, 1]


class VectorIndex:
    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None  # cached stacked copy

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        for row in vectors:
            self._rows.append(row)
        self._matrix = None

    def _stacked(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (
                np.stack(self._rows) if self._rows else np.zeros((0, self.dim), np.float32)
            )
        return self._matrix

    def search(self, query: np.ndarray, k: int = 5) -> list[Hit]:
        if len(self) == 0 or k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"expected query dim {self.dim}, got {query.shape[0]}")
        scores = self._stacked() @ query
        k = min(k, len(self))
        # argsort ascending, take the tail, reverse for descending score
        top = np.argsort(scores)[-k:][::-1]
        return [Hit(position=int(i), score=float(scores[i])) for i in top]

    def vector_at(self, position: int) -> np.ndarray:
        return self._stacked()[position]

    def save(self, path: str) -> None:
        np.savez_compressed(path, matrix=self._stacked(), dim=self.dim)

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        data = np.load(path)
        index = cls(int(data["dim"]))
        index.add(data["matrix"])
        return index
