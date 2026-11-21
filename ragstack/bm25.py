"""BM25 (Okapi) ranking implemented from scratch on NumPy.

TF-IDF cosine ranks by normalized presence; BM25 adds two things the flat
index does not have: term-frequency saturation (repeating a word 10 times
is not 10x the evidence of repeating it twice) and document-length
normalization (long documents should not win just for containing more
words). ``k1`` controls saturation, ``b`` controls length normalization;
the defaults (1.5, 0.75) are the standard textbook values.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .embeddings import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._len: list[int] = []
        self._avgdl = 0.0
        self._n = 0

    def fit(self, texts: list[str]) -> "BM25Index":
        if not texts:
            raise ValueError("cannot fit on an empty corpus")
        self._tf = [Counter(tokenize(t)) for t in texts]
        self._len = [sum(c.values()) for c in self._tf]
        self._avgdl = (sum(self._len) / len(self._len)) or 1.0
        self._df = Counter()
        for counts in self._tf:
            self._df.update(counts.keys())
        self._n = len(texts)
        return self

    def __len__(self) -> int:
        return self._n

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document against ``query`` (higher is better)."""
        out = np.zeros(self._n, dtype=np.float64)
        for term in set(tokenize(query)):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self._tf):
                f = counts.get(term, 0)
                if not f:
                    continue
                norm = 1.0 - self.b + self.b * self._len[i] / self._avgdl
                out[i] += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * norm)
        return out

    def rank(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Top-``k`` (position, score) pairs, best first."""
        if self._n == 0 or k <= 0:
            return []
        s = self.scores(query)
        k = min(k, self._n)
        top = np.argsort(s, kind="stable")[-k:][::-1]
        return [(int(i), float(s[i])) for i in top]
