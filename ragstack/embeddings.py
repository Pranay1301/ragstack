"""TF-IDF embeddings implemented from scratch on NumPy.

``TfidfEmbedder.fit`` learns the vocabulary and inverse-document
frequencies on a corpus; ``transform`` maps new texts into the same
L2-normalized sparse space (as dense float32 rows, which is fine at the
scale this project targets: thousands of chunks, not millions).

Sub-linear TF scaling (1 + log tf) and cosine-ready L2 normalization
follow the standard recipe, so dot products on transformed rows are
cosine similarities.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfEmbedder:
    def __init__(self, *, max_features: int = 30000, min_df: int = 1) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        if not texts:
            raise ValueError("cannot fit on an empty corpus")
        df: Counter[str] = Counter()
        tf_total: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize(text)
            df.update(set(tokens))
            tf_total.update(tokens)
        # Keep the most frequent terms that pass min_df, capped at max_features.
        terms = [t for t, c in tf_total.most_common() if df[t] >= self.min_df]
        terms = terms[: self.max_features]
        terms.sort()
        self.vocab_ = {t: i for i, t in enumerate(terms)}
        n_docs = len(texts)
        self.idf_ = np.array(
            [math.log((1 + n_docs) / (1 + df[t])) + 1.0 for t in terms],
            dtype=np.float32,
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("embedder is not fitted")
        rows = np.zeros((len(texts), len(self.vocab_)), dtype=np.float32)
        for r, text in enumerate(texts):
            counts: Counter[str] = Counter(
                t for t in tokenize(text) if t in self.vocab_
            )
            for tok, c in counts.items():
                rows[r, self.vocab_[tok]] = 1.0 + math.log(c)
        rows *= self.idf_
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # empty/OOV text -> zero vector, not NaN
        return rows / norms

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @property
    def n_features(self) -> int:
        return len(self.vocab_)
