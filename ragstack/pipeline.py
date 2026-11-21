"""The assembled retrieval stack: documents in, cited answers out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .chunking import split_text
from .embeddings import TfidfEmbedder, tokenize
from .fusion import rrf
from .generate import _STOPWORDS, Answer, compose_extractive, compose_llm, llm_available
from .index import VectorIndex
from .retrieve import Retrieved, mmr


@dataclass(frozen=True)
class StoredChunk:
    document: str
    chunk_index: int
    text: str


class RagStack:
    def __init__(self, *, max_chars: int = 800, overlap: int = 120) -> None:
        self.max_chars = max_chars
        self.overlap = overlap
        self._texts: list[str] = []
        self._meta: list[StoredChunk] = []
        self._embedder: TfidfEmbedder | None = None
        self._index: VectorIndex | None = None
        self._bm25: BM25Index | None = None

    def add_document(self, name: str, text: str) -> int:
        """Chunk and store a document. Returns how many chunks it produced."""
        chunks = split_text(text, max_chars=self.max_chars, overlap=self.overlap)
        for chunk in chunks:
            self._meta.append(
                StoredChunk(document=name, chunk_index=chunk.index, text=chunk.text)
            )
            self._texts.append(chunk.text)
        self._embedder = None  # corpus changed; embeddings are stale
        self._bm25 = None
        return len(chunks)

    def build(self) -> None:
        if not self._texts:
            raise RuntimeError("add documents before build()")
        self._embedder = TfidfEmbedder()
        matrix = self._embedder.fit_transform(self._texts)
        self._index = VectorIndex(matrix.shape[1])
        self._index.add(matrix)
        self._bm25 = BM25Index().fit(self._texts)

    @property
    def n_chunks(self) -> int:
        return len(self._texts)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        use_mmr: bool = True,
        retriever: str = "hybrid",
    ) -> list[Retrieved]:
        """Rank chunks for a query.

        ``retriever`` is ``"tfidf"`` (cosine + MMR), ``"bm25"`` (Okapi
        BM25) or ``"hybrid"`` (reciprocal rank fusion of the two).
        """
        if self._embedder is None or self._index is None or self._bm25 is None:
            raise RuntimeError("call build() after adding documents")
        q = self._embedder.transform([query])[0]
        if retriever == "tfidf":
            return mmr(self._index, q, k) if use_mmr else mmr(self._index, q, k, lambda_rel=1.0)
        if retriever == "bm25":
            return [Retrieved(position=p, score=s) for p, s in self._bm25.rank(query, k)]
        if retriever == "hybrid":
            tfidf_ranking = [(h.position, h.score) for h in mmr(self._index, q, max(k, 10))]
            bm25_ranking = self._bm25.rank(query, max(k, 10))
            return [Retrieved(position=p, score=s) for p, s in rrf([tfidf_ranking, bm25_ranking])[:k]]
        raise ValueError(f"unknown retriever {retriever!r}")

    def confidence(self, query: str, hits: list[Retrieved]) -> float:
        """Retrieval confidence in [0, 1]: the best TF-IDF cosine among hits.

        Retriever-independent (RRF and BM25 scores have no fixed scale, so
        the cosine against the query vector is the comparable signal).
        """
        if not hits or self._embedder is None or self._index is None:
            return 0.0
        # Question/function words ("the", "of", "what") live in every
        # document; without stripping them an off-topic question still
        # scores nonzero and the abstention gate never fires.
        probe = " ".join(t for t in tokenize(query) if t not in _STOPWORDS)
        if not probe:
            return 0.0
        q = self._embedder.transform([probe])[0]
        return max(float(self._index.vector_at(h.position) @ q) for h in hits)

    def answer(
        self,
        query: str,
        k: int = 5,
        *,
        backend: str = "auto",
        retriever: str = "hybrid",
        min_confidence: float = 0.02,
    ) -> Answer:
        hits = self.retrieve(query, k, retriever=retriever)
        if self.confidence(query, hits) < min_confidence:
            return Answer(
                text=(
                    "I couldn't find this in the documents. The closest "
                    "passages have no meaningful overlap with the question, "
                    "so I would rather say that than guess."
                ),
                citations=[],
                backend="abstain",
                abstained=True,
            )
        retrieved = [
            (self._meta[h.position].chunk_index, self._meta[h.position].document, self._meta[h.position].text)
            for h in hits
        ]
        if backend == "llm" or (backend == "auto" and llm_available()):
            try:
                return compose_llm(query, retrieved)
            except Exception:
                if backend == "llm":
                    raise
        return compose_extractive(query, retrieved)

    def chunk_text(self, position: int) -> str:
        return self._meta[position].text

    # -- persistence -----------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the built stack so later runs skip re-indexing."""
        if self._embedder is None or self._index is None or self._bm25 is None:
            raise RuntimeError("call build() before save()")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._index.save(str(path / "vectors.npz"))
        payload = {
            "version": 2,
            "max_chars": self.max_chars,
            "overlap": self.overlap,
            "embedder": self._embedder.to_dict(),
            "texts": self._texts,
            "meta": [
                {"document": m.document, "chunk_index": m.chunk_index} for m in self._meta
            ],
        }
        (path / "stack.json").write_text(json.dumps(payload))

    @classmethod
    def load(cls, directory: str) -> "RagStack":
        """Load a stack saved with :meth:`save`. Retrieval is bit-identical."""
        path = Path(directory)
        payload = json.loads((path / "stack.json").read_text())
        if payload.get("version") != 2:
            raise ValueError(f"unsupported saved-stack version {payload.get('version')!r}")
        stack = cls(max_chars=int(payload["max_chars"]), overlap=int(payload["overlap"]))
        stack._texts = list(payload["texts"])
        stack._meta = [
            StoredChunk(document=m["document"], chunk_index=int(m["chunk_index"]), text=t)
            for m, t in zip(payload["meta"], stack._texts)
        ]
        stack._embedder = TfidfEmbedder.from_dict(payload["embedder"])
        stack._index = VectorIndex.load(str(path / "vectors.npz"))
        stack._bm25 = BM25Index().fit(stack._texts)
        return stack
