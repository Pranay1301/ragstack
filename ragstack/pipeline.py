"""The assembled retrieval stack: documents in, cited answers out."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import split_text
from .embeddings import TfidfEmbedder
from .generate import Answer, compose_extractive, compose_llm, llm_available
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

    def add_document(self, name: str, text: str) -> int:
        """Chunk and store a document. Returns how many chunks it produced."""
        chunks = split_text(text, max_chars=self.max_chars, overlap=self.overlap)
        for chunk in chunks:
            self._meta.append(
                StoredChunk(document=name, chunk_index=chunk.index, text=chunk.text)
            )
            self._texts.append(chunk.text)
        self._embedder = None  # corpus changed; embeddings are stale
        return len(chunks)

    def build(self) -> None:
        if not self._texts:
            raise RuntimeError("add documents before build()")
        self._embedder = TfidfEmbedder()
        matrix = self._embedder.fit_transform(self._texts)
        self._index = VectorIndex(matrix.shape[1])
        self._index.add(matrix)

    @property
    def n_chunks(self) -> int:
        return len(self._texts)

    def retrieve(self, query: str, k: int = 5, *, use_mmr: bool = True) -> list[Retrieved]:
        if self._embedder is None or self._index is None:
            raise RuntimeError("call build() after adding documents")
        q = self._embedder.transform([query])[0]
        return mmr(self._index, q, k) if use_mmr else mmr(self._index, q, k, lambda_rel=1.0)

    def answer(self, query: str, k: int = 5, *, backend: str = "auto") -> Answer:
        hits = self.retrieve(query, k)
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
