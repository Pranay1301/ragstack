"""Answer composing with citations.

The default composer is deterministic and fully offline: it selects the
sentences in the retrieved chunks that best overlap the query and stitches
them into an answer, citing the chunk each sentence came from. Determinism
is what makes the answer layer testable without an API key - the same
retrieval always yields the same answer.

An optional LLM backend (any OpenAI-compatible endpoint, e.g. a free-tier
Groq/OpenRouter key in the environment) can phrase the final answer, but
it receives ONLY the retrieved chunks as context and must cite chunk ids;
the project never claims LLM output quality without running the eval.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .embeddings import tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Citation:
    chunk_index: int
    document: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    backend: str  # "extractive" or the model name


def _sentence_scores(query_tokens: set[str], sentence: str) -> float:
    tokens = set(tokenize(sentence))
    if not tokens:
        return 0.0
    return len(query_tokens & tokens) / len(query_tokens | tokens)


def compose_extractive(
    query: str,
    retrieved: list[tuple[int, str, str]],  # (chunk_index, document, text)
    *,
    max_sentences: int = 4,
) -> Answer:
    """Stitch the most query-relevant retrieved sentences into an answer."""
    if not retrieved:
        return Answer(
            text="I don't have any documents that answer this yet.",
            citations=[],
            backend="extractive",
        )
    query_tokens = set(tokenize(query))
    scored: list[tuple[float, int, int, str]] = []  # (score, rank, chunk_index, sentence)
    for rank, (chunk_index, _doc, text) in enumerate(retrieved):
        for sentence in _SENTENCE_RE.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            scored.append((_sentence_scores(query_tokens, sentence), rank, chunk_index, sentence))
    scored = [s for s in scored if s[0] > 0.0]
    scored.sort(key=lambda s: (-s[0], s[1]))
    picked = scored[:max_sentences]
    if not picked:
        # Retrieval found chunks but no lexical overlap worth quoting;
        # answer with the top chunk's lead sentence rather than nothing.
        first = retrieved[0]
        lead = _SENTENCE_RE.split(first[2].strip())[0]
        return Answer(
            text=f"{lead} [1]",
            citations=[Citation(chunk_index=first[0], document=first[1])],
            backend="extractive",
        )
    citations: list[Citation] = []
    parts: list[str] = []
    for _score, _rank, chunk_index, sentence in picked:
        doc = next(d for ci, d, _t in retrieved if ci == chunk_index)
        try:
            ref = [c.chunk_index for c in citations].index(chunk_index) + 1
        except ValueError:
            citations.append(Citation(chunk_index=chunk_index, document=doc))
            ref = len(citations)
        parts.append(f"{sentence} [{ref}]")
    return Answer(text=" ".join(parts), citations=citations, backend="extractive")


def llm_available() -> bool:
    """True when an OpenAI-compatible endpoint is configured in the env."""
    return bool(os.environ.get("RAGSTACK_LLM_API_KEY"))


def compose_llm(
    query: str,
    retrieved: list[tuple[int, str, str]],
    *,
    model: str = "llama-3.1-8b-instant",
) -> Answer:
    """Phrase an answer with a free-tier LLM, context = retrieved chunks only.

    Deliberately small surface: one chat-completions POST, stdlib urllib,
    no SDK. Raises RuntimeError when unconfigured - callers fall back to
    ``compose_extractive``, which is also what the tests pin.
    """
    import json
    import urllib.request

    key = os.environ.get("RAGSTACK_LLM_API_KEY")
    base = os.environ.get("RAGSTACK_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    if not key:
        raise RuntimeError("set RAGSTACK_LLM_API_KEY to use the LLM backend")
    context = "\n\n".join(f"[chunk {ci} from {doc}]\n{text}" for ci, doc, text in retrieved)
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer ONLY from the provided chunks. Cite every claim "
                        "with [chunk N]. If the chunks do not answer the question, say so."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\n\n{context}"},
            ],
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())
    text = payload["choices"][0]["message"]["content"].strip()
    cited = sorted({int(n) for n in re.findall(r"\[chunk (\d+)\]", text)})
    citations = [
        Citation(chunk_index=ci, document=next(d for c, d, _t in retrieved if c == ci))
        for ci in cited
    ]
    return Answer(text=text, citations=citations, backend=model)
