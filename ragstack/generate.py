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

# Question/function words carry no topical signal; dropping them keeps the
# sentence-overlap score from being diluted by "how", "the", "an".
_STOPWORDS = frozenset(
    "the a an and or to of in on for with by at from as is are was were be been "
    "how what where when why which who does do did can could should would it its "
    "this that these those i you he she we they me him her us them".split()
)


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
    tokens = set(tokenize(sentence)) - _STOPWORDS
    probe = query_tokens - _STOPWORDS
    if not tokens or not probe:
        return 0.0
    return len(probe & tokens) / len(probe | tokens)


def _complete_sentences(text: str) -> list[str]:
    """Whole sentences only. Markdown headers are stripped first (a header
    line would otherwise swallow the first real sentence), then a leading
    fragment (passage starts lowercase, i.e. mid-sentence) and a trailing
    fragment (no terminal punctuation) are dropped.
    """
    t = " ".join(
        line for line in (ln.strip() for ln in text.splitlines()) if not line.startswith("#")
    )
    t = " ".join(t.split())
    if t and t[0].islower():
        m = _SENTENCE_RE.search(t)
        t = t[m.end() :] if m else ""
    last = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    t = t[: last + 1] if last >= 0 else ""
    return [s.strip() for s in _SENTENCE_RE.split(t) if s.strip()]


def _merge_passage(parts: list[str]) -> str:
    """Stitch consecutive overlapping chunks back into one passage."""
    merged = parts[0]
    for nxt in parts[1:]:
        limit = min(len(merged), len(nxt))
        k = 0
        for size in range(limit, 19, -1):  # overlap is >= 20 chars by config
            if merged[-size:] == nxt[:size]:
                k = size
                break
        merged = merged + nxt[k:]
    return merged


def compose_extractive(
    query: str,
    retrieved: list[tuple[int, str, str]],  # (chunk_index, document, text)
    *,
    max_sentences: int = 6,
) -> Answer:
    """Stitch the most query-relevant retrieved sentences into an answer."""
    if not retrieved:
        return Answer(
            text="I don't have any documents that answer this yet.",
            citations=[],
            backend="extractive",
        )
    query_tokens = set(tokenize(query))
    # Merge consecutive chunks of the same document into whole passages, so
    # sentences split across a chunk boundary score as complete sentences.
    # Group by document first: retrieval ranks by relevance, so a document's
    # adjacent chunks may not sit next to each other in the list. Passage
    # order follows each group's best (first-seen) rank.
    groups: dict[str, list[tuple[int, str]]] = {}
    order: list[str] = []
    for chunk_index, doc, text in retrieved:
        if doc not in groups:
            groups[doc] = []
            order.append(doc)
        groups[doc].append((chunk_index, text))
    passages: list[tuple[str, list[tuple[int, str]]]] = []
    for doc in order:
        parts = sorted(groups[doc])
        run = [parts[0]]
        for part in parts[1:]:
            if part[0] == run[-1][0] + 1:
                run.append(part)
            else:
                passages.append((doc, run))
                run = [part]
        passages.append((doc, run))

    scored: list[tuple[float, int, int, str, str]] = []  # (score, rank, chunk_index, doc, sentence)
    seen: set[str] = set()  # overlapping passages repeat sentences; pick each once
    for rank, (doc, parts) in enumerate(passages):
        merged = _merge_passage([text for _ci, text in parts])
        for sentence in _complete_sentences(merged):
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            # cite the constituent chunk that actually contains the sentence
            chunk_index = next(
                (ci for ci, text in parts if sentence[:40] in " ".join(text.split())),
                parts[0][0],
            )
            scored.append((_sentence_scores(query_tokens, sentence), rank, chunk_index, doc, sentence))
    scored = [s for s in scored if s[0] > 0.0]
    scored.sort(key=lambda s: (-s[0], s[1]))
    picked = scored[:max_sentences]
    if not picked:
        # Retrieval found chunks but no lexical overlap worth quoting;
        # answer with the top chunk's lead sentence rather than nothing.
        first = retrieved[0]
        complete = _complete_sentences(first[2])
        lead = complete[0] if complete else first[2].strip()
        return Answer(
            text=f"{lead} [1]",
            citations=[Citation(chunk_index=first[0], document=first[1])],
            backend="extractive",
        )
    citations: list[Citation] = []
    out_parts: list[str] = []
    for _score, _rank, chunk_index, doc, sentence in picked:
        # chunk_index is per-document, so the citation key is the (doc, chunk) pair
        key = (doc, chunk_index)
        existing = [i for i, c in enumerate(citations) if (c.document, c.chunk_index) == key]
        if existing:
            ref = existing[0] + 1
        else:
            citations.append(Citation(chunk_index=chunk_index, document=doc))
            ref = len(citations)
        out_parts.append(f"{sentence} [{ref}]")
    return Answer(text=" ".join(out_parts), citations=citations, backend="extractive")


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
