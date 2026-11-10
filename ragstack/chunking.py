"""Structure-aware text chunking with overlap.

The splitter tries paragraph, then line, then sentence, then word
boundaries, packing as much as fits under ``max_chars`` and carrying a
trailing window of ``overlap`` characters into the next chunk so context
is not lost at boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEPARATOR_RE = re.compile(r"(\n\n+|\n|(?<=[.!?])\s+|\s+)")


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    start: int  # char offset of chunk start in the source document


def _split_units(text: str) -> list[str]:
    """Split into units, attaching each separator to the unit it ends.

    ``re.split`` with a capture group yields [text, sep, text, sep, ...];
    every separator closes the unit it follows, so a unit always ends on
    the coarsest boundary seen (paragraph > line > sentence > space).
    """
    units: list[str] = []
    buf = ""
    for part in _SEPARATOR_RE.split(text):
        if not part:
            continue
        buf += part
        if _SEPARATOR_RE.fullmatch(part):
            units.append(buf)
            buf = ""
    if buf:
        units.append(buf)
    return [u for u in units if u.strip()]


def split_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Every chunk (except possibly the last) ends on a paragraph, line,
    sentence or word boundary, in that order of preference. ``overlap``
    characters of trailing context are repeated at the head of the next
    chunk.
    """
    if not text or not text.strip():
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")

    units = _split_units(text)
    chunks: list[Chunk] = []
    buf = ""
    buf_start = 0  # offset of buf's first char within the document
    consumed = 0  # document offset just past the last unit appended

    def flush() -> None:
        nonlocal buf, buf_start
        if buf.strip():
            chunks.append(Chunk(text=buf.strip(), index=len(chunks), start=buf_start))
        if overlap and buf:
            tail = buf[-overlap:]
            # start the overlap window on a word boundary when possible
            space = tail.find(" ")
            if 0 < space < len(tail) - 1:
                tail = tail[space + 1 :]
            buf_start = consumed - len(tail)
            buf = tail
        else:
            buf_start = consumed
            buf = ""

    for unit in units:
        if buf and len(buf) + len(unit) > max_chars:
            flush()
        if len(unit) > max_chars:
            # hard-split an oversized single unit on raw characters
            for i in range(0, len(unit), max_chars):
                piece = unit[i : i + max_chars]
                if buf and len(buf) + len(piece) > max_chars:
                    flush()
                buf += piece
                consumed += len(piece)
                if len(buf) >= max_chars:
                    flush()
            continue
        if not buf:
            buf_start = consumed
        buf += unit
        consumed += len(unit)
    if buf.strip():
        chunks.append(Chunk(text=buf.strip(), index=len(chunks), start=buf_start))
    return chunks
