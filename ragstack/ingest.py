"""Document ingestion: load a directory of .md/.txt/.html files as text.

HTML is stripped to visible text with the stdlib parser - no dependency,
no network fetch. Script and style blocks are dropped, and block-level
tags become newlines so the chunker still sees document structure.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

_SUFFIXES = {".md", ".txt", ".html", ".htm"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "li", "ul", "ol",
    "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre",
}
_SKIP_TAGS = {"script", "style", "noscript", "template"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS or tag == "br":
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(ln.split()) for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def load_documents(docs_dir: str) -> list[tuple[str, str]]:
    """Return ``(relative_name, text)`` for every supported file under a dir."""
    root = Path(docs_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in _SUFFIXES)
    if not files:
        raise SystemExit(f"no .md/.txt/.html files under {docs_dir}")
    docs: list[tuple[str, str]] = []
    for p in files:
        raw = p.read_text(encoding="utf-8", errors="replace")
        text = html_to_text(raw) if p.suffix in {".html", ".htm"} else raw
        docs.append((str(p.relative_to(root)), text))
    return docs
