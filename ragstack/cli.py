"""Try the stack: python3 -m ragstack.cli <docs-dir> ["question"]

With no question it reads them from stdin, one per line.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .pipeline import RagStack


def _load(stack: RagStack, docs_dir: str) -> None:
    files = sorted(
        p for p in Path(docs_dir).rglob("*") if p.suffix in {".md", ".txt"} and p.is_file()
    )
    if not files:
        raise SystemExit(f"no .md/.txt files under {docs_dir}")
    for p in files:
        n = stack.add_document(str(p.relative_to(docs_dir)), p.read_text())
        print(f"indexed {p.name}: {n} chunks", file=sys.stderr)
    stack.build()


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    stack = RagStack()
    _load(stack, sys.argv[1])
    backend = "auto"
    questions = sys.argv[2:] or [line.strip() for line in sys.stdin if line.strip()]
    for q in questions:
        answer = stack.answer(q, backend=backend)
        print(f"\nQ: {q}\nA: {answer.text}")
        for c in answer.citations:
            print(f"   [{c.document} chunk {c.chunk_index}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
