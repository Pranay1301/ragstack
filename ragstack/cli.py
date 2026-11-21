"""Try the stack:

    python3 -m ragstack.cli build <docs-dir> <index-dir>   # index once
    python3 -m ragstack.cli ask <index-dir> "question"     # reuse the index
    python3 -m ragstack.cli <docs-dir> ["question"]        # legacy one-shot

With no question, questions are read from stdin, one per line.
"""

from __future__ import annotations

import sys

from .ingest import load_documents
from .pipeline import RagStack


def _index_dir(stack: RagStack, docs_dir: str) -> None:
    for name, text in load_documents(docs_dir):
        n = stack.add_document(name, text)
        print(f"indexed {name}: {n} chunks", file=sys.stderr)
    stack.build()


def _ask(stack: RagStack, questions: list[str], retriever: str) -> None:
    for q in questions:
        answer = stack.answer(q, backend="auto", retriever=retriever)
        print(f"\nQ: {q}\nA: {answer.text}")
        for c in answer.citations:
            print(f"   [{c.document} chunk {c.chunk_index}]")


def main() -> int:
    args = sys.argv[1:]
    retriever = "hybrid"
    if "--retriever" in args:
        i = args.index("--retriever")
        retriever = args[i + 1]
        del args[i : i + 2]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "build":
        if len(args) != 3:
            raise SystemExit("usage: build <docs-dir> <index-dir>")
        stack = RagStack()
        _index_dir(stack, args[1])
        stack.save(args[2])
        print(f"saved index to {args[2]} ({stack.n_chunks} chunks)", file=sys.stderr)
        return 0
    if args[0] == "ask":
        if len(args) < 2:
            raise SystemExit("usage: ask <index-dir> [question ...]")
        stack = RagStack.load(args[1])
        questions = args[2:] or [ln.strip() for ln in sys.stdin if ln.strip()]
        _ask(stack, questions, retriever)
        return 0
    # legacy: cli <docs-dir> ["question"]
    stack = RagStack()
    _index_dir(stack, args[0])
    questions = args[1:] or [ln.strip() for ln in sys.stdin if ln.strip()]
    _ask(stack, questions, retriever)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
