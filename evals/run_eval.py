"""Run the retrieval + answer eval against the demo corpus.

Usage: python3 -m evals.run_eval [--corpus DIR] [--questions FILE] [--k 5]

Reports, per question and in total:
  - retrieval hit: the expected document appears in the top-k retrieved chunks
  - phrase hit: every key phrase appears in the composed extractive answer

The numbers printed here are the ONLY numbers the README may quote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragstack.pipeline import RagStack  # noqa: E402


def phrase_in(phrase: str, text: str) -> bool:
    return phrase.lower() in text.lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--corpus", default=str(here.parent / "data" / "demo-corpus"))
    parser.add_argument("--questions", default=str(here / "questions.json"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    stack = RagStack(max_chars=400, overlap=60)
    for doc in sorted(Path(args.corpus).glob("*.md")):
        stack.add_document(doc.name, doc.read_text())
    stack.build()

    questions = json.loads(Path(args.questions).read_text())
    retrieval_hits = 0
    phrase_hits = 0
    rows = []
    for q in questions:
        hits = stack.retrieve(q["question"], args.k)
        docs_hit = {stack._meta[h.position].document for h in hits}
        retrieval_ok = q["expected_document"] in docs_hit
        answer = stack.answer(q["question"], args.k, backend="extractive")
        missing = [p for p in q["key_phrases"] if not phrase_in(p, answer.text)]
        phrase_ok = not missing
        retrieval_hits += retrieval_ok
        phrase_hits += phrase_ok
        rows.append(
            (
                "OK " if retrieval_ok else "MISS",
                "OK " if phrase_ok else "MISS",
                q["question"],
                "" if phrase_ok else f"(missing: {', '.join(missing)})",
            )
        )

    for r_ok, p_ok, question, note in rows:
        print(f"retrieval:{r_ok} phrases:{p_ok}  {question} {note}")
    total = len(questions)
    print()
    print(f"retrieval top-{args.k}: {retrieval_hits}/{total}")
    print(f"answer key phrases:   {phrase_hits}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
