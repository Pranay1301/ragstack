"""Run the retrieval + answer + abstention eval against the demo corpus.

Usage: python3 -m evals.run_eval [--corpus DIR] [--k 5]

Fixtures:
  - questions.json       the original 15 in-scope questions (unchanged)
  - wording_probe.json   5 wording-shifted in-scope questions (added in 0.2)
  - out_of_scope.json    5 questions the corpus cannot answer

Reports, printed so misses are visible rather than hidden:
  - per retriever (tfidf / bm25 / hybrid): recall@k and MRR@k per fixture
  - answer key-phrase coverage (extractive composer, hybrid retrieval)
  - abstention: out-of-scope questions must abstain, in-scope must not

The numbers printed here are the ONLY numbers the README may quote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragstack.pipeline import RagStack  # noqa: E402

RETRIEVERS = ("tfidf", "bm25", "hybrid")


def phrase_in(phrase: str, text: str) -> bool:
    return phrase.lower() in text.lower()


def recall_and_mrr(stack: RagStack, questions: list[dict], k: int, retriever: str):
    hits = 0
    rr_sum = 0.0
    for q in questions:
        results = stack.retrieve(q["question"], k, retriever=retriever)
        docs = [stack._meta[h.position].document for h in results]
        if q["expected_document"] in docs:
            hits += 1
            rr_sum += 1.0 / (docs.index(q["expected_document"]) + 1)
    total = len(questions)
    return hits, total, rr_sum / total


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--corpus", default=str(here.parent / "data" / "demo-corpus"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    stack = RagStack(max_chars=400, overlap=60)
    for doc in sorted(Path(args.corpus).glob("*.md")):
        stack.add_document(doc.name, doc.read_text())
    stack.build()

    original = json.loads((here / "questions.json").read_text())
    probe = json.loads((here / "wording_probe.json").read_text())
    out_of_scope = json.loads((here / "out_of_scope.json").read_text())
    in_scope = original + probe

    print("== retrieval per retriever ==")
    for retriever in RETRIEVERS:
        for name, fixture in (("original", original), ("wording-probe", probe)):
            hits, total, mrr = recall_and_mrr(stack, fixture, args.k, retriever)
            print(f"{retriever:7s} {name:14s} recall@{args.k}: {hits}/{total}   MRR@{args.k}: {mrr:.3f}")
    print()

    print("== extractive answers (hybrid retrieval) ==")
    phrase_hits = 0
    for q in in_scope:
        answer = stack.answer(q["question"], args.k, backend="extractive")
        missing = [p for p in q["key_phrases"] if not phrase_in(p, answer.text)]
        phrase_hits += not missing
        flag = "OK " if not missing else "MISS"
        note = "" if not missing else f"(missing: {', '.join(missing)})"
        print(f"phrases:{flag} abstained:{str(answer.abstained):5s} {q['question']} {note}")
    print(f"answer key phrases: {phrase_hits}/{len(in_scope)}")
    print()

    print("== abstention gate ==")
    correct = 0
    total = 0
    for q in out_of_scope:
        answer = stack.answer(q["question"], args.k, backend="extractive")
        ok = answer.abstained
        correct += ok
        total += 1
        print(f"{'OK ' if ok else 'FAIL'} should-abstain   abstained={answer.abstained}  {q['question']}")
    for q in in_scope:
        answer = stack.answer(q["question"], args.k, backend="extractive")
        ok = not answer.abstained
        correct += ok
        total += 1
        if not ok:
            print(f"FAIL should-answer    abstained={answer.abstained}  {q['question']}")
    print(f"abstention decisions correct: {correct}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
