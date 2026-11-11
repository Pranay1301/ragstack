import pytest

from ragstack.generate import compose_extractive
from ragstack.pipeline import RagStack

DOCS = {
    "pos.md": (
        "The point of sale terminal records every order at the counter. "
        "Staff ring up items, apply taxes, and print bills in one flow. "
        "Offline mode keeps the terminal working when the internet drops.\n\n"
        "Orders sync to the back office when connectivity returns."
    ),
    "kitchen.md": (
        "The kitchen display system shows incoming tickets to the cooks. "
        "Drinks route to the bar station and food routes to the hot line. "
        "Cooks tap a ticket to mark it preparing, then ready.\n\n"
        "A new ticket plays a sound so staff notice it immediately."
    ),
    "physics.md": (
        "Quantum entanglement correlates measurements of distant particles. "
        "The correlations violate Bell inequalities, which no local hidden "
        "variable theory can reproduce."
    ),
}


def build_stack() -> RagStack:
    stack = RagStack(max_chars=400, overlap=60)
    for name, text in DOCS.items():
        stack.add_document(name, text)
    stack.build()
    return stack


def test_chunk_counts_and_stale_invalidation():
    stack = RagStack(max_chars=400, overlap=60)
    long = (DOCS["pos.md"] + "\n\n") * 6  # ~1.8k chars -> multiple chunks
    n = stack.add_document("pos.md", long)
    assert n >= 2
    stack.build()
    assert stack.n_chunks == n
    stack.add_document("kitchen.md", DOCS["kitchen.md"])
    with pytest.raises(RuntimeError):
        stack.retrieve("anything")


def test_retrieve_finds_the_right_document():
    stack = build_stack()
    hits = stack.retrieve("how do cooks see new tickets?", k=3)
    assert hits
    assert stack.chunk_text(hits[0].position) in DOCS["kitchen.md"] or (
        stack._meta[hits[0].position].document == "kitchen.md"
    )


def test_answer_is_deterministic_and_cites_real_chunks():
    stack = build_stack()
    a1 = stack.answer("how do cooks see new tickets?", backend="extractive")
    a2 = stack.answer("how do cooks see new tickets?", backend="extractive")
    assert a1.text == a2.text
    assert a1.citations, "expected at least one citation"
    valid = {(m.chunk_index, m.document) for m in stack._meta}
    for c in a1.citations:
        assert (c.chunk_index, c.document) in valid
    assert "[1]" in a1.text


def test_answer_without_documents():
    stack = RagStack()
    with pytest.raises(RuntimeError):
        stack.answer("hello", backend="extractive")


def test_compose_extractive_empty_retrieval():
    ans = compose_extractive("anything", [])
    assert "don't have any documents" in ans.text
    assert ans.citations == []


def test_unrelated_query_stays_low_scored():
    stack = build_stack()
    hits = stack.retrieve("bell inequalities hidden variables", k=3)
    assert stack._meta[hits[0].position].document == "physics.md"


def test_same_chunk_number_different_documents_keeps_source():
    result = compose_extractive('apples oranges', [
        (0, 'apple.md', 'Apples grow on trees.'),
        (0, 'orange.md', 'Oranges grow in groves.'),
    ])
    assert {c.document for c in result.citations} == {'apple.md', 'orange.md'}
