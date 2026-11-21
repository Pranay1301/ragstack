"""Tests for the abstention gate and retriever selection."""

import pytest

from ragstack.pipeline import RagStack

DOCS = {
    "pos.md": "The POS terminal accepts cards and cash. Taxes are applied at checkout.",
    "qr.md": "Guests scan a QR code to open the menu and order from the table.",
}


def _built() -> RagStack:
    stack = RagStack()
    for name, text in DOCS.items():
        stack.add_document(name, text)
    stack.build()
    return stack


def test_out_of_scope_question_abstains():
    answer = _built().answer("What is the capital of France?", backend="extractive")
    assert answer.abstained
    assert answer.backend == "abstain"
    assert answer.citations == []
    assert "couldn't find" in answer.text


def test_in_scope_question_does_not_abstain():
    answer = _built().answer("How do guests order from the table?", backend="extractive")
    assert not answer.abstained
    assert answer.citations


def test_min_confidence_zero_disables_abstention():
    answer = _built().answer(
        "What is the capital of France?", backend="extractive", min_confidence=0.0
    )
    assert not answer.abstained


def test_confidence_is_higher_for_in_scope():
    stack = _built()
    hits_in = stack.retrieve("guests order QR", retriever="hybrid")
    hits_out = stack.retrieve("capital of France", retriever="hybrid")
    assert stack.confidence("guests order QR", hits_in) > stack.confidence(
        "capital of France", hits_out
    )


@pytest.mark.parametrize("retriever", ["tfidf", "bm25", "hybrid"])
def test_all_retrievers_find_the_right_document(retriever):
    stack = _built()
    hits = stack.retrieve("scan code to order food", k=1, retriever=retriever)
    assert stack._meta[hits[0].position].document == "qr.md"


def test_unknown_retriever_rejected():
    with pytest.raises(ValueError):
        _built().retrieve("anything", retriever="neural")
