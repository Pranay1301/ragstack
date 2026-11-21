"""Tests for index persistence (save/load round-trip)."""

import pytest

from ragstack.pipeline import RagStack

DOCS = {
    "a.md": "Refunds are issued to the original payment method within 5 days.",
    "b.md": "The kitchen display shows orders in arrival order, oldest first.",
}


def _built() -> RagStack:
    stack = RagStack()
    for name, text in DOCS.items():
        stack.add_document(name, text)
    stack.build()
    return stack


def test_round_trip_preserves_retrieval(tmp_path):
    stack = _built()
    stack.save(str(tmp_path / "idx"))
    loaded = RagStack.load(str(tmp_path / "idx"))
    q = "how are refunds issued"
    before = [(h.position, round(h.score, 8)) for h in stack.retrieve(q, retriever="hybrid")]
    after = [(h.position, round(h.score, 8)) for h in loaded.retrieve(q, retriever="hybrid")]
    assert before == after
    assert loaded.answer(q, backend="extractive").text == stack.answer(q, backend="extractive").text


def test_round_trip_preserves_metadata(tmp_path):
    stack = _built()
    stack.save(str(tmp_path / "idx"))
    loaded = RagStack.load(str(tmp_path / "idx"))
    assert loaded.n_chunks == stack.n_chunks
    assert loaded.max_chars == stack.max_chars
    hits = loaded.retrieve("kitchen display order", retriever="tfidf")
    assert any("b.md" in loaded._meta[h.position].document for h in hits)


def test_save_before_build_raises(tmp_path):
    with pytest.raises(RuntimeError):
        RagStack().save(str(tmp_path / "idx"))


def test_load_rejects_wrong_version(tmp_path):
    stack = _built()
    stack.save(str(tmp_path / "idx"))
    import json

    payload = json.loads((tmp_path / "idx" / "stack.json").read_text())
    payload["version"] = 99
    (tmp_path / "idx" / "stack.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        RagStack.load(str(tmp_path / "idx"))


def test_cli_build_then_ask(tmp_path, capsys):
    import subprocess
    import sys

    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    idx = tmp_path / "idx"
    subprocess.run(
        [sys.executable, "-m", "ragstack.cli", "build", str(docs), str(idx)],
        check=True, capture_output=True,
    )
    out = subprocess.run(
        [sys.executable, "-m", "ragstack.cli", "ask", str(idx), "how are refunds issued"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "Q: how are refunds issued" in out
    assert "refund" in out.lower()
