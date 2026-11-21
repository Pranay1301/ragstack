"""Tests for directory ingestion, including HTML stripping."""

import pytest

from ragstack.ingest import html_to_text, load_documents


def test_html_to_text_drops_markup_and_scripts():
    html = """
    <html><head><style>body { color: red; }</style></head>
    <body><h1>Menu</h1><p>Prices include <b>tax</b>.</p>
    <script>alert('not text')</script></body></html>
    """
    text = html_to_text(html)
    assert "Menu" in text
    assert "Prices include tax." in text
    assert "alert" not in text
    assert "color: red" not in text
    assert "<" not in text


def test_html_block_tags_become_line_breaks():
    text = html_to_text("<p>first</p><p>second</p>")
    assert text.splitlines() == ["first", "second"]


def test_load_documents_reads_md_txt_html(tmp_path):
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    (tmp_path / "c.html").write_text("<p>gamma</p>")
    (tmp_path / "d.py").write_text("not ingested")
    docs = dict(load_documents(str(tmp_path)))
    assert docs == {"a.md": "alpha", "b.txt": "beta", "c.html": "gamma"}


def test_load_documents_empty_dir_raises(tmp_path):
    with pytest.raises(SystemExit):
        load_documents(str(tmp_path))


def test_nested_directories_keep_relative_names(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "note.md").write_text("nested")
    (tmp_path / "top.md").write_text("top")
    names = {name for name, _ in load_documents(str(tmp_path))}
    assert names == {"top.md", "sub/note.md"}
