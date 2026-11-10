from ragstack.chunking import split_text


def test_empty_and_blank():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_short_text_single_chunk():
    chunks = split_text("hello world", max_chars=800)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].index == 0
    assert chunks[0].start == 0


def test_respects_max_chars_and_prefers_boundaries():
    sentences = ["Sentence number %d ends here." % i for i in range(40)]
    text = " ".join(sentences)
    chunks = split_text(text, max_chars=120, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 120
        assert c.text.endswith(".")  # broke on a sentence boundary


def test_overlap_carries_context():
    text = " ".join("word%03d" % i for i in range(200))  # 8 chars each
    chunks = split_text(text, max_chars=100, overlap=24)
    assert len(chunks) > 2
    # each chunk head repeats tail words of the previous chunk
    prev_tail = chunks[0].text.split()[-2:]
    assert chunks[1].text.split()[:2] == prev_tail


def test_oversized_unit_hard_splits():
    text = "x" * 1000  # no boundaries at all
    chunks = split_text(text, max_chars=300, overlap=0)
    assert all(len(c.text) <= 300 for c in chunks)
    assert sum(len(c.text) for c in chunks) == 1000


def test_invalid_params():
    for bad in (0, -5):
        try:
            split_text("abc", max_chars=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    try:
        split_text("abc", max_chars=10, overlap=10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for overlap >= max_chars")


def test_indices_and_offsets_monotonic():
    text = "alpha beta gamma. delta epsilon zeta. " * 20
    chunks = split_text(text, max_chars=80, overlap=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
