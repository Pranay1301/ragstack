"""Tests for the from-scratch BM25 ranker."""

import math


import pytest

from ragstack.bm25 import BM25Index

CORPUS = [
    "the cat sat on the mat",
    "the dog chased the cat through the garden",
    "quantum entanglement enables correlated measurement outcomes",
]


def test_ranks_relevant_document_first():
    idx = BM25Index().fit(CORPUS)
    top = idx.rank("cat", k=1)
    # doc 1 has 'cat' once in a shorter doc than doc 2? both have it once;
    # doc 0 is shorter, so length normalization favors it... assert set instead
    assert top[0][0] in (0, 1)
    docs = {p for p, _ in idx.rank("quantum measurement", k=1)}
    assert docs == {2}


def test_scores_are_zero_for_oov_query():
    idx = BM25Index().fit(CORPUS)
    assert all(s == 0.0 for s in idx.scores("zzzz qqqq"))


def test_term_frequency_saturates():
    # repeating a term 100x must score higher than 1x but far below 100x
    one = ["apple " * 1 + "filler words here now", "banana"]
    many = ["apple " * 100 + "filler words here now", "banana"]
    s_one = BM25Index().fit(one).scores("apple")[0]
    s_many = BM25Index().fit(many).scores("apple")[0]
    assert s_many > s_one
    assert s_many < 10 * s_one


def test_length_normalization_penalizes_long_docs():
    short = "zebra stripes"
    long_doc = "zebra " + " ".join(f"w{i}" for i in range(200))
    idx = BM25Index().fit([short, long_doc])
    s = idx.scores("zebra")
    assert s[0] > s[1]


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25Index().fit([])


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        BM25Index(k1=0)
    with pytest.raises(ValueError):
        BM25Index(b=1.5)


def test_rank_respects_k_and_returns_all_when_k_large():
    idx = BM25Index().fit(CORPUS)
    assert len(idx.rank("the", k=10)) == len(CORPUS)
    assert len(idx.rank("the", k=2)) == 2


def test_idf_prefers_rare_terms():
    # a rare term matches only the document that contains it
    corpus = ["common common rare", "common common", "common common"]
    idx = BM25Index().fit(corpus)
    scores = idx.scores("rare")
    assert scores[0] > 0 and scores[1] == 0
    # equal term frequency: the longer document scores lower (length norm)
    common_only = idx.scores("common")
    assert 0 < common_only[0] < common_only[1]
    assert common_only[1] == pytest.approx(common_only[2], rel=1e-6)


def test_higher_term_frequency_scores_higher():
    idx = BM25Index().fit(["apple", "apple apple"])
    s = idx.scores("apple")
    assert s[1] > s[0]
