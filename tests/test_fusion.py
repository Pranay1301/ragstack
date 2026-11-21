"""Tests for reciprocal rank fusion."""

import pytest

from ragstack.fusion import rrf


def test_document_in_both_rankings_wins():
    a = [(1, 0.9), (2, 0.8), (3, 0.1)]
    b = [(1, 5.0), (4, 4.0), (2, 3.0)]
    fused = rrf([a, b])
    assert fused[0][0] == 1


def test_rank_positions_not_raw_scores_decide():
    # b's scores are on a wildly different scale; fusion must ignore them
    a = [(1, 0.51), (2, 0.50)]
    b = [(2, 1e9), (1, 1e-9)]
    fused = rrf([a, b])
    # 1 and 2 each get one first and one second place -> tie broken by position
    assert {p for p, _ in fused[:2]} == {1, 2}


def test_fused_scores_decrease_with_rank():
    fused = rrf([[(5, 1.0), (6, 0.9), (7, 0.8)]])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_empty_rankings():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        rrf([[(1, 1.0)]], k=0)
