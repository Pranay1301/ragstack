import numpy as np

from ragstack.index import VectorIndex
from ragstack.retrieve import mmr, top_k


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_index():
    # two near-duplicates about cats, one distinct doc about physics
    idx = VectorIndex(dim=4)
    idx.add(
        np.stack(
            [
                unit([1, 0, 0, 0]),      # cat doc A
                unit([0.99, 0.01, 0, 0]),  # cat doc B (near-dup of A)
                unit([0, 0, 1, 0]),      # physics doc
            ]
        )
    )
    return idx


def test_top_k_returns_in_relevance_order():
    idx = make_index()
    hits = top_k(idx, unit([1, 0, 0, 0]), k=2)
    assert [h.position for h in hits] == [0, 1]


def test_mmr_prefers_diversity_over_second_duplicate():
    idx = make_index()
    hits = mmr(idx, unit([1, 0, 0, 0]), k=2, lambda_rel=0.3)
    positions = {h.position for h in hits}
    assert 0 in positions  # best match always first
    assert 2 in positions  # diverse physics doc beats the near-duplicate


def test_mmr_lambda_one_equals_pure_relevance():
    idx = make_index()
    hits = mmr(idx, unit([1, 0, 0, 0]), k=2, lambda_rel=1.0)
    assert [h.position for h in hits] == [0, 1]


def test_mmr_rejects_bad_lambda():
    idx = make_index()
    for bad in (-0.1, 1.1):
        try:
            mmr(idx, unit([1, 0, 0, 0]), k=2, lambda_rel=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_mmr_empty_index():
    idx = VectorIndex(dim=4)
    assert mmr(idx, unit([1, 0, 0, 0]), k=3) == []
