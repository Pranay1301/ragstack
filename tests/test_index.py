import numpy as np

from ragstack.index import VectorIndex


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_add_and_search_orders_by_cosine():
    idx = VectorIndex(dim=3)
    idx.add(np.stack([unit([1, 0, 0]), unit([0.9, 0.1, 0]), unit([0, 1, 0])]))
    hits = idx.search(unit([1, 0, 0]), k=3)
    assert [h.position for h in hits] == [0, 1, 2]
    assert hits[0].score > hits[1].score > hits[2].score


def test_search_k_capped_and_empty_index():
    idx = VectorIndex(dim=2)
    assert idx.search(unit([1, 0]), k=5) == []
    idx.add(unit([1, 0]))
    hits = idx.search(unit([1, 0]), k=10)
    assert len(hits) == 1


def test_dim_mismatch_rejected():
    idx = VectorIndex(dim=3)
    try:
        idx.add(np.zeros((1, 4), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    idx.add(np.zeros((1, 3), dtype=np.float32))
    try:
        idx.search(np.zeros(2, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_save_load_roundtrip(tmp_path):
    idx = VectorIndex(dim=4)
    idx.add(np.stack([unit([1, 2, 3, 4]), unit([4, 3, 2, 1])]))
    p = tmp_path / "idx.npz"
    idx.save(str(p))
    loaded = VectorIndex.load(str(p))
    assert len(loaded) == 2
    assert loaded.dim == 4
    before = idx.search(unit([1, 2, 3, 4]), k=2)
    after = loaded.search(unit([1, 2, 3, 4]), k=2)
    assert [h.position for h in before] == [h.position for h in after]
    assert np.allclose([h.score for h in before], [h.score for h in after])


def test_one_dimensional_add():
    idx = VectorIndex(dim=2)
    idx.add(unit([3, 4]))  # single row, not stacked
    assert len(idx) == 1
