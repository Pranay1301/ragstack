import numpy as np

from ragstack.embeddings import TfidfEmbedder, tokenize

CORPUS = [
    "the restaurant kitchen prepares north indian curries and breads",
    "the point of sale terminal records every restaurant order",
    "quantum entanglement correlates distant particle measurements",
    "the kitchen display shows tickets to the restaurant cooks",
]


def test_tokenize_lowercase_alnum():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_rows_are_unit_norm():
    emb = TfidfEmbedder().fit(CORPUS)
    m = emb.transform(CORPUS)
    norms = np.linalg.norm(m, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_texts_score_higher_than_unrelated():
    emb = TfidfEmbedder().fit(CORPUS)
    m = emb.transform(CORPUS)
    sim_related = float(m[0] @ m[3])  # both about the restaurant kitchen
    sim_unrelated = float(m[0] @ m[2])  # kitchen vs quantum
    assert sim_related > sim_unrelated
    assert sim_related > 0.1


def test_query_transform_uses_fitted_vocab():
    emb = TfidfEmbedder().fit(CORPUS)
    q = emb.transform(["kitchen tickets for the cooks"])
    doc = emb.transform([CORPUS[3]])
    assert float(q[0] @ doc[0]) > 0.5


def test_out_of_vocab_text_is_zero_vector_not_nan():
    emb = TfidfEmbedder().fit(CORPUS)
    m = emb.transform(["zzz qqq jjj"])
    assert np.all(m == 0.0)
    assert not np.isnan(m).any()


def test_min_df_filters_rare_terms():
    emb = TfidfEmbedder(min_df=2).fit(CORPUS)
    # "quantum" appears in only one document, so it must be dropped
    assert "quantum" not in emb.vocab_
    assert "restaurant" in emb.vocab_


def test_fit_requires_nonempty_corpus():
    try:
        TfidfEmbedder().fit([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_transform_before_fit_raises():
    try:
        TfidfEmbedder().transform(["x"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
