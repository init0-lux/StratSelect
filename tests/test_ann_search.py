import numpy as np

from src.ann_search import ann_post_filter


def test_ann_post_filter_is_deterministic_and_applies_filter_after_candidates():
    vectors = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    mask = np.array([False, False, True, True, True])
    query = np.array([0.0])

    first = ann_post_filter(query, vectors, mask, k=2, ef_search=3)
    second = ann_post_filter(query, vectors, mask, k=2, ef_search=3)

    assert first == [2]
    assert second == first
