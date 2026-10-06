import numpy as np
from numpy.typing import NDArray


def ann_post_filter(
    query_vector: NDArray[np.float64],
    vectors: NDArray[np.float64],
    candidate_mask: NDArray[np.bool_],
    k: int,
    ef_search: int,
) -> list[int]:
    if ef_search < 1:
        raise ValueError("ef_search must be positive")
    distances = np.linalg.norm(vectors - query_vector, axis=1)
    candidate_count = min(vectors.shape[0], ef_search)
    ann_ids = np.lexsort((np.arange(vectors.shape[0]), distances))[:candidate_count]
    filtered = ann_ids[candidate_mask[ann_ids]]
    return filtered[:k].tolist()


def default_ann_post_filter(
    query_vector: NDArray[np.float64],
    vectors: NDArray[np.float64],
    candidate_mask: NDArray[np.bool_],
    k: int,
) -> list[int]:
    return ann_post_filter(query_vector, vectors, candidate_mask, k, ef_search=3 * k)
