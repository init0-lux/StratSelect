import numpy as np
from numpy.typing import NDArray


def exact_filter_first(
    query_vector: NDArray[np.float64],
    vectors: NDArray[np.float64],
    candidate_mask: NDArray[np.bool_],
    k: int,
) -> list[int]:
    if vectors.ndim != 2 or query_vector.shape != (vectors.shape[1],):
        raise ValueError("query and vector dimensions do not match")
    if candidate_mask.shape != (vectors.shape[0],):
        raise ValueError("candidate mask length must match vectors")
    if k < 1:
        raise ValueError("k must be positive")

    ids = np.flatnonzero(candidate_mask)
    if ids.size == 0:
        return []
    distances = np.linalg.norm(vectors[ids] - query_vector, axis=1)
    order = np.lexsort((ids, distances))[:k]
    return ids[order].tolist()
