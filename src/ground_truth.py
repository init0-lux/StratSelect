from numpy.typing import NDArray

from src.exact_search import exact_filter_first


def ground_truth(
    query_vector: NDArray,
    vectors: NDArray,
    candidate_mask: NDArray,
    k: int,
) -> list[int]:
    return exact_filter_first(query_vector, vectors, candidate_mask, k)
