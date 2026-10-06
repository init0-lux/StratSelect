from numpy.typing import NDArray

from src.ann_search import ann_post_filter
from src.cost_model import ann_estimated_cost, ef_search, exact_cost
from src.exact_search import exact_filter_first
from src.selectivity import selectivity_fraction


def adaptive_search(
    query_vector: NDArray,
    vectors: NDArray,
    candidate_mask: NDArray,
    k: int,
    selectivity_percent: float,
) -> tuple[list[int], str]:
    s = selectivity_fraction(selectivity_percent)
    n = vectors.shape[0]
    exact = exact_cost(n, s)
    ann = ann_estimated_cost(k, s)
    if exact <= ann:
        return exact_filter_first(query_vector, vectors, candidate_mask, k), "exact"
    effort = ef_search(k, s)
    return ann_post_filter(query_vector, vectors, candidate_mask, k, effort), f"ann-ef{effort + 1}"
