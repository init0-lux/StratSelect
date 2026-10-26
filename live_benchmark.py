"""Recompute recall@k from the simulation instead of printing reference rows.

Every number below is computed on this run from the seeded dataset, so it
can be checked live. Work is reported in deterministic cost-model units
(vectors compared), not wall-clock time.
"""

import numpy as np

from src.adaptive_search import adaptive_search
from src.ann_search import ann_post_filter
from src.config import DIM, K, N, QUERIES_PER_POINT, SEED, SELECTIVITY_POINTS
from src.dataset import generate_dataset
from src.exact_search import exact_filter_first
from src.ground_truth import ground_truth

PGVECTOR_DEFAULT_EF_SEARCH = 40  # pgvector's default hnsw.ef_search


def recall(result: list[int], truth: list[int]) -> float:
    return len(set(result) & set(truth)) / len(truth) if truth else 1.0


def main() -> None:
    vectors = generate_dataset(N, DIM, SEED).vectors
    rng = np.random.default_rng(SEED + 1)
    print(f"# live run: N={N:,}  dim={DIM}  k={K}  {QUERIES_PER_POINT} queries/point  seed={SEED}")
    print("select%  | default ANN+post-filter (ef=40) | exact filter-first | ADAPTIVE (plan)")
    print("---------|--------------------------------|--------------------|----------------")
    for percent in SELECTIVITY_POINTS:
        totals = {"default": 0.0, "exact": 0.0, "adaptive": 0.0}
        plan = ""
        for _ in range(QUERIES_PER_POINT):
            query = rng.standard_normal(DIM)
            mask = np.zeros(N, dtype=bool)
            mask[rng.choice(N, size=max(1, round(N * percent / 100)), replace=False)] = True
            truth = ground_truth(query, vectors, mask, K)
            totals["default"] += recall(ann_post_filter(query, vectors, mask, K, PGVECTOR_DEFAULT_EF_SEARCH), truth)
            totals["exact"] += recall(exact_filter_first(query, vectors, mask, K), truth)
            result, plan = adaptive_search(query, vectors, mask, K, percent)
            totals["adaptive"] += recall(result, truth)
        r = {name: total / QUERIES_PER_POINT for name, total in totals.items()}
        exact_work = max(1, round(N * percent / 100))
        print(
            f"{percent:7.1f}  | R={r['default']:.3f}  work={PGVECTOR_DEFAULT_EF_SEARCH:<6}          "
            f"| R={r['exact']:.3f} work={exact_work:<6}| R={r['adaptive']:.3f} ({plan})"
        )


if __name__ == "__main__":
    main()
