from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Dataset:
    vectors: NDArray[np.float64]
    categories: NDArray[np.str_]
    prices: NDArray[np.int64]


def generate_dataset(n: int = 50_000, dim: int = 64, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    return Dataset(
        vectors=rng.standard_normal((n, dim)),
        categories=rng.choice(np.array(["shoes", "books", "home", "tech"]), size=n),
        prices=rng.integers(100, 10_001, size=n),
    )
