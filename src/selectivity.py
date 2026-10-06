import numpy as np
from numpy.typing import NDArray


def selectivity_fraction(percent: float) -> float:
    if not 0 < percent <= 100:
        raise ValueError("selectivity percent must be in (0, 100]")
    return percent / 100


def estimate_selectivity(candidate_mask: NDArray[np.bool_]) -> float:
    if candidate_mask.size == 0:
        raise ValueError("cannot estimate selectivity for an empty dataset")
    return float(np.count_nonzero(candidate_mask) / candidate_mask.size)
