import numpy as np
from numpy.typing import NDArray


def euclidean_distance(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(left - right))
