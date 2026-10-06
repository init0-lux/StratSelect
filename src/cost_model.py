import math

CALIBRATION_MULTIPLIER = 5


def _validate(n: int, k: int, s: float) -> None:
    if n <= 0 or k <= 0 or not 0 < s <= 1:
        raise ValueError("n, k must be positive and s must be in (0, 1]")


def exact_cost(n: int, s: float) -> float:
    _validate(n, 1, s)
    return n * s


def ef_search(k: int, s: float) -> int:
    _validate(1, k, s)
    return math.ceil(3 * k / s)


def ann_estimated_cost(k: int, s: float) -> float:
    return CALIBRATION_MULTIPLIER * ef_search(k, s)
