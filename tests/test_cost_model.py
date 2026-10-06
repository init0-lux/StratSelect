import math

import pytest

from src.cost_model import ann_estimated_cost, exact_cost, ef_search


def test_cost_model_keeps_reference_formulas_and_calibration():
    assert exact_cost(50_000, 0.01) == 500
    assert ef_search(10, 0.01) == math.ceil(3 * 10 / 0.01) == 3000
    assert ann_estimated_cost(10, 0.01) == 15_000


@pytest.mark.parametrize("selectivity", [0, -0.1, 1.01])
def test_cost_model_rejects_invalid_selectivity(selectivity):
    with pytest.raises(ValueError):
        ef_search(10, selectivity)


def test_cost_model_rejects_nonpositive_k_or_n():
    with pytest.raises(ValueError):
        exact_cost(0, 0.5)
    with pytest.raises(ValueError):
        ef_search(0, 0.5)
