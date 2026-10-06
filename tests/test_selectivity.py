import numpy as np
import pytest

from src.selectivity import estimate_selectivity, selectivity_fraction


@pytest.mark.parametrize("percent", [0.1, 0.5, 1.0, 5.0, 20.0, 50.0, 100.0])
def test_selectivity_points_convert_to_fractions(percent):
    assert selectivity_fraction(percent) == percent / 100


def test_estimate_selectivity_uses_fraction_of_matching_rows():
    assert estimate_selectivity(np.array([True, False, True, False])) == 0.5


@pytest.mark.parametrize("percent", [0, -1, 100.1])
def test_selectivity_percent_rejects_values_outside_supported_range(percent):
    with pytest.raises(ValueError):
        selectivity_fraction(percent)
