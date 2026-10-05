import numpy as np

from src.dataset import generate_dataset
from src.exact_search import exact_filter_first
from src.similarity import euclidean_distance


def test_distance_is_deterministic_euclidean_distance():
    assert euclidean_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_generated_dataset_is_repeatable_and_has_expected_shape():
    first = generate_dataset(n=12, dim=4, seed=7)
    second = generate_dataset(n=12, dim=4, seed=7)

    assert first.vectors.shape == (12, 4)
    assert np.array_equal(first.vectors, second.vectors)
    assert np.array_equal(first.categories, second.categories)
    assert np.array_equal(first.prices, second.prices)


def test_exact_filter_first_returns_nearest_matching_ids():
    vectors = np.array([[5.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    mask = np.array([True, True, False, True])

    assert exact_filter_first(np.array([0.0, 0.0]), vectors, mask, 2) == [3, 1]


def test_exact_filter_first_returns_empty_for_empty_filter():
    vectors = np.zeros((3, 2))

    assert exact_filter_first(np.zeros(2), vectors, np.zeros(3, dtype=bool), 10) == []
