import numpy as np

from src.adaptive_search import adaptive_search


def test_adaptive_plan_labels_match_required_points():
    vectors = np.arange(50_000, dtype=float).reshape(50_000, 1)
    query = np.array([0.0])
    plans = []
    for percent in [0.1, 0.5, 1.0, 5.0, 20.0, 50.0, 100.0]:
        mask = np.zeros(50_000, dtype=bool)
        mask[:max(1, round(percent))] = True
        _, plan = adaptive_search(query, vectors, mask, 10, percent)
        plans.append(plan)

    assert plans == ["exact", "exact", "exact", "exact", "ann-ef151", "ann-ef61", "ann-ef31"]


def test_adaptive_exact_execution_has_perfect_recall_against_ground_truth():
    vectors = np.array([[0.0], [2.0], [1.0]])
    result, plan = adaptive_search(np.array([0.0]), vectors, np.ones(3, dtype=bool), 2, 1.0)

    assert plan == "exact"
    assert result == [0, 2]
