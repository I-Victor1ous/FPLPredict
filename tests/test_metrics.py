import numpy as np

from fplpredict.model.metrics import (
    accuracy_with_interval,
    brier_score,
    ranked_probability_score,
    reliability_bins,
    wilson_interval,
)


def test_rps_perfect_forecast():
    y = np.array([2, 0, 1])
    proba = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert ranked_probability_score(y, proba) == 0.0


def test_wilson_interval_bounds():
    low, high = wilson_interval(25, 49)
    assert 0.0 <= low < 0.51 < high <= 1.0


def test_accuracy_with_interval():
    stats = accuracy_with_interval(25, 49)
    assert stats["n"] == 49
    assert stats["ci95Low"] is not None


def test_brier_score_perfect_is_zero():
    y = np.array([0, 1, 2])
    proba = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert brier_score(y, proba) == 0.0


def test_reliability_bins_cover_hits():
    y = np.array([2, 2, 1, 0])
    proba = np.array(
        [
            [0.1, 0.1, 0.8],
            [0.2, 0.2, 0.6],
            [0.3, 0.5, 0.2],
            [0.7, 0.2, 0.1],
        ]
    )
    bins = reliability_bins(y, proba)
    assert bins and all("observed" in b and "meanPredicted" in b for b in bins)
