import numpy as np
import pandas as pd
import pytest

from fplpredict.model.market import implied_probabilities, market_baseline


def test_margin_removal_sums_to_one_and_keeps_order():
    probs = implied_probabilities(np.array([2.0, 1.5]), np.array([3.4, 4.0]), np.array([3.8, 6.5]))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] > probs[0, 2] > 0
    # Evens on both sides of a two-way market would be 0.5 each after removing margin.
    two_way = implied_probabilities(np.array([1.9]), np.array([1e9]), np.array([1.9]))
    assert two_way[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_market_baseline_orients_odds_to_team_perspective():
    # Home side heavily favoured; one row seen from the home team, one from the away team.
    lookup = {("2026-08-22", "Arsenal", "Chelsea"): np.array([0.7, 0.2, 0.1])}
    test_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-08-22", "2026-08-23"]),
            "Team": ["Arsenal", "Chelsea"],
            "Opponent": ["Chelsea", "Arsenal"],
            "is_home": [1, 0],
            "Target": [2, 1],  # Arsenal win from each side: Win, then Loss
            "Season": [2027, 2027],
        }
    )
    model_proba = np.array([[0.2, 0.2, 0.6], [0.2, 0.6, 0.2]])
    result = market_baseline(test_df, model_proba, "Premier League", lookup=lookup)
    assert result["test_n"] == 2
    assert result["n"] == 2
    assert result["accuracy"] == 1.0
    assert result["model_accuracy"] == 1.0
    assert result["log_loss"] < result["model_log_loss"]


def test_market_baseline_without_matching_odds_is_none():
    test_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-08-22"]),
            "Team": ["Arsenal"],
            "Opponent": ["Chelsea"],
            "is_home": [1],
            "Target": [2],
            "Season": [2027],
        }
    )
    lookup = {("2025-01-01", "Arsenal", "Chelsea"): np.array([0.5, 0.3, 0.2])}
    assert market_baseline(test_df, np.array([[0.2, 0.2, 0.6]]), "Premier League", lookup=lookup) is None
