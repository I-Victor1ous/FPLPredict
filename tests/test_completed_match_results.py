import pandas as pd

from fplpredict.predictions_util import (
    backfill_actuals,
    completed_match_results,
)


def test_completed_match_results_backfills_unscored_predictions():
    matches = pd.DataFrame(
        [
            {
                "Comp": "La Liga",
                "Date": "2026-10-09",
                "Team": "Espanyol",
                "Opponent": "Malaga",
                "Result": "D",
                "GF": 1,
                "GA": 1,
            },
            {
                "Comp": "La Liga",
                "Date": "2026-10-09",
                "Team": "Malaga",
                "Opponent": "Espanyol",
                "Result": "D",
                "GF": 1,
                "GA": 1,
            },
        ]
    )
    preds = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-09"),
                "Team": "Espanyol",
                "Opponent": "Malaga",
                "Pred": "Win",
                "Actual": None,
            }
        ]
    )
    completed = completed_match_results(matches, "La Liga")
    out = backfill_actuals(preds, completed)
    assert out.iloc[0]["Actual"] == "Draw"


def test_backfill_uses_reverse_fixture_row_when_only_opponent_perspective_in_csv():
    matches = pd.DataFrame(
        [
            {
                "Comp": "Serie A",
                "Date": "2026-10-11",
                "Team": "Juventus",
                "Opponent": "Cagliari",
                "Result": "W",
                "GF": 2,
                "GA": 0,
            },
        ]
    )
    preds = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-11"),
                "Team": "Cagliari",
                "Opponent": "Juventus",
                "Pred": "Loss",
                "Actual": None,
            }
        ]
    )
    completed = completed_match_results(matches, "Serie A")
    out = backfill_actuals(preds, completed)
    assert out.iloc[0]["Actual"] == "Loss"
