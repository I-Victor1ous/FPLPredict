import pandas as pd

from fplpredict.predictions_util import propagate_fixture_actuals


def test_propagate_actuals_fills_reverse_perspective():
    df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-10"),
                "Team": "Tottenham Hotspur",
                "Opponent": "Manchester United",
                "Actual": "Draw",
                "Pred": "Win",
            },
            {
                "Date": pd.Timestamp("2026-10-10"),
                "Team": "Manchester United",
                "Opponent": "Tottenham Hotspur",
                "Actual": None,
                "Pred": "Loss",
            },
        ]
    )
    out = propagate_fixture_actuals(df)
    mu_row = out[out["Team"] == "Manchester United"].iloc[0]
    assert mu_row["Actual"] == "Draw"


def test_propagate_actuals_flips_win_loss_for_reverse_row():
    df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-10"),
                "Team": "Ipswich Town",
                "Opponent": "Fulham",
                "Actual": "Win",
                "Pred": "Win",
            },
            {
                "Date": pd.Timestamp("2026-10-10"),
                "Team": "Fulham",
                "Opponent": "Ipswich Town",
                "Actual": None,
                "Pred": "Win",
            },
        ]
    )
    out = propagate_fixture_actuals(df)
    fulham = out[out["Team"] == "Fulham"].iloc[0]
    assert fulham["Actual"] == "Loss"
