import pandas as pd

from fplpredict.export import _partition_predictions


def test_upcoming_excludes_scored_and_past():
    today = pd.Timestamp("2026-10-01")
    df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-10"),
                "Team": "A",
                "Opponent": "B",
                "Actual": None,
                "Team_win%": 40,
                "Draw%": 30,
                "Opp_win%": 30,
                "Pred": "Win",
            },
            {
                "Date": pd.Timestamp("2026-09-20"),
                "Team": "C",
                "Opponent": "D",
                "Actual": "Win",
                "Team_win%": 40,
                "Draw%": 30,
                "Opp_win%": 30,
                "Pred": "Win",
            },
            {
                "Date": pd.Timestamp("2026-09-25"),
                "Team": "E",
                "Opponent": "F",
                "Actual": None,
                "Team_win%": 40,
                "Draw%": 30,
                "Opp_win%": 30,
                "Pred": "Win",
            },
        ]
    )
    upcoming, awaiting, history = _partition_predictions(df, today)
    assert len(upcoming) == 1
    assert upcoming.iloc[0]["Team"] == "A"
    assert len(history) == 1
    assert history.iloc[0]["Team"] == "C"
    assert list(awaiting["Team"]) == ["E"]


def test_awaiting_result_drops_stale_unscored_fixtures():
    today = pd.Timestamp("2026-10-10")
    row = {"Team": "A", "Opponent": "B", "Actual": None, "Pred": "Win"}
    df = pd.DataFrame(
        [
            {**row, "Date": pd.Timestamp("2026-10-09")},
            {**row, "Date": pd.Timestamp("2026-09-01")},
        ]
    )
    _, awaiting, history = _partition_predictions(df, today)
    assert list(awaiting["Date"]) == [pd.Timestamp("2026-10-09")]
    assert history.empty


def test_dedupe_fixtures_compares_predicted_at_with_mixed_tz():
    from fplpredict.export import _dedupe_fixtures

    df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-10"),
                "Team": "Arsenal",
                "Opponent": "Chelsea",
                "Pred": "Win",
                "Team_win%": 40,
                "Draw%": 30,
                "Opp_win%": 30,
                "Predicted_at": "2026-10-01T10:00:00+00:00",
            },
            {
                "Date": pd.Timestamp("2026-10-10"),
                "Team": "Arsenal",
                "Opponent": "Chelsea",
                "Pred": "Draw",
                "Team_win%": 10,
                "Draw%": 80,
                "Opp_win%": 10,
                "Predicted_at": "2026-10-02T10:00:00",
            },
        ]
    )
    out = _dedupe_fixtures(df)
    assert len(out) == 1
    assert out.iloc[0]["Pred"] == "Win"
