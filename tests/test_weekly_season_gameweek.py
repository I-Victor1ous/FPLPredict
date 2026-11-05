import pandas as pd

from fplpredict.performance_export import weekly_performance_by_gameweek


def test_gameweek_buckets_are_per_season_not_round_label_only(monkeypatch):
    """Same 'Matchweek 6' in two seasons must not merge live-accuracy bars."""

    def fake_read(_name: str):
        return pd.DataFrame(
            [
                {
                    "Comp": "Premier League",
                    "Date": "2025-09-27",
                    "Team": "A",
                    "Opponent": "B",
                    "Round": "Matchweek 6",
                    "Season": 2026,
                },
                {
                    "Comp": "Premier League",
                    "Date": "2026-10-11",
                    "Team": "C",
                    "Opponent": "D",
                    "Round": "Matchweek 6",
                    "Season": 2027,
                },
            ]
        )

    monkeypatch.setattr("fplpredict.performance_export.read_csv", fake_read)
    df = pd.DataFrame(
        [
            {
                "Date": "2025-09-27",
                "Team": "A",
                "Opponent": "B",
                "Pred": "Win",
                "Actual": "Win",
            },
            {
                "Date": "2026-10-11",
                "Team": "C",
                "Opponent": "D",
                "Pred": "Loss",
                "Actual": "Win",
            },
        ]
    )
    weeks = weekly_performance_by_gameweek(df, "Premier League")
    assert len(weeks) == 2
    assert weeks[0]["seasonEnd"] == 2026
    assert weeks[1]["seasonEnd"] == 2027
    assert weeks[0]["n"] == 1
    assert weeks[1]["n"] == 1
