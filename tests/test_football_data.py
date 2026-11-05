import pandas as pd

from fplpredict.data.football_data import (
    _fixture_lookup,
    _team_xg_from_fixture,
    enrich_team_match_xg,
)


def test_team_xg_from_home_perspective():
    pair = _team_xg_from_fixture("Arsenal", "Liverpool", 2.0, 0.5, "Arsenal", "Liverpool")
    assert pair == (2.0, 0.5)
    pair = _team_xg_from_fixture("Liverpool", "Arsenal", 2.0, 0.5, "Arsenal", "Liverpool")
    assert pair == (0.5, 2.0)


def test_enrich_team_match_xg_fills_missing():
    lookup = _fixture_lookup(
        pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-08-21"]),
                "HomeTeam": ["Arsenal"],
                "AwayTeam": ["Liverpool"],
                "HxG": [1.8],
                "AxG": [0.9],
            }
        )
    )
    matches = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-08-21"]),
            "Team": ["Arsenal"],
            "Opponent": ["Liverpool"],
            "xG": [pd.NA],
            "xGA": [pd.NA],
        }
    )
    out, filled = enrich_team_match_xg(matches, lookup)
    assert filled == 1
    assert float(out.iloc[0]["xG"]) == 1.8
    assert float(out.iloc[0]["xGA"]) == 0.9
