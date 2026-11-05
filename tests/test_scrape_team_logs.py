"""Tests for FBref team log processing (no browser)."""

import pandas as pd

from fplpredict.scrape.team_logs import (
    build_team_league_rows,
    filter_league_fixtures,
    merge_optional_match_stats,
)


def _fixture_row(comp: str, date: str, team: str, opp: str) -> dict:
    return {
        "Date": date,
        "Comp": comp,
        "Team": team,
        "Opponent": opp,
        "Result": "W",
        "GF": 2,
        "GA": 0,
    }


def test_keeps_league_rows_when_shooting_and_misc_missing():
    past = pd.DataFrame(
        [
            _fixture_row("Premier League", "2026-09-01", "Arsenal", "Chelsea"),
            _fixture_row("FA Cup", "2026-09-08", "Arsenal", "Reading"),
        ]
    )
    out = build_team_league_rows(past, "Premier League", "Arsenal", None, None)
    assert len(out) == 1
    assert out.iloc[0]["Opponent"] == "Chelsea"


def test_merge_optional_does_not_drop_rows_on_bad_misc():
    past = pd.DataFrame([_fixture_row("Premier League", "2026-09-01", "Arsenal", "Chelsea")])
    shooting = pd.DataFrame({"Date": ["2026-09-01"], "Sh": [10], "SoT": [4], "PK": [0], "PKatt": [0]})
    misc = pd.DataFrame({"Date": ["2026-09-01"], "BadCol": [1]})
    merged = merge_optional_match_stats(past, shooting, misc)
    assert len(merged) == 1
    assert merged.iloc[0]["Sh"] == 10


def test_normalizes_manchester_utd_opponent():
    past = pd.DataFrame(
        [
            {
                **_fixture_row("Premier League", "2026-09-01", "Liverpool", "Manchester Utd"),
            }
        ]
    )
    out = filter_league_fixtures(past, "Premier League", team_name="Liverpool")
    assert out.iloc[0]["Opponent"] == "Manchester United"
