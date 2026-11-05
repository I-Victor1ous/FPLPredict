import numpy as np
import pandas as pd

from fplpredict.data.matches_clean import (
    clean_matches_for_modeling,
    keep_modeling_competitions,
    mirror_missing_perspectives,
    propagate_xg_within_fixtures,
)


def test_keep_modeling_competitions_drops_cups():
    df = pd.DataFrame(
        {
            "Comp": ["Premier League", "FA Cup"],
            "Team": ["Arsenal", "Arsenal"],
            "Opponent": ["Chelsea", "Liverpool"],
            "Date": ["2024-01-01", "2024-01-02"],
        }
    )
    out = keep_modeling_competitions(df)
    assert len(out) == 1
    assert out.iloc[0]["Comp"] == "Premier League"


def test_propagate_xg_within_fixtures():
    df = pd.DataFrame(
        {
            "Season": [2026, 2026],
            "Date": pd.to_datetime(["2026-08-21", "2026-08-21"]),
            "Team": ["Arsenal", "Liverpool"],
            "Opponent": ["Liverpool", "Arsenal"],
            "xG": [1.5, pd.NA],
            "xGA": [0.8, pd.NA],
        }
    )
    out, n = propagate_xg_within_fixtures(df)
    assert n >= 2
    assert float(out.iloc[1]["xG"]) == 0.8
    assert float(out.iloc[1]["xGA"]) == 1.5


def _row(team, opp, result, gf, ga, venue="Home", xg=np.nan, xga=np.nan, sh=10):
    return {
        "Comp": "Premier League",
        "Season": "2027",
        "Date": "2026-09-01",
        "Team": team,
        "Opponent": opp,
        "Result": result,
        "GF": gf,
        "GA": ga,
        "Venue": venue,
        "xG": xg,
        "xGA": xga,
        "Sh": sh,
    }


def test_mirror_missing_perspectives_swaps_fields():
    df = pd.DataFrame([_row("Coventry City", "Arsenal", "W", 2, 1, xg=1.4, xga=0.9)])
    out, n = mirror_missing_perspectives(df)
    assert n == 1
    m = out[out["Team"] == "Arsenal"].iloc[0]
    assert (m["Opponent"], m["Result"], m["GF"], m["GA"]) == ("Coventry City", "L", 1, 2)
    assert m["Venue"] == "Away"
    assert (m["xG"], m["xGA"]) == (0.9, 1.4)
    assert np.isnan(m["Sh"])


def test_clean_keeps_promoted_clubs_and_rows_without_xg():
    df = pd.DataFrame(
        [
            _row("Coventry City", "Arsenal", "W", 2, 1),
            _row("Arsenal", "Coventry City", "L", 1, 2, venue="Away"),
            _row("Hull City", "Manchester Utd", "D", 0, 0),
            _row("Hull City", "Manchester Utd", "D", 0, 0),
        ]
    )
    out = clean_matches_for_modeling(df)
    assert set(out["Team"]) == {"Coventry City", "Arsenal", "Hull City", "Manchester United"}
    assert len(out) == 4  # exact duplicate dropped, missing Man Utd perspective mirrored
    assert out["Season"].iloc[0] == 2027


def test_clean_keeps_relegation_tie_breaker_round():
    df = pd.DataFrame(
        [
            {
                "Comp": "Serie A",
                "Season": 2023,
                "Date": "2023-06-11",
                "Team": "Hellas Verona",
                "Opponent": "Spezia",
                "Round": "Relegation tie-breaker",
                "Result": "W",
                "GF": 3,
                "GA": 1,
            },
        ]
    )
    out = clean_matches_for_modeling(df)
    assert len(out) == 2  # mirror adds Spezia's perspective
    assert set(out["Round"]) == {"Relegation tie-breaker"}
