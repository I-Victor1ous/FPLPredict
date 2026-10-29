import pandas as pd

from fplpredict.data.football_data import (
    backfill_next_match_results_from_football_data,
    completed_results_from_football_data,
)


def test_backfill_next_match_results_from_local_csv(tmp_path, monkeypatch):
    from fplpredict.data import football_data as fd

    odds = tmp_path / "odds"
    odds.mkdir()
    monkeypatch.setattr(fd, "ODDS_DIR", odds)
    csv = odds / "2027_e0.csv"
    csv.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HxG\n"
        "17/10/2026,Newcastle,Aston Villa,2,1,H,1.5\n",
        encoding="utf-8",
    )
    nextm = pd.DataFrame(
        [
            {
                "Date": "2026-10-17",
                "Comp": "Premier League",
                "Season": 2027,
                "Team": "Newcastle United",
                "Opponent": "Aston Villa",
                "Venue": "Home",
                "Result": None,
                "GF": None,
                "GA": None,
            }
        ]
    )
    out, n = backfill_next_match_results_from_football_data(
        nextm, leagues=["Premier League"], as_of=pd.Timestamp("2026-10-18")
    )
    assert n == 1
    assert out.iloc[0]["Result"] == "W"
    assert int(out.iloc[0]["GF"]) == 2


def test_completed_results_from_football_data(tmp_path, monkeypatch):
    from fplpredict.data import football_data as fd

    odds = tmp_path / "odds"
    odds.mkdir()
    monkeypatch.setattr(fd, "ODDS_DIR", odds)
    (odds / "2027_e0.csv").write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HxG\n"
        "17/10/2026,Newcastle,Aston Villa,2,1,H,1.5\n",
        encoding="utf-8",
    )
    matches = pd.DataFrame({"Comp": ["Premier League"], "Season": [2027]})
    res = completed_results_from_football_data(
        matches, "Premier League", as_of=pd.Timestamp("2026-10-18")
    )
    home = res[
        (res["Team"] == "Newcastle United") & (res["Opponent"] == "Aston Villa")
    ]
    assert len(home) == 1
    assert int(home.iloc[0]["Target"]) == 2
