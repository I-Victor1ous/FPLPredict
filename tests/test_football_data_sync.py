from pathlib import Path

import pandas as pd

from fplpredict.data import football_data as fd


def test_sync_always_refetches_current_season_file(monkeypatch, tmp_path):
    monkeypatch.setattr(fd, "ODDS_DIR", tmp_path)
    monkeypatch.setattr(fd, "current_season_end", lambda today=None: 2027)
    monkeypatch.setattr(fd, "sync_fixtures_csv", lambda: None)

    path = tmp_path / "2027_e0.csv"
    path.write_text("Div,Date,HxG\nE0,01/01/2020,1\n", encoding="utf-8")

    calls = {"n": 0}

    def fake_download(div: str, code: str) -> pd.DataFrame:
        calls["n"] += 1
        assert div == "E0"
        return pd.DataFrame(
            {
                "Div": ["E0"],
                "Date": ["01/01/2026"],
                "HxG": [1.2],
                "AxG": [0.8],
                "AvgH": [2.0],
                "AvgD": [3.0],
                "AvgA": [4.0],
            }
        )

    monkeypatch.setattr(fd, "_download_div_season", fake_download)

    saved = fd.sync_football_data_csvs(["Premier League"], season_end_years=[2026, 2027])

    assert calls["n"] == 1
    assert path in saved
    out = pd.read_csv(path)
    assert "2026" in str(out["Date"].iloc[0])


def test_sync_skips_cached_past_season_when_hxg_present(monkeypatch, tmp_path):
    monkeypatch.setattr(fd, "ODDS_DIR", tmp_path)
    monkeypatch.setattr(fd, "current_season_end", lambda today=None: 2027)
    monkeypatch.setattr(fd, "sync_fixtures_csv", lambda: None)

    path = tmp_path / "2026_e0.csv"
    path.write_text("Div,Date,HxG\nE0,01/01/2025,1\n", encoding="utf-8")

    def fake_download(_div: str, _code: str) -> pd.DataFrame:
        raise AssertionError("should not download past season")

    monkeypatch.setattr(fd, "_download_div_season", fake_download)

    saved = fd.sync_football_data_csvs(["Premier League"], season_end_years=[2026])

    assert saved == []
