import pandas as pd

from fplpredict.model.splits import (
    completed_past_matches,
    restrict_last_n_seasons,
    season_holdout_frames,
)


def _sample_past() -> pd.DataFrame:
    rows = []
    for season in (2022, 2023, 2024, 2025, 2026):
        for day in (1, 15):
            rows.append(
                {
                    "Date": pd.Timestamp(f"{season - 1}-08-{day:02d}"),
                    "Season": season,
                    "Target": 2,
                    "Team": "A",
                    "Opponent": "B",
                }
            )
    return pd.DataFrame(rows)


def test_restrict_last_five_seasons():
    df = _sample_past()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "Date": pd.Timestamp("2019-01-01"),
                        "Season": 2019,
                        "Target": 0,
                        "Team": "A",
                        "Opponent": "B",
                    }
                ]
            ),
        ]
    )
    trimmed = restrict_last_n_seasons(df, 5)
    assert set(trimmed["Season"].unique()) == {2022, 2023, 2024, 2025, 2026}


def test_season_holdout_three_one_one():
    df = completed_past_matches(_sample_past())
    train, val, test, meta = season_holdout_frames(df)
    assert meta.train_seasons == (2022, 2023, 2024)
    assert meta.val_season == 2025
    assert meta.test_season == 2026
    assert set(train["Season"].unique()) == {2022, 2023, 2024}
    assert set(val["Season"].unique()) == {2025}
    assert set(test["Season"].unique()) == {2026}


def test_completed_excludes_today_and_future():
    df = _sample_past()
    today = pd.Timestamp("2026-02-01")
    df.loc[df["Date"] == pd.Timestamp("2025-08-15"), "Date"] = today
    out = completed_past_matches(df, as_of=today)
    assert (out["Date"] < today).all()
