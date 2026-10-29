import pandas as pd

from fplpredict.model.walkforward_selection import (
    build_fold_split,
    walkforward_test_seasons,
)


def _synthetic_modeling() -> pd.DataFrame:
    rows = []
    for season in range(2021, 2028):
        for i in range(120):
            rows.append(
                {
                    "Season": season,
                    "Date": pd.Timestamp(f"{season - 1}-08-{i % 28 + 1:02d}"),
                    "Team": "A",
                    "Opponent": "B",
                    "Target": i % 3,
                    "f1": float(i),
                    "f2": float(season),
                }
            )
    return pd.DataFrame(rows)


def test_walkforward_test_seasons_with_four_season_window(monkeypatch):
    monkeypatch.setattr(
        "fplpredict.model.walkforward_selection.current_season_end",
        lambda today=None: 2028,
    )
    df = _synthetic_modeling()
    seasons = walkforward_test_seasons(df)
    assert 2025 in seasons
    assert 2026 in seasons
    assert 2027 in seasons
    assert 2024 not in seasons


def test_build_fold_split_uses_prior_window_only():
    df = _synthetic_modeling()
    split = build_fold_split(df, ["f1", "f2"], test_season=2027)
    assert split.split_meta.test_season == 2027
    assert split.split_meta.val_season == 2026
    assert 2027 not in split.split_meta.train_seasons
    assert len(split.test_y) == 120
