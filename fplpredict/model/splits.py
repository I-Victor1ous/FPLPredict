"""Season-window filtering and train / val / test splits for modeling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from fplpredict.config import (
    MODEL_DATA_SEASONS,
    TEST_SEASON_COUNT,
    TRAIN_SEASON_COUNT,
    VAL_SEASON_COUNT,
)


@dataclass(frozen=True)
class SeasonSplitMeta:
    """Which seasons were used for each split (FBref ``Season`` end-year ints)."""

    all_seasons: tuple[int, ...]
    train_seasons: tuple[int, ...]
    val_season: int
    test_season: int


def modeling_season_counts() -> tuple[int, int, int, int]:
    """Return (data_window, train, val, test) season counts."""
    train, val, test = TRAIN_SEASON_COUNT, VAL_SEASON_COUNT, TEST_SEASON_COUNT
    window = MODEL_DATA_SEASONS
    if train + val + test != window:
        raise ValueError(
            f"Train/val/test seasons ({train}+{val}+{test}) must equal "
            f"MODEL_DATA_SEASONS ({window})"
        )
    return window, train, val, test


def sorted_seasons(df: pd.DataFrame, column: str = "Season") -> list[int]:
    seasons = df[column].dropna().unique()
    return sorted(int(s) for s in seasons)


def restrict_last_n_seasons(
    df: pd.DataFrame, n: int = MODEL_DATA_SEASONS, column: str = "Season"
) -> pd.DataFrame:
    seasons = sorted_seasons(df, column)
    if len(seasons) < n:
        raise RuntimeError(
            f"Need at least {n} seasons for modeling; found {len(seasons)}: {seasons}"
        )
    keep = set(seasons[-n:])
    return df[df[column].isin(keep)].copy()


def completed_past_matches(
    past_matches: pd.DataFrame, as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Completed fixtures only (known result, strictly before today if ``as_of`` set)."""
    out = past_matches[past_matches["Target"].notna()].copy()
    if as_of is None:
        as_of = pd.Timestamp.now(tz=timezone.utc).normalize().tz_localize(None)
    else:
        as_of = pd.Timestamp(as_of).normalize()
    return out[out["Date"] < as_of].sort_values("Date")


def season_holdout_frames(
    past_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SeasonSplitMeta]:
    """
    Last ``MODEL_DATA_SEASONS`` seasons: earliest 3 train, next val, latest test.

    ``past_matches`` must already be completed-only rows with a ``Season`` column.
    """
    window, n_train, _, _ = modeling_season_counts()
    scoped = restrict_last_n_seasons(past_matches, window)
    seasons = tuple(sorted_seasons(scoped))
    train_seasons = seasons[:n_train]
    val_season = seasons[n_train]
    test_season = seasons[n_train + 1]
    meta = SeasonSplitMeta(
        all_seasons=seasons,
        train_seasons=train_seasons,
        val_season=val_season,
        test_season=test_season,
    )
    train_df = scoped[scoped["Season"].isin(train_seasons)].sort_values("Date")
    val_df = scoped[scoped["Season"] == val_season].sort_values("Date")
    test_df = scoped[scoped["Season"] == test_season].sort_values("Date")
    return train_df, val_df, test_df, meta


def prune_matches_by_season_window(
    matches: pd.DataFrame, n: int = MODEL_DATA_SEASONS
) -> pd.DataFrame:
    """Keep only the last ``n`` seasons per competition in scraped match data."""
    if matches.empty or "Comp" not in matches.columns:
        return matches
    parts = []
    for _, group in matches.groupby("Comp", dropna=False):
        seasons = sorted_seasons(group)
        if len(seasons) <= n:
            parts.append(group)
            continue
        keep = set(seasons[-n:])
        parts.append(group[group["Season"].isin(keep)])
    return pd.concat(parts, ignore_index=True)
