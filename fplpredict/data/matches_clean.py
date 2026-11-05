"""Keep matches.csv scoped to modeling leagues without deleting real fixtures."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fplpredict.config import LEAGUE_INFO_URLS
from fplpredict.teams import normalize_team_name

log = logging.getLogger(__name__)

MODELING_COMPS = frozenset(LEAGUE_INFO_URLS.keys())

LEAGUE_TEAM_SLOTS: dict[str, int] = {
    "Premier League": 20,
    "La Liga": 20,
    "Bundesliga": 18,
    "Serie A": 20,
    "Ligue 1": 18,
}
# Ligue 1 had 20 clubs until it shrank to 18 for 2023-24.
_SEASON_TEAM_SLOTS: dict[tuple[str, int], int] = {
    ("Ligue 1", 2021): 20,
    ("Ligue 1", 2022): 20,
    ("Ligue 1", 2023): 20,
}


def expected_club_count(comp: str, season) -> int:
    return _SEASON_TEAM_SLOTS.get((comp, int(season)), LEAGUE_TEAM_SLOTS.get(comp, 20))

ROW_KEY = ["Date", "Team", "Opponent"]
# Rows synthesized from the opponent's log; a real scraped row replaces them.
MIRRORED_COL = "Mirrored"

_MIRROR_SWAPS = [("GF", "GA"), ("xG", "xGA")]
_RESULT_FLIP = {"W": "L", "L": "W", "D": "D"}
_VENUE_FLIP = {"Home": "Away", "Away": "Home"}
# Stats that describe the focal club only; unknown for a mirrored row.
_FOCAL_ONLY_COLS = [
    "Poss",
    "Captain",
    "Formation",
    "Opp Formation",
    "Sh",
    "SoT",
    "PK",
    "PKatt",
    "CrdY",
    "CrdR",
    "2CrdY",
    "Fls",
    "Fld",
    "Off",
    "Crs",
    "Int",
    "TklW",
    "Dist",
    "FK",
    "Won%",
    "SCA",
    "GCA",
]


def keep_modeling_competitions(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or "Comp" not in matches.columns:
        return matches
    out = matches[matches["Comp"].isin(MODELING_COMPS)].copy()
    dropped = len(matches) - len(out)
    if dropped:
        log.info("Dropped %d non-league rows (cups/Europe).", dropped)
    return out


def normalize_season_column(matches: pd.DataFrame) -> pd.DataFrame:
    """FBref scrapes can mix string and int season labels."""
    if matches.empty or "Season" not in matches.columns:
        return matches
    out = matches.copy()
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce").astype("Int64")
    return out


def normalize_team_columns(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches
    out = matches.copy()
    for col in ("Team", "Opponent"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: normalize_team_name(x) or x)
    return out


def drop_exact_duplicates(matches: pd.DataFrame) -> pd.DataFrame:
    """Same club, same opponent, same date is one row; the first (newest) copy wins."""
    if matches.empty or not set(ROW_KEY) <= set(matches.columns):
        return matches
    out = matches.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    return out.drop_duplicates(subset=ROW_KEY, keep="first")


def mirror_missing_perspectives(matches: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add the opponent's row for completed fixtures where only one club's log was scraped."""
    if matches.empty or not set(ROW_KEY) <= set(matches.columns):
        return matches, 0
    completed = matches[matches["Result"].notna()] if "Result" in matches.columns else matches
    existing = set(zip(matches["Date"], matches["Team"], matches["Opponent"]))
    need = completed[
        [
            (d, o, t) not in existing
            for d, t, o in zip(completed["Date"], completed["Team"], completed["Opponent"])
        ]
    ]
    if need.empty:
        return matches, 0
    mirrored = need.copy()
    mirrored["Team"], mirrored["Opponent"] = need["Opponent"].values, need["Team"].values
    for a, b in _MIRROR_SWAPS:
        if a in mirrored.columns and b in mirrored.columns:
            mirrored[a], mirrored[b] = need[b].values, need[a].values
    if "Result" in mirrored.columns:
        mirrored["Result"] = need["Result"].map(lambda r: _RESULT_FLIP.get(r, r))
    if "Venue" in mirrored.columns:
        mirrored["Venue"] = need["Venue"].map(lambda v: _VENUE_FLIP.get(v, v))
    for col in _FOCAL_ONLY_COLS:
        if col in mirrored.columns:
            mirrored[col] = np.nan
    mirrored[MIRRORED_COL] = True
    mirrored = mirrored.drop_duplicates(subset=ROW_KEY)
    log.info("Mirrored %d fixture rows for clubs whose own log was missing.", len(mirrored))
    return pd.concat([matches, mirrored], ignore_index=True), len(mirrored)


def propagate_xg_within_fixtures(matches: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Copy xG/xGA from one club's row to the opponent's row of the same fixture."""
    if matches.empty or "xG" not in matches.columns or "xGA" not in matches.columns:
        return matches, 0
    out = matches.copy()
    lookup = {
        (d, t, o): (xg, xga)
        for d, t, o, xg, xga in zip(out["Date"], out["Team"], out["Opponent"], out["xG"], out["xGA"])
        if pd.notna(xg) and pd.notna(xga)
    }
    filled = 0
    for idx, (d, t, o) in zip(out.index, zip(out["Date"], out["Team"], out["Opponent"])):
        other = lookup.get((d, o, t))
        if other is None:
            continue
        if pd.isna(out.at[idx, "xG"]):
            out.at[idx, "xG"] = other[1]
            filled += 1
        if pd.isna(out.at[idx, "xGA"]):
            out.at[idx, "xGA"] = other[0]
            filled += 1
    if filled:
        log.info("Propagated %d xG/xGA values to opponent-perspective rows.", filled)
    return out, filled


def dedupe_next_match_fixtures(next_matches: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated upcoming rows for the same club and date (keeps both perspectives)."""
    if next_matches.empty or not set(ROW_KEY) <= set(next_matches.columns):
        return next_matches
    out = next_matches.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out.drop_duplicates(subset=ROW_KEY, keep="first")


def clean_matches_for_modeling(matches: pd.DataFrame) -> pd.DataFrame:
    matches = keep_modeling_competitions(matches)
    matches = normalize_season_column(matches)
    matches = normalize_team_columns(matches)
    matches = drop_exact_duplicates(matches)
    matches, _ = mirror_missing_perspectives(matches)
    return matches


def clean_next_matches(next_matches: pd.DataFrame) -> pd.DataFrame:
    next_matches = keep_modeling_competitions(next_matches)
    next_matches = normalize_season_column(next_matches)
    next_matches = normalize_team_columns(next_matches)
    return dedupe_next_match_fixtures(next_matches)
