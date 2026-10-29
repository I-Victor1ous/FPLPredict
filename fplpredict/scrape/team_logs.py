"""Pure helpers for FBref team fixture logs (testable without Selenium)."""

from __future__ import annotations

import logging

import pandas as pd

from fplpredict.teams import normalize_team_name

log = logging.getLogger(__name__)

SHOOTING_MERGE_COLS = ["Date", "Sh", "SoT", "PK", "PKatt"]
MISC_MERGE_COLS = [
    "Date",
    "CrdY",
    "CrdR",
    "2CrdY",
    "Fls",
    "Fld",
    "Off",
    "Crs",
    "Int",
    "TklW",
]


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        out = frame.copy()
        out.columns = out.columns.droplevel()
        return out
    return frame


def merge_optional_match_stats(
    team_data: pd.DataFrame,
    shooting: pd.DataFrame | None,
    misc: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-merge shooting/misc onto fixture rows; never drops fixture rows on failure."""
    if team_data.empty:
        return team_data
    out = team_data.copy()
    if shooting is not None and not shooting.empty:
        sh = _flatten_columns(shooting)
        cols = [c for c in SHOOTING_MERGE_COLS if c in sh.columns]
        if len(cols) > 1:
            try:
                out = out.merge(sh[cols], on="Date", how="left")
            except (ValueError, KeyError) as exc:
                log.warning("Shooting merge skipped: %s", exc)
    if misc is not None and not misc.empty:
        mi = _flatten_columns(misc)
        cols = [c for c in MISC_MERGE_COLS if c in mi.columns]
        if len(cols) > 1:
            try:
                out = out.merge(mi[cols], on="Date", how="left")
            except (ValueError, KeyError) as exc:
                log.warning("Misc merge skipped: %s", exc)
    return out


def filter_league_fixtures(
    matches: pd.DataFrame,
    league: str,
    *,
    team_name: str | None = None,
) -> pd.DataFrame:
    """Keep completed fixture rows for ``league`` and normalize team names."""
    if matches.empty:
        return matches
    out = matches[matches["Date"] != "Date"].copy()
    if "Comp" in out.columns:
        out = out[out["Comp"] == league]
    if out.empty:
        return out
    if team_name is not None:
        canonical = normalize_team_name(team_name) or team_name
        out["Team"] = canonical
    elif "Team" in out.columns:
        out["Team"] = out["Team"].map(lambda x: normalize_team_name(x) or x)
    if "Opponent" in out.columns:
        out["Opponent"] = out["Opponent"].map(lambda x: normalize_team_name(x) or x)
    return out


def build_team_league_rows(
    past_matches: pd.DataFrame,
    league: str,
    team_name: str,
    shooting: pd.DataFrame | None = None,
    misc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fixture rows for one club in one league (stats optional)."""
    base = past_matches[past_matches["Result"].notna()].copy()
    if base.empty:
        return base
    merged = merge_optional_match_stats(base, shooting, misc)
    return filter_league_fixtures(merged, league, team_name=team_name)
