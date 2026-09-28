"""Canonical Premier League team names."""

from __future__ import annotations

import pandas as pd

# FBref / historical spellings → canonical name used in predictions and the site.
TEAM_ALIASES: dict[str, str] = {
    "Nott'ham Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Manchester Utd": "Manchester United",
    "Tottenham": "Tottenham Hotspur",
    "Brighton": "Brighton and Hove Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    # Bundesliga
    "Koln": "Köln",
    "Monchengladbach": "Gladbach",
    "Mönchengladbach": "Gladbach",
    "Borussia M'gladbach": "Gladbach",
    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
}


def normalize_team_name(name) -> str | None:
    if name is None:
        return None
    if isinstance(name, float) and pd.isna(name):
        return None
    if pd.isna(name):
        return None
    text = str(name).strip()
    if not text:
        return None
    return TEAM_ALIASES.get(text, text)


def normalize_team_column(series: pd.Series) -> pd.Series:
    return series.map(lambda x: normalize_team_name(x) if pd.notna(x) else x)
