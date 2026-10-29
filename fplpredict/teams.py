"""Canonical team names across leagues (FBref / historical spellings)."""

from __future__ import annotations

import pandas as pd

# Alternate spellings → one canonical name per club (standings + predictions).
TEAM_ALIASES: dict[str, str] = {
    # Premier League
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
    "Sheffield Utd": "Sheffield United",
    "West Brom": "West Bromwich Albion",
    # Understat short names (EPL)
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Norwich": "Norwich City",
    # La Liga (accent / spelling splits in matches.csv)
    "Alavés": "Alaves",
    "Atlético Madrid": "Atletico Madrid",
    "Málaga": "Malaga",
    "Leganés": "Leganes",
    "Almería": "Almeria",
    "Cádiz": "Cadiz",
    "Dep. A Coruña": "Dep. La Coruña",
    "Deportivo A Coruna": "Dep. La Coruña",
    "Racing Sant": "Racing Santander",
    "Real Valladolid": "Valladolid",
    "Real Oviedo": "Oviedo",
    # Bundesliga (FBref short names ↔ external xG sources)
    "Koln": "Köln",
    "FC Cologne": "Köln",
    "Cologne": "Köln",
    "Monchengladbach": "Gladbach",
    "Mönchengladbach": "Gladbach",
    "Borussia M'gladbach": "Gladbach",
    "Borussia M.Gladbach": "Gladbach",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "RasenBallsport Leipzig": "RB Leipzig",
    "FC Heidenheim": "Heidenheim",
    "VfB Stuttgart": "Stuttgart",
    "St. Pauli": "St Pauli",
    "Frankfurt": "Eintracht Frankfurt",
    "Hertha Berlin": "Hertha BSC",
    "Mainz": "Mainz 05",
    "Arminia": "Arminia Bielefeld",
    "Greuther Furth": "Greuther Fürth",
    "Greuther Fuerth": "Greuther Fürth",
    "Darmstadt": "Darmstadt 98",
    # Understat / football-data La Liga
    "SD Huesca": "Huesca",
    # Serie A
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "Verona": "Hellas Verona",
    "Parma Calcio 1913": "Parma",
    "SPAL": "Spal",
    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Saint-Etienne": "Saint-Étienne",
    "Saint Etienne": "Saint-Étienne",
    "Nimes": "Nîmes",
}


def normalize_team_name(name) -> str | None:
    if name is None or pd.isna(name):
        return None
    text = str(name).strip()
    if not text:
        return None
    canonical = TEAM_ALIASES.get(text, text)
    # One hop for alias chains (rare).
    return TEAM_ALIASES.get(canonical, canonical)
