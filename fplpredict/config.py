import json
import os
from pathlib import Path

from fplpredict.paths import PROJECT_ROOT

LEAGUE = os.environ.get("FPL_LEAGUE", "Premier League")

# Modeling uses the last N seasons; train / val / test = 3 / 1 / 1 full seasons.
MODEL_DATA_SEASONS = int(os.environ.get("FPL_MODEL_DATA_SEASONS", "5"))
TRAIN_SEASON_COUNT = int(os.environ.get("FPL_TRAIN_SEASONS", "3"))
VAL_SEASON_COUNT = int(os.environ.get("FPL_VAL_SEASONS", "1"))
TEST_SEASON_COUNT = int(os.environ.get("FPL_TEST_SEASONS", "1"))

# FBref scrape depth (season pages to walk back); defaults to modeling window.
YEAR_DIFF = int(os.environ.get("FPL_YEAR_DIFF", str(MODEL_DATA_SEASONS)))

MODEL_RESELECT_DAYS = int(os.environ.get("FPL_MODEL_RESELECT_DAYS", "30"))

LEAGUE_INFO_URLS = {
    "Premier League": "https://fbref.com/en/comps/9/Premier-League-Stats",
    "La Liga": "https://fbref.com/en/comps/12/La-Liga-Stats",
    "Bundesliga": "https://fbref.com/en/comps/20/Bundesliga-Stats",
    "Serie A": "https://fbref.com/en/comps/11/Serie-A-Stats",
    "Ligue 1": "https://fbref.com/en/comps/13/Ligue-1-Stats",
}


def league_slug(league_name: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", league_name.lower()).strip("-")
    return slug or "league"


def active_leagues() -> list[str]:
    """Leagues to scrape, train, and export (comma-separated ``FPL_LEAGUES`` or all)."""
    raw = os.environ.get("FPL_LEAGUES")
    if raw:
        names = [part.strip() for part in raw.split(",") if part.strip()]
        unknown = [n for n in names if n not in LEAGUE_INFO_URLS]
        if unknown:
            raise ValueError(
                f"Unknown league(s) in FPL_LEAGUES: {unknown}. "
                f"Known: {list(LEAGUE_INFO_URLS)}"
            )
        return names
    return list(LEAGUE_INFO_URLS.keys())

# Selected via notebook permutation importance (top 13).
MODEL_FEATURES = [
    "Relative_ratings",
    "TklWlast_rolling",
    "xGhth_rolling",
    "Relative_pos",
    "Opp_Distleague_rolling",
    "GFhth_rolling",
    "Opp_Fldlast_rolling",
    "Opp_rolling_rests_1",
    "xGAleague_rolling",
    "Relative_Fldlast_rolling",
    "Relative_Intleague_rolling",
    "Relative_GCAleague_rolling",
    "Relative_Intlast_rolling",
]

RESULT_LABELS = {0: "Draw", 1: "Loss", 2: "Win"}


def load_api_key() -> str | None:
    env_key = os.environ.get("FBRAPI_KEY")
    if env_key:
        return env_key
    config_path = PROJECT_ROOT / "config.json"
    if not config_path.exists():
        return None
    with config_path.open() as handle:
        data = json.load(handle)
    return data.get("API_key")
