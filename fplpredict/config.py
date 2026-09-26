import json
import os
from pathlib import Path

from fplpredict.paths import PROJECT_ROOT

LEAGUE = os.environ.get("FPL_LEAGUE", "Premier League")
SEASON_SPLIT = int(os.environ.get("FPL_SEASON_SPLIT", "2024"))
YEAR_DIFF = int(os.environ.get("FPL_YEAR_DIFF", "4"))
VAL_CUTOFF = os.environ.get("FPL_VAL_CUTOFF", "2024-03-14")
MODEL_RESELECT_DAYS = int(os.environ.get("FPL_MODEL_RESELECT_DAYS", "30"))

LEAGUE_INFO_URLS = {
    "Premier League": "https://fbref.com/en/comps/9/Premier-League-Stats",
    "La Liga": "https://fbref.com/en/comps/12/La-Liga-Stats",
}

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
