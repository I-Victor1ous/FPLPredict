import os
from datetime import date

LEAGUE = os.environ.get("FPL_LEAGUE", "Premier League")

# Modeling uses the last N seasons; train / val / test = 5 / 1 / 1 full seasons.
MODEL_DATA_SEASONS = int(os.environ.get("FPL_MODEL_DATA_SEASONS", "7"))
TRAIN_SEASON_COUNT = int(os.environ.get("FPL_TRAIN_SEASONS", "5"))
VAL_SEASON_COUNT = int(os.environ.get("FPL_VAL_SEASONS", "1"))
TEST_SEASON_COUNT = int(os.environ.get("FPL_TEST_SEASONS", "1"))

# Walk-forward folds: prior seasons used per test year (last season in window = val).
WF_TRAIN_WINDOW_SEASONS = int(os.environ.get("FPL_WF_TRAIN_SEASONS", "4"))
# Minimum completed rows in a season to use it as a walk-forward test fold.
WF_MIN_TEST_ROWS = int(os.environ.get("FPL_WF_MIN_TEST_ROWS", "100"))
# Strategies within this log-loss band of the best are treated as tied (accuracy wins).
WF_LOG_LOSS_TIE_EPS = float(os.environ.get("FPL_WF_LOG_LOSS_TIE_EPS", "0.005"))

# FBref scrape depth (season pages to walk back); defaults to modeling window.
YEAR_DIFF = int(os.environ.get("FPL_YEAR_DIFF", 7))

MODEL_RESELECT_DAYS = int(os.environ.get("FPL_MODEL_RESELECT_DAYS", "30"))

# European seasons roll over on 1 July (FBref season ids are end-years).
SEASON_ROLLOVER_MONTH = 7


def current_season_end(today: date | None = None) -> int:
    """FBref end-year of the season in progress (2027 = 2026-27)."""
    today = today or date.today()
    return today.year + 1 if today.month >= SEASON_ROLLOVER_MONTH else today.year


def model_season_window(today: date | None = None) -> list[int]:
    """End-years of the last ``MODEL_DATA_SEASONS`` seasons, oldest first."""
    end = current_season_end(today)
    return list(range(end - MODEL_DATA_SEASONS + 1, end + 1))

# Rolling form windows (league / H2H stats; ``last`` windows stay at 1 match).
ROLLING_FORM_WINDOW = int(os.environ.get("FPL_ROLLING_FORM_WINDOW", "5"))

LEAGUE_INFO_URLS = {
    "Premier League": "https://fbref.com/en/comps/9/Premier-League-Stats",
    "La Liga": "https://fbref.com/en/comps/12/La-Liga-Stats",
    "Bundesliga": "https://fbref.com/en/comps/20/Bundesliga-Stats",
    "Serie A": "https://fbref.com/en/comps/11/Serie-A-Stats",
    "Ligue 1": "https://fbref.com/en/comps/13/Ligue-1-Stats",
}


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

# Features available from FBref + football-data xG (dead Opta-only columns removed).
MODEL_FEATURES = [
    "is_home",
    "Relative_ratings",
    "Relative_pos",
    "Relative_rolling_rests_1",
    "Relative_xGleague_rolling",
    "Relative_xGAleague_rolling",
    "GFhth_rolling",
    "Relative_attack_strength",
    "Predicted_team_win",
    "Shlast_rolling",
    "SoTlast_rolling",
]

_ROLLING_STATS = ["GF", "GA", "Sh", "SoT", "CrdY", "CrdR", "xG", "xGA"]

# Pool for monthly forward selection: pre-match values only (no possession, IDs, or result).
CANDIDATE_FEATURES = list(
    dict.fromkeys(
        MODEL_FEATURES
        + [
            "is_home",
            "Team_position",
            "Opp_position",
            "Relative_pos",
            "Relative_ratings",
            "Relative_attack_strength",
            "Relative_defense_strength",
            "Expected_goal_team",
            "Expected_goal_opp",
            "Predicted_team_win",
            "Predicted_team_draw",
            "Predicted_team_loss",
            "rolling_rests_1",
            "rolling_rests_3",
            "Opp_rolling_rests_1",
            "Opp_rolling_rests_3",
            "Relative_rolling_rests_1",
            "Relative_rolling_rests_3",
            "leaguenum_wins_rolling",
            "Opp_leaguenum_wins_rolling",
            "Relative_leaguenum_wins_rolling",
            "GFhth_rolling",
            "GAhth_rolling",
            "hthnum_wins_rolling",
        ]
        + [f"{s}league_rolling" for s in _ROLLING_STATS]
        + [f"Opp_{s}league_rolling" for s in _ROLLING_STATS]
        + [f"Relative_{s}league_rolling" for s in _ROLLING_STATS]
        + [f"{s}last_rolling" for s in _ROLLING_STATS + ["Target"]]
        + [f"Opp_{s}last_rolling" for s in _ROLLING_STATS + ["Target"]]
        + [f"Relative_{s}last_rolling" for s in _ROLLING_STATS + ["Target"]]
    )
)
FEATURE_SELECTION_MAX = int(os.environ.get("FPL_FEATURE_SELECTION_MAX", "15"))
FEATURE_SELECTION_MIN_GAIN = float(os.environ.get("FPL_FEATURE_SELECTION_MIN_GAIN", "0.001"))

RESULT_LABELS = {0: "Draw", 1: "Loss", 2: "Win"}
