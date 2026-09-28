from pathlib import Path

import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WEB_DATA_DIR = PROJECT_ROOT / "web" / "public" / "data"
LOG_DIR = PROJECT_ROOT / "logs"

MATCHES_CSV = DATA_DIR / "matches.csv"
HISTORY_CSV = DATA_DIR / "history.csv"
NEXT_MATCHES_CSV = DATA_DIR / "next_matches.csv"


def _league_slug(league_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", league_name.lower()).strip("-")
    return slug or "league"


def league_artifacts_dir(league_name: str) -> Path:
    return ARTIFACTS_DIR / _league_slug(league_name)


def league_predictions_csv(league_name: str) -> Path:
    return OUTPUT_DIR / f"predictions-{_league_slug(league_name)}.csv"


def league_web_json(league_name: str) -> Path:
    return WEB_DATA_DIR / f"{_league_slug(league_name)}.json"


for directory in (DATA_DIR, OUTPUT_DIR, ARTIFACTS_DIR, WEB_DATA_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
