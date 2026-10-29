import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WEB_DATA_DIR = PROJECT_ROOT / "web" / "public" / "data"
LOG_DIR = PROJECT_ROOT / "logs"


def league_slug(league_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", league_name.lower()).strip("-")
    return slug or "league"


def team_slug(team_name: str) -> str:
    return league_slug(team_name)


def league_artifacts_dir(league_name: str) -> Path:
    return ARTIFACTS_DIR / league_slug(league_name)


def league_predictions_csv(league_name: str) -> Path:
    return OUTPUT_DIR / f"predictions-{league_slug(league_name)}.csv"


def league_prediction_log(league_name: str) -> Path:
    return OUTPUT_DIR / f"predictions-log-{league_slug(league_name)}.jsonl"


def league_web_json(league_name: str) -> Path:
    return WEB_DATA_DIR / f"{league_slug(league_name)}.json"


for directory in (DATA_DIR, OUTPUT_DIR, ARTIFACTS_DIR, WEB_DATA_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
