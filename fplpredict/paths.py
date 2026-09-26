from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WEB_DATA_DIR = PROJECT_ROOT / "web" / "public" / "data"
LOG_DIR = PROJECT_ROOT / "logs"

MATCHES_CSV = DATA_DIR / "matches.csv"
HISTORY_CSV = DATA_DIR / "history.csv"
NEXT_MATCHES_CSV = DATA_DIR / "next_matches.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
PREDICTIONS_JSON = OUTPUT_DIR / "predictions.json"

for directory in (DATA_DIR, OUTPUT_DIR, ARTIFACTS_DIR, WEB_DATA_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
