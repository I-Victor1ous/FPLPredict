#!/usr/bin/env bash
# Backfill specific FBref seasons (default 2021–2022) for all leagues + Understat xG.
#
# Set a wide enough modeling window before running so prune keeps new seasons:
#   export FPL_MODEL_DATA_SEASONS=7 FPL_YEAR_DIFF=7
#
# Examples:
#   ./scripts/backfill_seasons.sh
#   ./scripts/backfill_seasons.sh --seasons 2021,2022
#   ./scripts/backfill_seasons.sh --league "Premier League"
#   ./scripts/backfill_seasons.sh --then-model-update   # backfill then force reselect + predict
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

THEN_MODEL_UPDATE=false
BACKFILL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --then-model-update) THEN_MODEL_UPDATE=true ;;
    *) BACKFILL_ARGS+=("$arg") ;;
  esac
done

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -d "$ROOT/venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Backfill seasons (FBref + xG). FPL_MODEL_DATA_SEASONS=${FPL_MODEL_DATA_SEASONS:-5}"
python -m fplpredict.scrape.fbref_backfill "${BACKFILL_ARGS[@]}"

echo ""
python scripts/check_scrape_coverage.py

if [[ "$THEN_MODEL_UPDATE" == true ]]; then
  echo ""
  echo "Running model reselect + predictions (--skip-scrape)…"
  ./scripts/weekly_update.sh --skip-scrape --force-model-reselect
fi
