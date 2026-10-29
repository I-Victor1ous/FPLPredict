#!/usr/bin/env bash
# Quick FBref scrape for one league (then coverage check).
# Examples:
#   ./scripts/scrape_one_league.sh
#   ./scripts/scrape_one_league.sh "Premier League"
#   ./scripts/scrape_one_league.sh "Premier League" --skip-post-process
#   ./scripts/scrape_one_league.sh "Premier League" --full   # re-walk every season
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEAGUE="${1:-Premier League}"
if [[ $# -gt 0 ]]; then
  shift
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -d "$ROOT/venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

echo "Scraping: $LEAGUE"
python -m fplpredict.scrape.fbref --league "$LEAGUE" "$@"

echo ""
python scripts/check_scrape_coverage.py
