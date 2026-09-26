#!/usr/bin/env bash
# Weekly FPLPredict update (WSL/Linux cron or manual run).
# - Scrape + predict + export JSON every run (see install_cron.sh for Sunday 20:00).
# - Model strategy is re-compared about monthly (FPL_MODEL_RESELECT_DAYS, default 30).
# - Force reselection: ./scripts/weekly_update.sh --force-model-reselect
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -d "$ROOT/venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m fplpredict.pipeline.weekly "$@"
