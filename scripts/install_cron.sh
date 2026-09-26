#!/usr/bin/env bash
# Install a weekly cron job (Sunday 20:00). Run from WSL/Linux.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$ROOT/scripts/weekly_update.sh"
MARKER="# FPLPredict weekly"

if [[ ! -x "$WRAPPER" ]]; then
  chmod +x "$WRAPPER"
fi

CRON_LINE="0 20 * * 0 cd $ROOT && $WRAPPER >> $ROOT/logs/cron.log 2>&1"
EXISTING="$(crontab -l 2>/dev/null || true)"

if echo "$EXISTING" | grep -qF "$MARKER"; then
  echo "Cron entry already installed."
  exit 0
fi

{
  echo "$EXISTING" | grep -vF "$MARKER" || true
  echo "$MARKER"
  echo "$CRON_LINE"
} | crontab -

echo "Installed weekly cron:"
echo "  $CRON_LINE"
echo "Logs: $ROOT/logs/cron.log"
echo ""
echo "Each run: scrape → predict → export site JSON."
echo "Model reselection: automatic when older than FPL_MODEL_RESELECT_DAYS (default 30)."
