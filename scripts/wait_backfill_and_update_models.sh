#!/usr/bin/env bash
# Wait for a running fbref_backfill (by PID or log file), then refit + force strategy reselect.
#
#   ./scripts/wait_backfill_and_update_models.sh --pid 519182 --log data/backfill_seriea_ligue1.log
#   ./scripts/wait_backfill_and_update_models.sh --log data/backfill_seriea_ligue1.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PID=""
LOG=""
POLL_SEC=45

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid) PID="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --poll) POLL_SEC="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$LOG" ]]; then
  echo "Usage: $0 --log <backfill.log> [--pid <scrape-pid>]" >&2
  exit 2
fi

echo "[wait] Watching $LOG (pid=${PID:-none}) for backfill completion…"

while true; do
  if grep -q "xG enrich:" "$LOG" 2>/dev/null; then
    echo "[wait] Backfill post-process finished (xG enrich logged)."
    break
  fi
  if grep -qE "Traceback|ModuleNotFoundError|Could not load league" "$LOG" 2>/dev/null; then
    echo "[wait] Backfill log shows an error — not running model update." >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  if [[ -n "$PID" ]] && ! kill -0 "$PID" 2>/dev/null; then
    if grep -q "xG enrich:" "$LOG" 2>/dev/null; then
      break
    fi
    echo "[wait] Scrape PID $PID exited but xG enrich not in log yet; waiting…"
  fi
  sleep "$POLL_SEC"
done

export FPL_MODEL_DATA_SEASONS="${FPL_MODEL_DATA_SEASONS:-7}"
export FPL_YEAR_DIFF="${FPL_YEAR_DIFF:-7}"
export FPL_TRAIN_SEASONS="${FPL_TRAIN_SEASONS:-5}"
export FPL_VAL_SEASONS="${FPL_VAL_SEASONS:-1}"
export FPL_TEST_SEASONS="${FPL_TEST_SEASONS:-1}"

OUT="${ROOT}/data/model_update_after_backfill.log"
echo "[wait] Starting weekly pipeline (skip scrape, force model reselect) → $OUT"
./scripts/weekly_update.sh --skip-scrape --force-model-reselect 2>&1 | tee "$OUT"
echo "[wait] Model update finished."
