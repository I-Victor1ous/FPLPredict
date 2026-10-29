#!/usr/bin/env bash
# Optional: commit exported site JSON and push (run after weekly_update.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if ! git diff --quiet -- web/public/data; then
  git add web/public/data/*.json
  git commit -m "chore: refresh site prediction JSON"
  git push
else
  echo "No changes in web/public/data"
fi
