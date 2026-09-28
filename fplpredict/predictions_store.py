"""Per-league prediction CSV paths and one-time legacy file migration."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fplpredict.config import LEAGUE as DEFAULT_LEAGUE
from fplpredict.paths import OUTPUT_DIR, league_predictions_csv
from fplpredict.predictions_util import dedupe_true_duplicates

log = logging.getLogger(__name__)

LEGACY_PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
LEGACY_PREDICTIONS_JSON = OUTPUT_DIR / "predictions.json"


def migrate_legacy_prediction_files() -> None:
    """
    Move ``outputs/predictions.csv`` into ``predictions-<default-slug>.csv``
    and remove duplicate legacy outputs.
    """
    target = league_predictions_csv(DEFAULT_LEAGUE)
    frames: list[pd.DataFrame] = []
    if target.exists():
        frames.append(pd.read_csv(target, index_col=0))
    if LEGACY_PREDICTIONS_CSV.exists():
        frames.append(pd.read_csv(LEGACY_PREDICTIONS_CSV, index_col=0))
    if frames:
        merged = dedupe_true_duplicates(pd.concat(frames, ignore_index=True))
        target.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(target)
        log.info("Canonical predictions for %s: %s", DEFAULT_LEAGUE, target)

    for path in (
        LEGACY_PREDICTIONS_CSV,
        LEGACY_PREDICTIONS_JSON,
    ):
        if path.exists():
            path.unlink()
            log.info("Removed legacy file %s", path)

    for path in OUTPUT_DIR.glob("predictions-*.json"):
        path.unlink()
        log.info("Removed outputs JSON %s (site JSON lives under web/)", path)
