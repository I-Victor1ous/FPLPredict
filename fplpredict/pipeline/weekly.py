"""Weekly update: scrape → predict → export for website."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from fplpredict.config import MODEL_RESELECT_DAYS
from fplpredict.export import export_predictions_json
from fplpredict.model.ensemble import load_selection_meta, selection_is_stale
from fplpredict.config import active_leagues
from fplpredict.model.predict import run_all_league_predictions
from fplpredict.paths import LOG_DIR
from fplpredict.scrape.fbref import run_scrape


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FPLPredict weekly pipeline")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip FBref scrape (use existing CSV data)",
    )
    parser.add_argument(
        "--force-model-reselect",
        action="store_true",
        help="Re-run model comparison and pick best strategy (ignores monthly cache)",
    )
    args = parser.parse_args()
    setup_logging()
    log = logging.getLogger(__name__)

    try:
        leagues = active_leagues()
        stale_any = any(
            selection_is_stale(load_selection_meta(league)) for league in leagues
        )
        will_reselect = args.force_model_reselect or stale_any
        log.info(
            "Schedule: weekly scrape + predict; model reselection every %d days "
            "(this run: %s).",
            MODEL_RESELECT_DAYS,
            "forced" if args.force_model_reselect else (
                "yes" if will_reselect else "no (using cached winner)"
            ),
        )

        if not args.skip_scrape:
            log.info("Starting FBref scrape…")
            run_scrape()
            log.info("Scrape finished.")
        else:
            log.info("Skipping scrape (--skip-scrape).")

        log.info("Running prediction pipeline for: %s", ", ".join(leagues))
        completed = run_all_league_predictions(
            force_model_reselect=args.force_model_reselect
        )
        log.info(
            "Predictions written for %d league(s): %s",
            len(completed),
            ", ".join(completed),
        )

        exported = export_predictions_json()
        if exported:
            log.info(
                "Exported site JSON (default league: %d upcoming, %d history).",
                len(exported.get("upcoming", [])),
                len(exported.get("history", [])),
            )
        else:
            log.warning("No league JSON exported (missing predictions CSVs).")
        return 0
    except Exception:
        log.exception("Weekly pipeline failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
