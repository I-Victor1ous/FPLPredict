"""Weekly update: scrape → predict → export for website."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from fplpredict.config import MODEL_RESELECT_DAYS
from fplpredict.export import export_predictions_json
from fplpredict.model.ensemble import load_selection_meta, selection_is_stale
from fplpredict.model.predict import run_prediction_pipeline
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
        meta = load_selection_meta()
        will_reselect = args.force_model_reselect or selection_is_stale(meta)
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

        log.info("Running prediction pipeline…")
        run_prediction_pipeline(force_model_reselect=args.force_model_reselect)
        log.info("Predictions written.")

        payload = export_predictions_json()
        log.info(
            "Exported JSON (%d upcoming, %d history rows).",
            len(payload["upcoming"]),
            len(payload["history"]),
        )
        return 0
    except Exception:
        log.exception("Weekly pipeline failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
