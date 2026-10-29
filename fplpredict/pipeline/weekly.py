"""Weekly update: scrape → refit/predict → export for website."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from fplpredict.config import MODEL_RESELECT_DAYS, active_leagues, model_season_window
from fplpredict.data.football_data import sync_football_data_csvs
from fplpredict.data.validate import run_data_quality_checks
from fplpredict.data.xg_enrich import enrich_csv_files, understat_backfill_done
from fplpredict.export import export_predictions_json
from fplpredict.model.ensemble import (
    load_selection_meta,
    refresh_test_season_metrics,
    selection_is_stale,
)
from fplpredict.model.predict import run_all_league_predictions
from fplpredict.paths import LOG_DIR
from fplpredict.pipeline.status import (
    PARTIAL_FAILURE_EXIT,
    PipelineLocked,
    build_pipeline_status,
    pipeline_lock,
    write_pipeline_status,
)
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


def _sync_and_enrich(leagues: list[str], log: logging.Logger) -> None:
    sync_football_data_csvs(leagues, model_season_window())
    enrich_stats = enrich_csv_files(use_understat=False)
    log.info(
        "xG enrichment (football-data): %d rows, missing rate %.1f%% → %.1f%%.",
        enrich_stats["rows"],
        enrich_stats["xG_missing_rate_before"] * 100,
        enrich_stats["xG_missing_rate_after"] * 100,
    )


def run_weekly(args: argparse.Namespace, log: logging.Logger) -> int:
    leagues = active_leagues()
    stale_leagues = [
        league for league in leagues if selection_is_stale(load_selection_meta(league))
    ]
    will_reselect = args.force_model_reselect or bool(stale_leagues)
    log.info(
        "Schedule: weekly scrape + refit on latest data; strategy re-selection "
        "every %d days (this run: %s%s).",
        MODEL_RESELECT_DAYS,
        "forced" if args.force_model_reselect else (
            "reselect" if will_reselect else "refit only"
        ),
        f" — stale: {', '.join(stale_leagues)}" if stale_leagues and not args.force_model_reselect else "",
    )

    errors: dict[str, str] = {}
    reports: dict[str, dict] = {}
    try:
        if args.skip_scrape:
            log.info("Skipping scrape (--skip-scrape).")
            _sync_and_enrich(leagues, log)
        else:
            log.info("Starting FBref scrape…")
            run_scrape()
            log.info("Scrape finished (includes football-data sync and xG enrich).")

        if not understat_backfill_done():
            log.warning(
                "Understat historical xG backfill not done yet — run "
                "PYTHONPATH=. python scripts/backfill_understat_xg.py once."
            )
        quality = run_data_quality_checks(leagues, strict=True)
        reports = {r.league: r.stats for r in quality}

        log.info("Running prediction pipeline for: %s", ", ".join(leagues))
        completed = run_all_league_predictions(
            force_model_reselect=args.force_model_reselect, errors=errors
        )
        log.info(
            "Predictions written for %d league(s): %s",
            len(completed),
            ", ".join(completed),
        )

        for league in completed:
            refresh_test_season_metrics(league)
            try:
                from fplpredict.model.rolling_backtest import write_rolling_backtest

                write_rolling_backtest(league)
            except Exception:
                log.warning("[%s] Rolling backtest failed.", league, exc_info=True)

        exported = export_predictions_json()
        if exported:
            log.info(
                "Exported site JSON (default league: %d upcoming, %d history).",
                len(exported.get("upcoming", [])),
                len(exported.get("history", [])),
            )
        else:
            log.warning("No league JSON exported (missing predictions CSVs).")
    except Exception as exc:
        log.exception("Weekly pipeline failed.")
        for league in leagues:
            errors.setdefault(league, f"{type(exc).__name__}: {exc}")
        write_pipeline_status(build_pipeline_status(leagues, reports, errors))
        return 1

    write_pipeline_status(build_pipeline_status(leagues, reports, errors))
    if errors:
        log.error("Leagues failed this run: %s", ", ".join(sorted(errors)))
        return PARTIAL_FAILURE_EXIT
    return 0


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
        with pipeline_lock():
            return run_weekly(args, log)
    except PipelineLocked as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
