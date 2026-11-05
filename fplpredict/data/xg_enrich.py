"""Orchestrate xG enrichment: Understat (historical backfill) + football-data (ongoing)."""

from __future__ import annotations

import logging

from fplpredict.paths import PROJECT_ROOT

log = logging.getLogger(__name__)

UNDERSTAT_BACKFILL_FLAG = PROJECT_ROOT / "data" / ".understat_xg_backfilled"


def understat_backfill_done() -> bool:
    return UNDERSTAT_BACKFILL_FLAG.is_file()


def mark_understat_backfill_done() -> None:
    UNDERSTAT_BACKFILL_FLAG.parent.mkdir(parents=True, exist_ok=True)
    UNDERSTAT_BACKFILL_FLAG.write_text("ok\n", encoding="utf-8")


def enrich_csv_files(
    *,
    use_understat: bool = False,
    force_understat: bool = False,
    refresh_understat_missing: bool = False,
) -> dict[str, float | int]:
    from fplpredict.data.football_data import enrich_matches_and_history
    from fplpredict.data.understat import enrich_matches_understat
    from fplpredict.data_io import read_csv, write_csv

    matches = read_csv("matches.csv")
    try:
        history = read_csv("history.csv")
    except FileNotFoundError:
        history = None

    before = float(matches["xG"].isna().mean()) if "xG" in matches.columns else 1.0
    understat_rows = 0

    run_us = (
        force_understat
        or (use_understat and not understat_backfill_done())
        or refresh_understat_missing
    )
    if (use_understat or force_understat) and not run_us and not refresh_understat_missing:
        log.info("Understat backfill already done; using football-data only.")

    if run_us:
        log.info("Enriching missing xG from Understat…")
        matches, understat_rows = enrich_matches_understat(matches, only_missing=True)
        if history is not None and not history.empty:
            from fplpredict.data.football_data import enrich_history_xg
            from fplpredict.data.understat import build_understat_lookup

            lookup = build_understat_lookup()
            history = enrich_history_xg(history, lookup)
        mark_understat_backfill_done()

    log.info("Enriching missing xG from football-data.co.uk…")
    matches, history = enrich_matches_and_history(matches, history)

    from fplpredict.data.matches_clean import (
        clean_matches_for_modeling,
        clean_next_matches,
        propagate_xg_within_fixtures,
    )

    matches = clean_matches_for_modeling(matches)
    matches, _ = propagate_xg_within_fixtures(matches)
    try:
        from fplpredict.data.football_data import backfill_next_match_results_from_football_data

        nextm = read_csv("next_matches.csv")
        nextm, _ = backfill_next_match_results_from_football_data(nextm)
        write_csv(clean_next_matches(nextm), "next_matches.csv")
    except FileNotFoundError:
        pass

    after = float(matches["xG"].isna().mean()) if "xG" in matches.columns else 1.0
    write_csv(matches, "matches.csv")
    if history is not None:
        write_csv(history, "history.csv")

    return {
        "rows": len(matches),
        "understat_filled": understat_rows,
        "xG_missing_rate_before": round(before, 4),
        "xG_missing_rate_after": round(after, 4),
    }
