"""Scrape specific historical FBref seasons (e.g. 2021–2022) without a full --full walk."""

from __future__ import annotations

import argparse
import logging
import re
import time

from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from seleniumbase import Driver

from fplpredict.config import LEAGUE_INFO_URLS, active_leagues
from fplpredict.paths import DATA_DIR, LOG_DIR
from fplpredict.scrape.fbref import (
    _get_matches_stats,
    _save_scrape_results,
)

log = logging.getLogger(__name__)

DEFAULT_TARGET_SEASONS = (2021, 2022)
MAX_SEASON_PAGES = 40


def parse_target_seasons(raw: str) -> set[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("No seasons in --seasons")
    return {int(p) for p in parts}


def _load_league_page(driver, info_url: str) -> BeautifulSoup:
    page_source = None
    for attempt in range(3):
        driver.uc_open_with_reconnect(info_url, reconnect_time=5)
        driver.uc_gui_click_captcha()
        (LOG_DIR / "headless_dom.html").write_text(
            driver.page_source, encoding="utf-8"
        )
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "stats_table"))
            )
            time.sleep(5)
            page_source = driver.page_source
            break
        except TimeoutException:
            if attempt < 2:
                time.sleep(5)
    if page_source is None:
        raise RuntimeError(f"Could not load league page: {info_url}")
    return BeautifulSoup(page_source, features="html.parser")


def _season_end_from_soup(soup: BeautifulSoup) -> int:
    title = soup.select("h1")[0]
    title_match = re.search(r"([0-9]{4})-([0-9]{4})", str(title))
    if not title_match:
        raise RuntimeError(f"Could not parse season from title: {title}")
    return int(title_match.group(2))


def _collect_target_seasons(
    driver,
    league: str,
    target_end_years: set[int],
) -> tuple[list, list, list]:
    """Walk backward from the current league page; scrape only target end-years."""
    if not target_end_years:
        return [], [], []
    min_target = min(target_end_years)
    past_matches, next_matches, history_matches = [], [], []
    head_to_head_urls: set[str] = set()
    info_url = LEAGUE_INFO_URLS[league]
    collected: set[int] = set()

    for _ in range(MAX_SEASON_PAGES):
        soup = _load_league_page(driver, info_url)
        season_end = _season_end_from_soup(soup)

        if season_end < min_target:
            log.warning(
                "[%s] Stopped at season %s (before min target %s); collected %s.",
                league,
                season_end,
                min_target,
                sorted(collected),
            )
            break

        if season_end in target_end_years:
            standings_table = soup.select("table.stats_table")[0]
            links = [
                a.get("href")
                for a in standings_table.find_all("a")
                if a.get("href") and "/squads/" in a.get("href")
            ]
            team_urls = [f"https://fbref.com{href}" for href in links]
            before = len(past_matches)
            past_matches, next_matches, history_matches = _get_matches_stats(
                driver,
                team_urls,
                season_end,
                past_matches,
                next_matches,
                history_matches,
                head_to_head_urls,
                league,
            )
            added = len(past_matches) - before
            collected.add(season_end)
            log.info(
                "[%s] Target season %s: %d/%d squads produced league match logs.",
                league,
                season_end,
                added,
                len(team_urls),
            )

        if collected >= target_end_years:
            break

        prev_links = soup.select("a.prev")
        if not prev_links:
            log.warning("[%s] No previous-season link at season %s.", league, season_end)
            break
        previous_season = prev_links[0].get("href")
        if not previous_season:
            break
        info_url = f"https://fbref.com{previous_season}"

    missing = target_end_years - collected
    if missing:
        log.warning("[%s] Did not scrape target seasons: %s", league, sorted(missing))

    return past_matches, next_matches, history_matches


def run_backfill_scrape(
    leagues: list[str] | None = None,
    *,
    target_end_years: set[int] | None = None,
    skip_post_process: bool = False,
) -> None:
    """Append match logs for ``target_end_years`` then enrich xG (Understat + football-data)."""
    targets = set(target_end_years or DEFAULT_TARGET_SEASONS)
    targets_list = sorted(targets)
    league_list = list(leagues) if leagues else active_leagues()
    unknown = set(league_list) - set(LEAGUE_INFO_URLS)
    if unknown:
        raise ValueError(f"Unknown league(s): {sorted(unknown)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "Backfill scrape for seasons %s across %d league(s).",
        targets_list,
        len(league_list),
    )

    driver = Driver(uc=True, browser="chrome")
    try:
        for league in league_list:
            log.info("Backfill %s…", league)
            past, nxt, hist = _collect_target_seasons(driver, league, targets)
            _save_scrape_results(past, nxt, hist)
        if skip_post_process:
            log.info("Skipping xG enrich (--skip-post-process).")
            return

        from fplpredict.data.football_data import sync_football_data_csvs
        from fplpredict.data.xg_enrich import enrich_csv_files
        from fplpredict.config import model_season_window

        sync_football_data_csvs(league_list, sorted(set(model_season_window()) | targets))
        stats = enrich_csv_files(refresh_understat_missing=True)
        log.info(
            "xG enrich: Understat filled %s rows; missing rate %.1f%% → %.1f%%.",
            stats.get("understat_filled", 0),
            float(stats["xG_missing_rate_before"]) * 100,
            float(stats["xG_missing_rate_after"]) * 100,
        )
    finally:
        driver.quit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape specific FBref season end-years (default 2021,2022) for all leagues."
    )
    parser.add_argument(
        "--seasons",
        default=",".join(str(y) for y in DEFAULT_TARGET_SEASONS),
        help="Comma-separated FBref season end-years (e.g. 2021,2022).",
    )
    parser.add_argument(
        "--league",
        action="append",
        dest="leagues",
        metavar="NAME",
        help="Limit to league(s); default all active leagues.",
    )
    parser.add_argument(
        "--skip-post-process",
        action="store_true",
        help="Skip Understat / football-data xG enrich after scrape.",
    )
    args = parser.parse_args()
    targets = parse_target_seasons(args.seasons)
    run_backfill_scrape(
        args.leagues,
        target_end_years=targets,
        skip_post_process=args.skip_post_process,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    raise SystemExit(main())
