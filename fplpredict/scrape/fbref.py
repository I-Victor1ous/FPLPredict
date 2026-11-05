"""Scrape match data from FBref."""

from __future__ import annotations

import logging
import re
import time
from io import StringIO

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from seleniumbase import Driver

from fplpredict.config import (
    LEAGUE_INFO_URLS,
    YEAR_DIFF,
    active_leagues,
    current_season_end,
    model_season_window,
)
from fplpredict.data_io import read_csv, write_csv
from fplpredict.model.splits import prune_matches_by_season_window
from fplpredict.paths import DATA_DIR, LOG_DIR
from fplpredict.scrape.team_logs import build_team_league_rows, filter_league_fixtures

log = logging.getLogger(__name__)


def _read_table_with_retry(
    driver,
    url,
    id_match,
    table_match,
    retries: int = 5,
    delay: int = 8,
):
    """Load an FBref table; retry on Selenium/HTTP timeouts (common on H2H pages)."""
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            driver.get(url)
        except Exception as exc:  # urllib3 ReadTimeoutError, WebDriverException, etc.
            last_error = exc
            if attempt < retries - 1:
                wait = delay * (attempt + 1)
                log.warning(
                    "driver.get failed (%s) for %s — retry %d/%d in %ds",
                    type(exc).__name__,
                    url,
                    attempt + 1,
                    retries,
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"Failed to load {url}") from exc
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, id_match))
            )
            page_source = driver.page_source
            time.sleep(5)
            table = pd.read_html(StringIO(page_source), match=table_match)[0]
            return table, page_source
        except (TimeoutException, ValueError, IndexError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to read {url}") from last_error


def _get_matches_stats(
    driver,
    team_urls,
    year,
    all_matches,
    next_matches,
    history_matches,
    head_to_head_urls,
    league: str,
):
    for i, team_url in enumerate(team_urls, start=1):
        team_name_with_dash = team_url.split("/")[-1].replace("-Stats", "")
        team_name = team_name_with_dash.replace("-", " ")
        log.info("[%s] Season %s: squad %d/%d %s", league, year, i, len(team_urls), team_name)
        team_id_match = re.search("/squads/[^/]+", team_url)
        team_id = team_id_match.group(0).split("/")[-1]
        team_url = re.sub(r"(/[^/]+)$", r"/all_comps\1-All-Competitions", team_url)

        try:
            matches, page_source = _read_table_with_retry(
                driver, team_url, "matchlogs_for", "Scores & Fixtures"
            )
        except RuntimeError as exc:
            log.error(
                "[%s] %s: could not load Scores & Fixtures (%s).",
                league,
                team_name,
                exc,
            )
            continue
        matches["Season"] = int(year) if str(year).isdigit() else year
        matches["Team"] = team_name
        past_matches = matches[matches["Result"].notna()]
        next_match = matches[matches["Result"].isna()]

        soup = BeautifulSoup(page_source, features="html.parser")
        links = [a.get("href") for a in soup.find_all("a")]

        td_tags = set(
            soup.find_all("td", {"class": "left", "data-stat": "opponent"})
        )
        for td in td_tags:
            link = td.find("a").get("href")
            opp_name = link.split("/")[-1].replace("-Stats", "")
            opp_id_match = re.search("/squads/[^/]+", link)
            opp_id = opp_id_match.group(0).split("/")[-1]
            hth_url = (
                "https://fbref.com/en/stathead/matchup/"
                f"teams/{team_id}/{opp_id}/{team_name_with_dash}"
                f"-vs-{opp_name}-History"
            )
            if hth_url in head_to_head_urls:
                continue
            head_to_head_urls.add(hth_url)

            try:
                hth_matches, _ = _read_table_with_retry(
                    driver,
                    hth_url,
                    "games_history_all",
                    "Head-to-Head Matches",
                )
                hth_matches = hth_matches[hth_matches["Score"].notna()]
                hth_matches = hth_matches[hth_matches["Date"] != "Date"]
                hth_matches["Team"] = team_name
                history_matches.append(hth_matches)
            except Exception as exc:
                log.debug(
                    "[%s] Skipping H2H %s (%s).",
                    league,
                    hth_url,
                    exc,
                )
                continue

        shooting_df: pd.DataFrame | None = None
        misc_df: pd.DataFrame | None = None
        shooting_links = [href for href in links if href and "all_comps/shooting/" in href]
        misc_links = [href for href in links if href and "all_comps/misc/" in href]
        if shooting_links and misc_links:
            try:
                shooting_df, _ = _read_table_with_retry(
                    driver,
                    f"https://fbref.com{shooting_links[0]}",
                    "matchlogs_for",
                    "Shooting",
                )
                misc_df, _ = _read_table_with_retry(
                    driver,
                    f"https://fbref.com{misc_links[0]}",
                    "matchlogs_for",
                    "Miscellaneous Stats",
                )
            except RuntimeError as exc:
                log.warning(
                    "[%s] %s: shooting/misc fetch failed (%s); keeping fixtures only.",
                    league,
                    team_name,
                    exc,
                )
        else:
            log.warning(
                "[%s] %s: no shooting/misc links; keeping fixture rows only.",
                league,
                team_name,
            )

        team_data = build_team_league_rows(
            past_matches, league, team_name, shooting_df, misc_df
        )
        if team_data.empty:
            log.warning("[%s] %s: no completed league fixtures on match log.", league, team_name)
            continue

        all_matches.append(team_data)
        if not next_match.empty:
            nxt = filter_league_fixtures(next_match, league, team_name=team_name)
            if not nxt.empty:
                next_matches.append(pd.DataFrame([nxt.iloc[0]]))

        time.sleep(5)

    return all_matches, next_matches, history_matches


def _collect_seasons(driver, league: str, current_data_year: int = 0):
    end_year = current_season_end()
    earliest = end_year - (YEAR_DIFF - 1)
    start_year = max(earliest, current_data_year)
    years = np.arange(end_year, start_year - 1, -1)
    if len(years) == 0:
        years = np.array([end_year])

    past_matches, next_matches, history_matches = [], [], []
    head_to_head_urls = set()
    info_url = LEAGUE_INFO_URLS[league]

    for _year in years:
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

        soup = BeautifulSoup(page_source, features="html.parser")
        title = soup.select("h1")[0]
        title_match = re.search(r"([0-9]{4})-([0-9]{4})", str(title))
        season_end = title_match.group(2)

        standings_table = soup.select("table.stats_table")[0]
        links = [
            a.get("href")
            for a in standings_table.find_all("a")
            if a.get("href") and "/squads/" in a.get("href")
        ]
        team_urls = [f"https://fbref.com{href}" for href in links]
        previous_season = soup.select("a.prev")[0].get("href")
        info_url = f"https://fbref.com{previous_season}"

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
        log.info(
            "[%s] Season %s: %d/%d squads produced league match logs.",
            league,
            season_end,
            added,
            len(team_urls),
        )
        if added < len(team_urls):
            log.warning(
                "[%s] Season %s: %d squads returned no league rows (check warnings above).",
                league,
                season_end,
                len(team_urls) - added,
            )

    return past_matches, next_matches, history_matches


def _save_scrape_results(new_matches, next_new_matches, history_new_matches) -> None:
    if new_matches:
        new_matches = pd.concat(new_matches, ignore_index=True)
        try:
            from fplpredict.data.matches_clean import MIRRORED_COL

            csv_all_matches = read_csv("matches.csv").sort_values(
                "Date", ascending=False
            )
            unique_keys = ["Date", "Team", "Opponent"]
            if MIRRORED_COL in csv_all_matches.columns:
                scraped_keys = new_matches.set_index(unique_keys).index
                stale = csv_all_matches[MIRRORED_COL].fillna(False).astype(bool) & (
                    csv_all_matches.set_index(unique_keys).index.isin(scraped_keys)
                )
                csv_all_matches = csv_all_matches[~stale]
            mask = ~new_matches.set_index(unique_keys).index.isin(
                csv_all_matches.set_index(unique_keys).index
            )
            new_unique_matches = new_matches[mask]
            match_df = pd.concat(
                [df for df in [new_unique_matches, csv_all_matches] if not df.empty],
                ignore_index=True,
            )
        except FileNotFoundError:
            match_df = new_matches
        from fplpredict.data.matches_clean import clean_matches_for_modeling

        match_df = clean_matches_for_modeling(match_df)
        match_df = prune_matches_by_season_window(match_df)
        write_csv(match_df, "matches.csv")

    if next_new_matches:
        from fplpredict.data.matches_clean import clean_next_matches

        next_df = pd.concat(next_new_matches, ignore_index=True)
        try:
            existing_next = read_csv("next_matches.csv")
            scraped_comps = set(next_df["Comp"].dropna().unique())
            keep = existing_next[~existing_next["Comp"].isin(scraped_comps)]
            next_df = pd.concat([next_df, keep], ignore_index=True)
        except FileNotFoundError:
            pass
        write_csv(clean_next_matches(next_df), "next_matches.csv")

    if history_new_matches:
        history_new_matches = pd.concat(history_new_matches, ignore_index=True)
        try:
            csv_history = read_csv("history.csv").sort_values("Date", ascending=False)
            history_new_matches["Match_key"] = history_new_matches.apply(
                lambda row: tuple(sorted([row["Home"], row["Away"]])), axis=1
            )
            history_new_matches.drop_duplicates(
                subset=["Match_key", "Date"], inplace=True
            )
            history_new_matches.drop(
                columns=["Match_key", "Match Report"], errors="ignore"
            )
            unique_keys = ["Date", "Home", "Away", "Comp"]
            mask = ~history_new_matches.set_index(unique_keys).index.isin(
                csv_history.set_index(unique_keys).index
            )
            history_unique = history_new_matches[mask]
            history_df = pd.concat(
                [df for df in [history_unique, csv_history] if not df.empty],
                ignore_index=True,
            )
        except FileNotFoundError:
            history_df = history_new_matches
        write_csv(history_df, "history.csv")


def _scrape_league(driver, league: str, *, full: bool = False) -> tuple:
    """Scrape from the newest stored season onward, or every window season if ``full``."""
    current_data_year = 0
    if full:
        return _collect_seasons(driver, league, current_data_year)
    try:
        csv_all_matches = read_csv("matches.csv").sort_values("Date", ascending=False)
        league_rows = csv_all_matches[csv_all_matches["Comp"] == league]
        if not league_rows.empty:
            current_data_year = int(
                pd.to_numeric(league_rows.iloc[0]["Season"], errors="coerce")
            )
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, TypeError):
        current_data_year = 0
    return _collect_seasons(driver, league, current_data_year)


def run_scrape(
    leagues: list[str] | None = None,
    *,
    skip_post_process: bool = False,
    full: bool = False,
) -> None:
    """Scrape FBref match logs. Default: all active leagues + xG enrichment."""
    targets = list(leagues) if leagues else active_leagues()
    unknown = set(targets) - set(LEAGUE_INFO_URLS)
    if unknown:
        raise ValueError(f"Unknown league(s): {sorted(unknown)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    driver = Driver(uc=True, browser="chrome")
    try:
        for league in targets:
            log.info("Scraping %s…", league)
            past, nxt, hist = _scrape_league(driver, league, full=full)
            _save_scrape_results(past, nxt, hist)
        if skip_post_process:
            log.info("Skipping football-data sync and xG enrich (--skip-post-process).")
            return
        from fplpredict.data.football_data import sync_football_data_csvs
        from fplpredict.data.xg_enrich import enrich_csv_files

        sync_football_data_csvs(targets, model_season_window())
        # football-data has no xG before 2026-27; past seasons come from Understat.
        enrich_csv_files(use_understat=False, refresh_understat_missing=full)
    finally:
        driver.quit()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Scrape match data from FBref")
    parser.add_argument(
        "--league",
        action="append",
        dest="leagues",
        metavar="NAME",
        help="Scrape only this league (repeatable). Default: all configured leagues.",
    )
    parser.add_argument(
        "--skip-post-process",
        action="store_true",
        help="Do not run football-data sync or xG enrich after scrape (faster smoke test).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-walk every season in the scrape window, not just the newest stored one.",
    )
    args = parser.parse_args()
    run_scrape(args.leagues, skip_post_process=args.skip_post_process, full=args.full)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
