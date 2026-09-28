"""Scrape match data from FBref."""

from __future__ import annotations

import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from seleniumbase import Driver

from fplpredict.config import LEAGUE_INFO_URLS, YEAR_DIFF, active_leagues
from fplpredict.model.splits import prune_matches_by_season_window
from fplpredict.data_io import read_csv, write_csv
from fplpredict.paths import DATA_DIR, LOG_DIR


def _read_table_with_retry(driver, url, id_match, table_match, retries=3, delay=5):
    last_error = None
    for attempt in range(retries):
        driver.get(url)
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, id_match))
            )
            page_source = driver.page_source
            time.sleep(5)
            table = pd.read_html(StringIO(page_source), match=table_match)[0]
            return table, page_source
        except (TimeoutException, ValueError, IndexError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay)
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
    for team_url in team_urls:
        team_name_with_dash = team_url.split("/")[-1].replace("-Stats", "")
        team_name = team_name_with_dash.replace("-", " ")
        team_id_match = re.search("/squads/[^/]+", team_url)
        team_id = team_id_match.group(0).split("/")[-1]
        team_url = re.sub(r"(/[^/]+)$", r"/all_comps\1-All-Competitions", team_url)

        matches, page_source = _read_table_with_retry(
            driver, team_url, "matchlogs_for", "Scores & Fixtures"
        )
        matches["Season"] = year
        matches["Team"] = team_name
        past_matches = matches[matches["Result"].notna()]
        next_match = matches[matches["Result"].isna()]

        soup = BeautifulSoup(page_source, features="html.parser")
        links = [a.get("href") for a in soup.find_all("a")]

        past_week_teams = set()
        try:
            past_week_matches_df = read_csv("next_matches.csv")
            past_week_teams = set(past_week_matches_df["Team"])
        except FileNotFoundError:
            pass

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

            if (
                team_name in past_week_teams
                and opp_name.replace("-", " ") in past_week_teams
            ) or len(past_week_teams) == 0:
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
                except (RuntimeError, ValueError):
                    continue

        shooting_links = [l for l in links if l and "all_comps/shooting/" in l]
        misc_links = [l for l in links if l and "all_comps/misc/" in l]
        if not shooting_links or not misc_links:
            continue

        shooting, _ = _read_table_with_retry(
            driver,
            f"https://fbref.com{shooting_links[0]}",
            "matchlogs_for",
            "Shooting",
        )
        misc, _ = _read_table_with_retry(
            driver,
            f"https://fbref.com{misc_links[0]}",
            "matchlogs_for",
            "Miscellaneous Stats",
        )

        shooting.columns = shooting.columns.droplevel()
        misc.columns = misc.columns.droplevel()

        try:
            team_data = past_matches.merge(
                shooting[["Date", "Sh", "SoT", "PK", "PKatt"]], on="Date", how="left"
            )
            team_data = team_data.merge(
                misc[
                    [
                        "Date",
                        "CrdY",
                        "CrdR",
                        "2CrdY",
                        "Fls",
                        "Fld",
                        "Off",
                        "Crs",
                        "Int",
                        "TklW",
                    ]
                ],
                on="Date",
                how="left",
            )
        except ValueError:
            continue

        team_data = team_data[team_data["Date"] != "Date"]
        all_matches.append(team_data)
        if not next_match.empty:
            next_match = next_match[next_match["Comp"] == league]
            next_matches.append(pd.DataFrame([next_match.iloc[0]]))

        time.sleep(5)

    return all_matches, next_matches, history_matches


def _collect_seasons(driver, league: str, current_data_year: int = 0):
    end_year = datetime.now().year
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
            l.get("href")
            for l in standings_table.find_all("a")
            if l.get("href") and "/squads/" in l.get("href")
        ]
        team_urls = [f"https://fbref.com{l}" for l in links]
        previous_season = soup.select("a.prev")[0].get("href")
        info_url = f"https://fbref.com{previous_season}"

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

    return past_matches, next_matches, history_matches


def _save_scrape_results(new_matches, next_new_matches, history_new_matches) -> None:
    if new_matches:
        new_matches = pd.concat(new_matches, ignore_index=True)
        try:
            csv_all_matches = read_csv("matches.csv").sort_values(
                "Date", ascending=False
            )
            unique_keys = ["Date", "Team", "Opponent"]
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
        match_df = prune_matches_by_season_window(match_df)
        write_csv(match_df, "matches.csv")

    if next_new_matches:
        next_df = pd.concat(next_new_matches, ignore_index=True)
        try:
            existing_next = read_csv("next_matches.csv")
            scraped_comps = set(next_df["Comp"].dropna().unique())
            keep = existing_next[~existing_next["Comp"].isin(scraped_comps)]
            next_df = pd.concat([next_df, keep], ignore_index=True)
        except FileNotFoundError:
            pass
        write_csv(next_df, "next_matches.csv")

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


def _scrape_league(driver, league: str) -> tuple:
    try:
        csv_all_matches = read_csv("matches.csv").sort_values("Date", ascending=False)
        league_rows = csv_all_matches[csv_all_matches["Comp"] == league]
        if league_rows.empty:
            current_data_year = 0
        else:
            current_data_year = int(league_rows.iloc[0]["Season"])
        return _collect_seasons(driver, league, current_data_year)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return _collect_seasons(driver, league)


def run_scrape() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    driver = Driver(uc=True, browser="chrome")
    try:
        all_past, all_next, all_history = [], [], []
        for league in active_leagues():
            past, nxt, hist = _scrape_league(driver, league)
            all_past.extend(past)
            all_next.extend(nxt)
            all_history.extend(hist)
        _save_scrape_results(all_past, all_next, all_history)
    finally:
        driver.quit()
