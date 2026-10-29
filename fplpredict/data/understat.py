"""Historical match xG from Understat (getLeagueData API)."""

from __future__ import annotations

import json
import logging
import time

import pandas as pd
import requests

from fplpredict.config import active_leagues, model_season_window
from fplpredict.teams import normalize_team_name

log = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}

def _understat_title(title: str) -> str:
    """Map Understat club label to the same canonical names as FBref rows."""
    return normalize_team_name(title)

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (compatible; FPLPredict/1.0)",
        "X-Requested-With": "XMLHttpRequest",
    }
)


def _us_team(title: str | None) -> str | None:
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return None
    return _understat_title(str(title).strip())


def _season_end_to_understat_start(season_end: int) -> int:
    return int(season_end) - 1


def _fetch_league_matches(league_slug: str, season_start: int) -> list[dict]:
    url = f"https://understat.com/getLeagueData/{league_slug}/{season_start}"
    referer = f"https://understat.com/league/{league_slug}/{season_start}"
    for attempt in range(3):
        try:
            resp = _SESSION.get(url, headers={"Referer": referer}, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            dates = payload.get("dates") or []
            out: list[dict] = []
            for item in dates:
                if isinstance(item, dict):
                    out.append(item)
            return out
        except (requests.RequestException, json.JSONDecodeError) as exc:
            log.warning(
                "Understat %s %s fetch failed (attempt %d): %s",
                league_slug,
                season_start,
                attempt + 1,
                exc,
            )
            time.sleep(2 * (attempt + 1))
    return []


def build_understat_lookup(
    leagues: list[str] | None = None,
    season_end_years: set[int] | None = None,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Map (iso_date, home, away) -> (home_xg, away_xg)."""
    leagues = leagues or active_leagues()
    if season_end_years is None:
        season_end_years = set(model_season_window())
    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    for league in leagues:
        slug = UNDERSTAT_LEAGUE.get(league)
        if not slug:
            continue
        starts = sorted({_season_end_to_understat_start(y) for y in season_end_years})
        for season_start in starts:
            log.info("Understat: fetching %s season %d…", league, season_start)
            for row in _fetch_league_matches(slug, season_start):
                if not row.get("isResult"):
                    continue
                dt_raw = row.get("datetime") or ""
                dt = pd.to_datetime(dt_raw, errors="coerce")
                if pd.isna(dt):
                    continue
                home = _us_team((row.get("h") or {}).get("title"))
                away = _us_team((row.get("a") or {}).get("title"))
                xg = row.get("xG") or {}
                try:
                    hxg = float(xg.get("h"))
                    axg = float(xg.get("a"))
                except (TypeError, ValueError):
                    continue
                if not home or not away:
                    continue
                key = (dt.date().isoformat(), home, away)
                lookup[key] = (hxg, axg)
            time.sleep(0.5)
    log.info("Understat lookup: %d fixtures.", len(lookup))
    return lookup


def enrich_from_understat_lookup(
    matches: pd.DataFrame,
    lookup: dict[tuple[str, str, str], tuple[float, float]],
    *,
    only_missing: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Fill team-match xG/xGA from Understat fixture lookup."""
    if matches.empty or not lookup:
        return matches, 0
    from fplpredict.data.football_data import enrich_team_match_xg

    return enrich_team_match_xg(matches, lookup, only_missing=only_missing)


def enrich_matches_understat(
    matches: pd.DataFrame,
    *,
    leagues: list[str] | None = None,
    only_missing: bool = True,
) -> tuple[pd.DataFrame, int]:
    seasons = set()
    if not matches.empty and "Season" in matches.columns:
        seasons = {int(s) for s in matches["Season"].dropna().unique()}
    if not seasons:
        seasons = set(model_season_window())
    lookup = build_understat_lookup(leagues, seasons)
    out, filled = enrich_from_understat_lookup(matches, lookup, only_missing=only_missing)
    if filled:
        log.info("Understat filled xG/xGA on %d team-match rows.", filled)
    return out, filled
