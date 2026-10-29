"""Download and join match xG from football-data.co.uk (HxG / AxG)."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from fplpredict.config import active_leagues, current_season_end, model_season_window
from fplpredict.paths import PROJECT_ROOT
from fplpredict.teams import normalize_team_name

log = logging.getLogger(__name__)

FOOTBALL_DATA_DIV = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}

FD_TEAM_ALIASES: dict[str, str] = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton and Hove Albion",
    "West Ham": "West Ham United",
    "Newcastle": "Newcastle United",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Luton": "Luton Town",
    "Santander": "Racing Santander",
    "Hamburg": "Hamburger SV",
    "Paderborn": "Paderborn 07",
    # La Liga (football-data abbreviations)
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Club",
    "Betis": "Real Betis",
    "Celta": "Celta Vigo",
    "Espanol": "Espanyol",
    "La Coruna": "Dep. La Coruña",
    "Sociedad": "Real Sociedad",
    "Vallecano": "Rayo Vallecano",
    # Bundesliga
    "M'gladbach": "Gladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "Köln",
    "Darmstadt": "Darmstadt 98",
    "Greuther Fuerth": "Greuther Fürth",
    "SD Huesca": "Huesca",
    # Serie A
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
}

ODDS_DIR = PROJECT_ROOT / "odds"
FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_CSV_PATH = ODDS_DIR / "fixtures.csv"
DIV_TO_LEAGUE = {v: k for k, v in FOOTBALL_DATA_DIV.items()}


def _fd_team(name: str) -> str | None:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = str(name).strip()
    text = FD_TEAM_ALIASES.get(text, text)
    return normalize_team_name(text)


def _season_code(end_year: int) -> str:
    y = int(end_year)
    return f"{(y - 1) % 100:02d}{y % 100:02d}"


def _local_csv_path(season_end: int, div: str) -> Path:
    return ODDS_DIR / f"{season_end}_{div.lower()}.csv"


def _download_div_season(div: str, code: str) -> pd.DataFrame | None:
    url = f"https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
    try:
        resp = requests.get(url, timeout=45)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        return pd.read_csv(StringIO(resp.text))
    except Exception:
        log.debug("football-data fetch failed %s", url, exc_info=True)
        return None


def sync_fixtures_csv() -> Path | None:
    """Cache upcoming odds from football-data ``fixtures.csv``."""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(FIXTURES_CSV_URL, timeout=45)
        if resp.status_code != 200 or not resp.text.strip():
            log.warning("fixtures.csv download failed (status %s).", resp.status_code)
            return FIXTURES_CSV_PATH if FIXTURES_CSV_PATH.exists() else None
        FIXTURES_CSV_PATH.write_text(resp.text, encoding="utf-8")
        log.info("Saved football-data fixtures.csv.")
        return FIXTURES_CSV_PATH
    except Exception:
        log.warning("fixtures.csv download failed.", exc_info=True)
        return FIXTURES_CSV_PATH if FIXTURES_CSV_PATH.exists() else None


def load_fixtures_frame(leagues: list[str] | None = None) -> pd.DataFrame:
    leagues = leagues or active_leagues()
    if not FIXTURES_CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(FIXTURES_CSV_PATH)
    if df.empty or "Div" not in df.columns:
        return pd.DataFrame()
    df = df[df["Div"].isin(DIV_TO_LEAGUE)].copy()
    df["Comp"] = df["Div"].map(DIV_TO_LEAGUE)
    df = df[df["Comp"].isin(leagues)]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["HomeTeam"] = df["HomeTeam"].map(_fd_team)
    df["AwayTeam"] = df["AwayTeam"].map(_fd_team)
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "Comp"])


def sync_football_data_csvs(
    leagues: list[str] | None = None,
    season_end_years: list[int] | None = None,
) -> list[Path]:
    """Cache football-data league CSVs under ``odds/{year}_{div}.csv``."""
    leagues = leagues or active_leagues()
    if season_end_years is None:
        season_end_years = model_season_window()
    saved: list[Path] = []
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    for league in leagues:
        div = FOOTBALL_DATA_DIV.get(league)
        if not div:
            continue
        current_end = current_season_end()
        for year in sorted({int(y) for y in season_end_years}):
            path = _local_csv_path(year, div)
            needs_fetch = int(year) >= current_end
            if not needs_fetch and path.exists():
                try:
                    head = pd.read_csv(path, nrows=2)
                    needs_fetch = "HxG" not in head.columns
                except Exception:
                    needs_fetch = True
            if not needs_fetch:
                continue
            df = _download_div_season(div, _season_code(year))
            if df is None or df.empty:
                continue
            if "HxG" not in df.columns:
                log.debug(
                    "football-data %s %s has no HxG column; skipping cache.",
                    div,
                    year,
                )
                continue
            df.to_csv(path, index=False)
            saved.append(path)
            log.info("Saved football-data %s (%d rows).", path.name, len(df))
    sync_fixtures_csv()
    return saved


def _load_fd_frames(leagues: list[str], season_years: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        div = FOOTBALL_DATA_DIV.get(league)
        if not div:
            continue
        for year in season_years:
            path = _local_csv_path(year, div)
            df = None
            if path.exists():
                df = pd.read_csv(path)
            else:
                df = _download_div_season(div, _season_code(year))
                if df is not None and not df.empty and "HxG" in df.columns:
                    ODDS_DIR.mkdir(parents=True, exist_ok=True)
                    df.to_csv(path, index=False)
            if df is None or df.empty or "HxG" not in df.columns:
                continue
            df = df.copy()
            df["Comp"] = league
            df["Season"] = year
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out["HomeTeam"] = out["HomeTeam"].map(_fd_team)
    out["AwayTeam"] = out["AwayTeam"].map(_fd_team)
    for col in ("HxG", "AxG"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Date", "HomeTeam", "AwayTeam"])


def _fixture_lookup(fd: pd.DataFrame) -> dict[tuple, tuple[float, float]]:
    lookup: dict[tuple, tuple[float, float]] = {}
    for _, row in fd.iterrows():
        if pd.isna(row["HxG"]) or pd.isna(row["AxG"]):
            continue
        key = (
            row["Date"].date().isoformat(),
            row["HomeTeam"],
            row["AwayTeam"],
        )
        lookup[key] = (float(row["HxG"]), float(row["AxG"]))
    return lookup


def _pair_xg_near_date(
    lookup: dict[tuple, tuple[float, float]],
    team: str,
    opponent: str,
    dt: pd.Timestamp,
    max_days: int = 14,
) -> tuple[float, float] | None:
    pair = frozenset({team, opponent})
    best: tuple[float, float] | None = None
    best_gap = max_days + 1
    for (day_s, home, away), (hxg, axg) in lookup.items():
        if frozenset({home, away}) != pair:
            continue
        gap = abs((pd.Timestamp(day_s) - dt).days)
        if gap > max_days or gap >= best_gap:
            continue
        vals = _team_xg_from_fixture(team, opponent, hxg, axg, home, away)
        if vals:
            best = vals
            best_gap = gap
    return best


def _team_xg_from_fixture(
    team: str,
    opponent: str,
    hxg: float,
    axg: float,
    home: str,
    away: str,
) -> tuple[float, float] | None:
    if team == home and opponent == away:
        return hxg, axg
    if team == away and opponent == home:
        return axg, hxg
    return None


def enrich_team_match_xg(
    df: pd.DataFrame,
    lookup: dict[tuple, tuple[float, float]],
    *,
    only_missing: bool = True,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    if "xGA" not in out.columns:
        out["xGA"] = pd.NA
    filled = 0
    for idx, row in out.iterrows():
        need_xg = pd.isna(row.get("xG"))
        need_xga = pd.isna(row.get("xGA"))
        if only_missing and not need_xg and not need_xga:
            continue
        team = normalize_team_name(row.get("Team"))
        opp = normalize_team_name(row.get("Opponent"))
        if not team or not opp:
            continue
        dt = pd.Timestamp(row["Date"]).normalize()
        pair_vals = None
        for day_offset in range(-3, 4):
            dt_key = (dt + pd.Timedelta(days=day_offset)).date().isoformat()
            for home, away in ((team, opp), (opp, team)):
                key = (dt_key, home, away)
                if key not in lookup:
                    continue
                hxg, axg = lookup[key]
                pair_vals = _team_xg_from_fixture(team, opp, hxg, axg, home, away)
                if pair_vals:
                    break
            if pair_vals:
                break
        if not pair_vals:
            pair_vals = _pair_xg_near_date(lookup, team, opp, dt)
        if not pair_vals:
            continue
        xg_val, xga_val = pair_vals
        if need_xg:
            out.at[idx, "xG"] = xg_val
        if need_xga:
            out.at[idx, "xGA"] = xga_val
        filled += 1
    if filled:
        log.info("Filled xG/xGA on %d team-match rows from football-data.", filled)
    return out, filled


def enrich_history_xg(history: pd.DataFrame, lookup: dict[tuple, tuple[float, float]]) -> pd.DataFrame:
    if history.empty:
        return history
    out = history.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    if "xG" not in out.columns:
        out["xG"] = pd.NA
    if "xG.1" not in out.columns:
        out["xG.1"] = pd.NA
    for idx, row in out.iterrows():
        home = _fd_team(row.get("Home"))
        away = _fd_team(row.get("Away"))
        if not home or not away:
            continue
        dt = pd.Timestamp(row["Date"]).date().isoformat()
        key = (dt, home, away)
        if key not in lookup:
            continue
        hxg, axg = lookup[key]
        if pd.isna(row.get("xG")):
            out.at[idx, "xG"] = hxg
        if pd.isna(row.get("xG.1")):
            out.at[idx, "xG.1"] = axg
    return out


def enrich_matches_and_history(
    matches: pd.DataFrame,
    history: pd.DataFrame | None = None,
    *,
    leagues: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    leagues = leagues or active_leagues()
    seasons = set()
    if not matches.empty and "Season" in matches.columns:
        seasons |= {int(s) for s in matches["Season"].dropna().unique()}
    if history is not None and not history.empty and "Season" in history.columns:
        seasons |= {int(s) for s in history["Season"].dropna().unique()}
    if not seasons:
        seasons = set(model_season_window())
    fd = _load_fd_frames(leagues, seasons)
    if fd.empty:
        log.warning("No football-data xG frames loaded; xG columns unchanged.")
        return matches, history
    lookup = _fixture_lookup(fd)
    matches_out, _ = enrich_team_match_xg(matches, lookup)
    history_out = enrich_history_xg(history, lookup) if history is not None else None
    return matches_out, history_out


def enrich_csv_files() -> dict[str, float | int]:
    """Football-data xG only (weekly path)."""
    from fplpredict.data.xg_enrich import enrich_csv_files as _enrich

    return _enrich(use_understat=False)


_FTR_HOME_AWAY = {"H": ("W", "L"), "A": ("L", "W"), "D": ("D", "D")}
_RESULT_CHAR_TO_TARGET = {"D": 0, "L": 1, "W": 2}


def _load_fd_results_frames(leagues: list[str], season_years: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        div = FOOTBALL_DATA_DIV.get(league)
        if not div:
            continue
        for year in season_years:
            path = _local_csv_path(year, div)
            df = None
            if path.exists():
                df = pd.read_csv(path)
            else:
                df = _download_div_season(div, _season_code(year))
                if df is not None and not df.empty:
                    ODDS_DIR.mkdir(parents=True, exist_ok=True)
                    df.to_csv(path, index=False)
            if df is None or df.empty or "FTR" not in df.columns:
                continue
            df = df.copy()
            df["Comp"] = league
            df["Season"] = year
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out["HomeTeam"] = out["HomeTeam"].map(_fd_team)
    out["AwayTeam"] = out["AwayTeam"].map(_fd_team)
    for col in ("FTHG", "FTAG"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["FTR"] = out["FTR"].astype(str).str.strip().str.upper()
    return out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])


def _fd_result_near_date(
    fd: pd.DataFrame,
    team: str,
    opponent: str,
    dt: pd.Timestamp,
    max_days: int = 3,
) -> tuple[str, int, int] | None:
    """Return (W/L/D result char, GF, GA) for ``team`` vs ``opponent`` near ``dt``."""
    pair = frozenset({team, opponent})
    best: tuple[str, int, int] | None = None
    best_gap = max_days + 1
    for _, row in fd.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        if frozenset({home, away}) != pair:
            continue
        gap = abs((pd.Timestamp(row["Date"]).normalize() - dt.normalize()).days)
        if gap > max_days or gap >= best_gap:
            continue
        ftr = str(row["FTR"]).strip().upper()
        if ftr not in _FTR_HOME_AWAY:
            continue
        home_res, away_res = _FTR_HOME_AWAY[ftr]
        fthg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
        ftag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
        if team == home:
            best = (home_res, fthg, ftag)
        elif team == away:
            best = (away_res, ftag, fthg)
        else:
            continue
        best_gap = gap
    return best


def backfill_next_match_results_from_football_data(
    next_matches: pd.DataFrame,
    *,
    leagues: list[str] | None = None,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fill Result/GF/GA on past unscored rows when FBref lags football-data."""
    if next_matches.empty or "Result" not in next_matches.columns:
        return next_matches, 0
    leagues = leagues or active_leagues()
    as_of = (as_of or pd.Timestamp.now()).normalize()
    seasons: set[int] = set()
    if "Season" in next_matches.columns:
        seasons |= {int(s) for s in next_matches["Season"].dropna().unique()}
    seasons |= set(model_season_window())
    fd = _load_fd_results_frames(leagues, seasons)
    if fd.empty:
        return next_matches, 0

    out = next_matches.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    filled = 0
    for idx, row in out.iterrows():
        if pd.notna(row.get("Result")) and str(row.get("Result")).strip():
            continue
        if pd.Timestamp(row["Date"]).normalize() >= as_of:
            continue
        if row.get("Comp") not in leagues:
            continue
        team = normalize_team_name(row.get("Team"))
        opp = normalize_team_name(row.get("Opponent"))
        if not team or not opp:
            continue
        hit = _fd_result_near_date(fd, team, opp, pd.Timestamp(row["Date"]))
        if not hit:
            continue
        res_char, gf, ga = hit
        out.at[idx, "Result"] = res_char
        out.at[idx, "GF"] = gf
        out.at[idx, "GA"] = ga
        filled += 1
    if filled:
        log.info("football-data filled Result on %d next_match row(s).", filled)
    return out, filled


def completed_results_from_football_data(
    matches: pd.DataFrame,
    league: str,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Played fixtures from football-data when ``matches.csv`` / FBref lag."""
    from fplpredict.model.features import build_team_name_mapping

    as_of = (as_of or pd.Timestamp.now()).normalize()
    seasons: set[int] = set(model_season_window())
    if not matches.empty and "Season" in matches.columns:
        sub = matches.loc[matches["Comp"] == league, "Season"].dropna()
        seasons |= {int(s) for s in sub.unique()}
    fd = _load_fd_results_frames([league], seasons)
    if fd.empty:
        return pd.DataFrame(columns=["Date", "Team", "Opponent", "Target"])

    mapping = build_team_name_mapping(matches, league=league) if not matches.empty else {}
    rows: list[dict] = []
    for _, row in fd.iterrows():
        if pd.Timestamp(row["Date"]).normalize() >= as_of:
            continue
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        if not home or not away:
            continue
        ftr = str(row["FTR"]).strip().upper()
        if ftr not in _FTR_HOME_AWAY:
            continue
        home_res, away_res = _FTR_HOME_AWAY[ftr]
        fthg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
        ftag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
        dt = pd.Timestamp(row["Date"])
        for team, opp, res, gf, ga in (
            (home, away, home_res, fthg, ftag),
            (away, home, away_res, ftag, fthg),
        ):
            team_n = normalize_team_name(mapping.get(team, team)) or team
            opp_n = normalize_team_name(mapping.get(opp, opp)) or opp
            rows.append(
                {
                    "Date": dt,
                    "Team": team_n,
                    "Opponent": opp_n,
                    "Target": _RESULT_CHAR_TO_TARGET[res],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Date", "Team", "Opponent", "Target"])
    out = pd.DataFrame(rows).drop_duplicates(
        subset=["Date", "Team", "Opponent"], keep="last"
    )
    return out.reset_index(drop=True)
