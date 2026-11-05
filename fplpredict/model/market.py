"""Bookmaker baseline: margin-free 1X2 probabilities from football-data average odds."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from fplpredict.data.football_data import (
    FOOTBALL_DATA_DIV,
    ODDS_DIR,
    _download_div_season,
    _fd_team,
    _local_csv_path,
    _season_code,
    load_fixtures_frame,
)
from fplpredict.model.metrics import ranked_probability_score
from fplpredict.teams import normalize_team_name

log = logging.getLogger(__name__)

ODDS_COLUMNS = ("AvgH", "AvgD", "AvgA")
MARKET_CACHE_DIR = ODDS_DIR / "market"
DATE_TOLERANCE_DAYS = 3


def implied_probabilities(home: np.ndarray, draw: np.ndarray, away: np.ndarray) -> np.ndarray:
    """(n, 3) [home, draw, away] probabilities with the bookmaker margin removed proportionally."""
    raw = 1.0 / np.column_stack([home, draw, away]).astype(float)
    return raw / raw.sum(axis=1, keepdims=True)


def _season_odds_frame(league: str, season_end: int) -> pd.DataFrame | None:
    div = FOOTBALL_DATA_DIV.get(league)
    if not div:
        return None
    cached = _local_csv_path(season_end, div)
    market_path = MARKET_CACHE_DIR / cached.name
    for path in (cached, market_path):
        if path.exists():
            df = pd.read_csv(path)
            if set(ODDS_COLUMNS) <= set(df.columns):
                return df
    df = _download_div_season(div, _season_code(season_end))
    if df is None or df.empty or not set(ODDS_COLUMNS) <= set(df.columns):
        return None
    MARKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(market_path, index=False)
    return df


def fixtures_market_lookup(league: str) -> dict[tuple[str, str, str], np.ndarray]:
    """Upcoming fixtures from cached ``fixtures.csv``."""
    df = load_fixtures_frame([league])
    if df.empty or not set(ODDS_COLUMNS) <= set(df.columns):
        return {}
    for col in ODDS_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=ODDS_COLUMNS)
    probs = implied_probabilities(df["AvgH"], df["AvgD"], df["AvgA"])
    return {
        (d.date().isoformat(), h, a): p
        for d, h, a, p in zip(df["Date"], df["HomeTeam"], df["AwayTeam"], probs)
    }


def combined_market_lookup(league: str, season_ends: set[int]) -> dict[tuple[str, str, str], np.ndarray]:
    lookup: dict[tuple[str, str, str], np.ndarray] = {}
    for year in season_ends:
        lookup.update(market_lookup(league, int(year)))
    lookup.update(fixtures_market_lookup(league))
    return lookup


def attach_market_implied_features(df: pd.DataFrame, league: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    seasons = {int(s) for s in pd.to_numeric(out["Season"], errors="coerce").dropna().unique()}
    lookup = combined_market_lookup(league, seasons)
    win, draw, loss = [], [], []
    for _, row in out.iterrows():
        p = _row_market_proba(row, lookup)
        if p is None:
            win.append(np.nan)
            draw.append(np.nan)
            loss.append(np.nan)
        else:
            draw.append(float(p[0]))
            loss.append(float(p[1]))
            win.append(float(p[2]))
    out["Market_implied_draw"] = draw
    out["Market_implied_loss"] = loss
    out["Market_implied_win"] = win
    return out


def market_probs_dict(row: pd.Series, lookup: dict) -> dict[str, float] | None:
    p = _row_market_proba(row, lookup)
    if p is None:
        return None
    return {
        "teamWin": round(float(p[2]) * 100, 2),
        "draw": round(float(p[0]) * 100, 2),
        "oppWin": round(float(p[1]) * 100, 2),
    }


def market_lookup(league: str, season_end: int) -> dict[tuple[str, str, str], np.ndarray]:
    """(date, home, away) -> margin-free [home, draw, away] probabilities."""
    df = _season_odds_frame(league, season_end)
    if df is None:
        return {}
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["HomeTeam"] = df["HomeTeam"].map(_fd_team)
    df["AwayTeam"] = df["AwayTeam"].map(_fd_team)
    for col in ODDS_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", *ODDS_COLUMNS])
    probs = implied_probabilities(df["AvgH"], df["AvgD"], df["AvgA"])
    return {
        (d.date().isoformat(), h, a): p
        for d, h, a, p in zip(df["Date"], df["HomeTeam"], df["AwayTeam"], probs)
    }


def _row_market_proba(row: pd.Series, lookup: dict) -> np.ndarray | None:
    """Team-perspective probabilities in sklearn order [Draw, Loss, Win]."""
    team = normalize_team_name(row["Team"])
    opp = normalize_team_name(row["Opponent"])
    home_side = bool(row.get("is_home", 1))
    home, away = (team, opp) if home_side else (opp, team)
    day = pd.Timestamp(row["Date"]).normalize()
    for offset in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1):
        key = ((day + pd.Timedelta(days=offset)).date().isoformat(), home, away)
        p = lookup.get(key)
        if p is not None:
            p_home, p_draw, p_away = p
            return np.array(
                [p_draw, p_away, p_home] if home_side else [p_draw, p_home, p_away]
            )
    return None


def market_baseline(
    test_df: pd.DataFrame,
    model_proba: np.ndarray,
    league: str,
    lookup: dict | None = None,
) -> dict | None:
    """Score bookmakers and the model on the test rows that have odds."""
    if test_df.empty:
        return None
    test_n = int(len(test_df))
    if lookup is None:
        seasons = {
            int(s) for s in pd.to_numeric(test_df["Season"], errors="coerce").dropna().unique()
        }
        lookup = combined_market_lookup(league, seasons)
    if not lookup:
        return None
    rows, market = [], []
    for i, (_, row) in enumerate(test_df.iterrows()):
        p = _row_market_proba(row, lookup)
        if p is not None:
            rows.append(i)
            market.append(p)
    if not rows:
        return None
    y = test_df["Target"].astype(int).to_numpy()[rows]
    market_p = np.vstack(market)
    model_p = model_proba[rows]
    labels = [0, 1, 2]
    return {
        "test_n": test_n,
        "n": len(rows),
        "coverage": round(len(rows) / test_n, 4),
        "accuracy": float(accuracy_score(y, market_p.argmax(axis=1))),
        "log_loss": float(log_loss(y, market_p, labels=labels)),
        "rps": ranked_probability_score(y, market_p, sklearn_order=True),
        "model_accuracy": float(accuracy_score(y, model_p.argmax(axis=1))),
        "model_log_loss": float(log_loss(y, np.clip(model_p, 1e-15, 1), labels=labels)),
        "model_rps": ranked_probability_score(y, model_p, sklearn_order=True),
    }
