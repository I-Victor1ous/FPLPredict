"""Small synthetic league in the scraped CSV schema, for feature and pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

LEAGUE = "Premier League"
TEAMS = [f"Club {c}" for c in "ABCDEFGH"]
SEASONS = (2019, 2020, 2021, 2022, 2023)


def _round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Circle-method double round robin: list of rounds of (home, away)."""
    n = len(teams)
    rotation = list(teams)
    rounds = []
    for _ in range(n - 1):
        pairs = [(rotation[i], rotation[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return rounds + [[(a, h) for h, a in rnd] for rnd in rounds]


def _row(date, season, rnd, team, opp, venue, gf, ga, xg, xga, sh, sot):
    result = "W" if gf > ga else ("L" if gf < ga else "D")
    return {
        "Date": date.date().isoformat(),
        "Time": "15:00",
        "Comp": LEAGUE,
        "Round": f"Matchweek {rnd}",
        "Day": date.strftime("%a"),
        "Venue": venue,
        "Result": result,
        "GF": gf,
        "GA": ga,
        "Opponent": opp,
        "Season": season,
        "Team": team,
        "Sh": sh,
        "SoT": sot,
        "CrdY": 1.0,
        "CrdR": 0.0,
        "xG": xg,
        "xGA": xga,
    }


def make_league(
    seed: int = 0,
    missing_xg_rate: float = 0.05,
    upcoming_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (matches, next_matches, history) frames shaped like the scraped CSVs.

    Dates are shifted so the single upcoming round falls on ``upcoming_date``
    (default: three days from today) and every other round is already played.
    """
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(0.6, 2.0, len(TEAMS))))
    n_rounds = 2 * (len(TEAMS) - 1)
    upcoming_date = upcoming_date or pd.Timestamp.now().normalize() + pd.Timedelta(days=3)
    natural_upcoming = pd.Timestamp(f"{SEASONS[-1] - 1}-08-03") + pd.Timedelta(weeks=n_rounds)
    shift = upcoming_date - natural_upcoming
    rows = []
    for season in SEASONS:
        start = pd.Timestamp(f"{season - 1}-08-03") + shift
        for r, fixtures in enumerate(_round_robin(TEAMS), start=1):
            date = start + pd.Timedelta(weeks=r - 1)
            for home, away in fixtures:
                lam_h = strength[home] * 1.15 / strength[away] ** 0.5
                lam_a = strength[away] / strength[home] ** 0.5
                hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
                hxg = float(max(0.05, lam_h + rng.normal(0, 0.3)))
                axg = float(max(0.05, lam_a + rng.normal(0, 0.3)))
                if rng.random() < missing_xg_rate:
                    hxg = axg = np.nan
                hsh, ash = int(rng.poisson(lam_h * 8)), int(rng.poisson(lam_a * 8))
                rows.append(_row(date, season, r, home, away, "Home", hg, ag, hxg, axg, hsh, hsh // 3))
                rows.append(_row(date, season, r, away, home, "Away", ag, hg, axg, hxg, ash, ash // 3))
    matches = pd.DataFrame(rows)

    last = upcoming_date
    upcoming = []
    for home, away in _round_robin(TEAMS)[0]:
        for team, opp, venue in ((home, away, "Home"), (away, home, "Away")):
            upcoming.append(
                {
                    "Date": last.date().isoformat(),
                    "Time": "15:00",
                    "Comp": LEAGUE,
                    "Round": "Matchweek 1",
                    "Day": last.strftime("%a"),
                    "Venue": venue,
                    "Result": np.nan,
                    "GF": np.nan,
                    "GA": np.nan,
                    "Opponent": opp,
                    "Season": SEASONS[-1] + 1,
                    "Team": team,
                }
            )
    return matches, pd.DataFrame(upcoming), pd.DataFrame()
