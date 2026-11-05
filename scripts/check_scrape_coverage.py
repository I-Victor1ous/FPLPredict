#!/usr/bin/env python3
"""Run scrape/cleaning unit tests, then report per-season coverage of data/matches.csv.

  PYTHONPATH=. python scripts/check_scrape_coverage.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TESTS = [
    "tests/test_scrape_team_logs.py",
    "tests/test_matches_clean.py",
    "tests/test_standings.py",
]


def coverage_report() -> None:
    from fplpredict.data.matches_clean import LEAGUE_TEAM_SLOTS, expected_club_count
    from fplpredict.data_io import read_csv

    matches = read_csv("matches.csv")
    matches["Season"] = pd.to_numeric(matches["Season"], errors="coerce")
    print("\nPer league and season: clubs | fixtures | one-sided fixtures | xG coverage")
    for comp in sorted(LEAGUE_TEAM_SLOTS):
        sub = matches[matches["Comp"] == comp]
        if sub.empty:
            continue
        print(f"  {comp} (expect {LEAGUE_TEAM_SLOTS[comp]} clubs)")
        for season, grp in sub.groupby("Season"):
            expected = expected_club_count(comp, season)
            clubs = set(grp["Team"]) | set(grp["Opponent"])
            rows = set(zip(grp["Date"], grp["Team"], grp["Opponent"]))
            fixtures = {(d, frozenset((t, o))) for d, t, o in rows}
            one_sided = sum(1 for d, t, o in rows if (d, o, t) not in rows)
            xg = 1 - float(grp["xG"].isna().mean()) if "xG" in grp.columns else 0.0
            flag = "" if len(clubs) == expected else "  <-- club count off"
            print(
                f"    {int(season)}: {len(clubs):>2} clubs | {len(fixtures):>3} fixtures | "
                f"{one_sided:>3} one-sided | xG {xg:.0%}{flag}"
            )


def main() -> int:
    print("Running scrape/cleaning/standings unit tests…")
    proc = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q"], cwd=ROOT)
    if proc.returncode != 0:
        return proc.returncode
    coverage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
