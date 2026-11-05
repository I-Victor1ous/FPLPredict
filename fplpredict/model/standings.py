"""League table positions computed locally from completed matches, per season."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _rank(table: dict[str, dict[str, int]], clubs: set[str]) -> dict[str, int]:
    rows = sorted(
        clubs,
        key=lambda c: (-table[c]["points"], -table[c]["goal_diff"], -table[c]["GF"], c),
    )
    return {club: pos for pos, club in enumerate(rows, start=1)}


def _unique_completed_fixtures(season_rows: pd.DataFrame) -> list[tuple]:
    """(date, team, opponent, GF, GA) once per fixture, whichever perspective came first."""
    seen: set[tuple] = set()
    out: list[tuple] = []
    done = season_rows[season_rows["GF"].notna() & season_rows["GA"].notna()]
    if "Target" in done.columns:
        done = done[done["Target"].notna()]
    for d, t, o, gf, ga in zip(done["Date"], done["Team"], done["Opponent"], done["GF"], done["GA"]):
        key = (d, frozenset((t, o)))
        if key in seen:
            continue
        seen.add(key)
        out.append((d, t, o, int(gf), int(ga)))
    return sorted(out, key=lambda x: x[0])


def pre_match_positions(rows: pd.DataFrame) -> pd.DataFrame:
    """Return ``Team_position`` / ``Opp_position`` for every row (completed or upcoming).

    Position = rank in that ``(Comp, Season)`` table built from completed matches dated
    strictly before the row's date. Blank until the season has a completed match.
    """
    out = pd.DataFrame(index=rows.index, columns=["Team_position", "Opp_position"], dtype=float)
    if rows.empty:
        return out
    frame = rows.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    group_cols = [c for c in ("Comp", "Season") if c in frame.columns]
    groups = frame.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), frame)]

    for _key, season_rows in groups:
        clubs = set(season_rows["Team"].dropna()) | set(season_rows["Opponent"].dropna())
        fixtures = _unique_completed_fixtures(season_rows)
        table: dict[str, dict[str, int]] = defaultdict(
            lambda: {"points": 0, "goal_diff": 0, "GF": 0}
        )
        snapshots: list[tuple[pd.Timestamp, dict[str, int]]] = []
        i = 0
        for match_date in sorted(season_rows["Date"].unique()):
            while i < len(fixtures) and fixtures[i][0] < match_date:
                _d, team, opp, gf, ga = fixtures[i]
                for club, scored, conceded in ((team, gf, ga), (opp, ga, gf)):
                    table[club]["GF"] += scored
                    table[club]["goal_diff"] += scored - conceded
                    if scored > conceded:
                        table[club]["points"] += 3
                    elif scored == conceded:
                        table[club]["points"] += 1
                i += 1
            ranking = _rank(table, clubs) if i > 0 else {}
            snapshots.append((match_date, ranking))

        by_date = dict(snapshots)
        for idx, d, t, o in zip(
            season_rows.index, season_rows["Date"], season_rows["Team"], season_rows["Opponent"]
        ):
            ranking = by_date[d]
            out.at[idx, "Team_position"] = ranking.get(t, np.nan)
            out.at[idx, "Opp_position"] = ranking.get(o, np.nan)
    return out
