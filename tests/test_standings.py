import numpy as np
import pandas as pd

from fplpredict.model.standings import pre_match_positions


def _both(rows):
    """Expand (date, home, away, hg, ag, season) into both team perspectives."""
    out = []
    for date, home, away, hg, ag, season in rows:
        out.append(
            {"Comp": "L", "Season": season, "Date": date, "Team": home, "Opponent": away, "GF": hg, "GA": ag}
        )
        out.append(
            {"Comp": "L", "Season": season, "Date": date, "Team": away, "Opponent": home, "GF": ag, "GA": hg}
        )
    return pd.DataFrame(out)


def _pos(df, positions, date, team):
    idx = df[(df["Date"] == date) & (df["Team"] == team)].index[0]
    return positions.at[idx, "Team_position"]


def test_blank_before_first_completed_match_and_no_double_count():
    df = _both(
        [
            ("2026-08-01", "A", "B", 2, 0, 2027),
            ("2026-08-01", "C", "D", 1, 1, 2027),
            ("2026-08-08", "A", "C", 0, 0, 2027),
            ("2026-08-08", "B", "D", 3, 0, 2027),
        ]
    )
    pos = pre_match_positions(df)
    assert np.isnan(_pos(df, pos, "2026-08-01", "A"))
    # After week 1: A 3pts, C/D 1pt (C before D by name), B 0pts.
    assert _pos(df, pos, "2026-08-08", "A") == 1
    assert _pos(df, pos, "2026-08-08", "C") == 2
    assert _pos(df, pos, "2026-08-08", "D") == 3
    assert _pos(df, pos, "2026-08-08", "B") == 4


def test_new_season_resets_and_promoted_club_is_ranked():
    df = _both(
        [
            ("2026-05-01", "A", "B", 5, 0, 2026),
            ("2026-08-01", "A", "P", 0, 1, 2027),
            ("2026-08-01", "B", "C", 0, 0, 2027),
            ("2026-08-08", "P", "B", 0, 0, 2027),
            ("2026-08-08", "A", "C", 0, 0, 2027),
        ]
    )
    pos = pre_match_positions(df)
    assert np.isnan(_pos(df, pos, "2026-08-01", "A"))
    assert _pos(df, pos, "2026-08-08", "P") == 1
    assert _pos(df, pos, "2026-08-08", "A") == 4


def test_postponed_game_counts_by_date_and_upcoming_uses_full_table():
    df = _both(
        [
            ("2026-08-01", "A", "B", 1, 0, 2027),
            ("2026-08-20", "C", "D", 4, 0, 2027),  # matchweek 1 game played late
            ("2026-08-10", "A", "C", 0, 0, 2027),
        ]
    )
    upcoming = pd.DataFrame(
        [{"Comp": "L", "Season": 2027, "Date": "2026-08-30", "Team": "D", "Opponent": "A", "GF": np.nan, "GA": np.nan}]
    )
    df = pd.concat([df, upcoming], ignore_index=True)
    pos = pre_match_positions(df)
    # On 08-10 only A-B is complete: C has 0 points, ranked by name behind B? B has GD -1.
    assert _pos(df, pos, "2026-08-10", "A") == 1
    assert _pos(df, pos, "2026-08-10", "C") == 2
    # Upcoming: A 4pts, C 4pts (+4 GD) -> C first.
    up_idx = df.index[-1]
    assert pos.at[up_idx, "Opp_position"] == 2
    assert pos.at[up_idx, "Team_position"] == 4
