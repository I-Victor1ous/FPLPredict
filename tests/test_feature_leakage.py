"""Pre-match features must not change when that match's own result changes."""

import pandas as pd
import pytest

from fplpredict.config import CANDIDATE_FEATURES
from fplpredict.model.features import build_feature_matrix
from tests.synthetic import LEAGUE, make_league

KEY = ["Date", "Team", "Opponent"]


@pytest.fixture(scope="module")
def frames():
    matches, upcoming, history = make_league(seed=3)
    target = matches[(matches["Season"] == 2021) & (matches["Round"] == "Matchweek 6")].iloc[0]
    fixture = (target["Date"], {target["Team"], target["Opponent"]})

    altered = matches.copy()
    is_fixture = (altered["Date"] == fixture[0]) & altered["Team"].isin(fixture[1]) & altered[
        "Opponent"
    ].isin(fixture[1])
    for idx in altered[is_fixture].index:
        team_side = altered.at[idx, "Team"] == target["Team"]
        altered.at[idx, "GF"] = 7 if team_side else 0
        altered.at[idx, "GA"] = 0 if team_side else 7
        altered.at[idx, "Result"] = "W" if team_side else "L"
        altered.at[idx, "xG"] = 5.0 if team_side else 0.1
        altered.at[idx, "xGA"] = 0.1 if team_side else 5.0
        altered.at[idx, "Sh"] = 30 if team_side else 1

    base, _, _ = build_feature_matrix(matches, upcoming, history, league=LEAGUE)
    changed, _, _ = build_feature_matrix(altered, upcoming, history, league=LEAGUE)
    return base, changed, pd.Timestamp(fixture[0])


def test_features_on_or_before_match_day_ignore_its_result(frames):
    base, changed, match_day = frames
    cols = [c for c in CANDIDATE_FEATURES if c in base.columns]
    assert cols, "no candidate features built"
    merged = base[base["Date"] <= match_day].merge(
        changed, on=KEY, suffixes=("", "_alt"), validate="one_to_one"
    )
    assert len(merged) == (base["Date"] <= match_day).sum()
    for col in cols:
        pd.testing.assert_series_equal(
            merged[col], merged[f"{col}_alt"], check_names=False, obj=f"feature {col}"
        )


def test_result_does_flow_into_later_matches(frames):
    """Guards the test above against comparing frames that never differ."""
    base, changed, match_day = frames
    later = base[base["Date"] > match_day].merge(changed, on=KEY, suffixes=("", "_alt"))
    assert (later["GFleague_rolling"] != later["GFleague_rolling_alt"]).any()
