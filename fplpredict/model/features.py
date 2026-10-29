import copy
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fplpredict.config import LEAGUE as DEFAULT_LEAGUE, ROLLING_FORM_WINDOW
from fplpredict.model.standings import pre_match_positions
from fplpredict.predictions_util import drop_invalid_fixtures
from fplpredict.teams import TEAM_ALIASES, normalize_team_name


class MissingDict(dict):
    def __missing__(self, key):
        return key


def build_team_name_mapping(
    matches: pd.DataFrame, league: str = DEFAULT_LEAGUE
) -> MissingDict:
    """Explicit aliases only — do not guess mappings from asymmetric name lists."""
    return MissingDict(**TEAM_ALIASES)


def get_target_from_goals(row: pd.Series) -> int:
    if row["GF"] > row["GA"]:
        return 2
    if row["GF"] < row["GA"]:
        return 1
    return 0


def rolling_avgs(
    group, cols, new_cols, num_past_matches_considered=ROLLING_FORM_WINDOW
):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(
        num_past_matches_considered, closed="left", min_periods=1
    ).mean()
    group[new_cols] = rolling_stats.fillna(0)
    return group


def rolling_avg_match_rest(group, num_past_matches_considered=3):
    group = group.sort_values("Date")
    group["days_diff"] = group["Date"].diff().dt.days
    rolling_stats = group["days_diff"].rolling(
        num_past_matches_considered, min_periods=1
    ).mean()
    group[f"rolling_rests_{num_past_matches_considered}"] = rolling_stats.fillna(0)
    return group


def rolling_match_results(
    group, name, num_past_matches_considered=ROLLING_FORM_WINDOW
):
    group = group.sort_values("Date")
    group["is_win"] = (group["Target"] == 2).astype(int)
    rolling_stats = group["is_win"].rolling(
        num_past_matches_considered, closed="left", min_periods=1
    ).sum()
    group[f"{name}num_wins_rolling"] = rolling_stats.fillna(0)
    return group


def match_outcome_probs(exp_team, exp_opp, max_goals=10):
    probs = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            probs[i, j] = poisson.pmf(i, exp_team) * poisson.pmf(j, exp_opp)
    p_team_win = np.sum(np.tril(probs, -1))
    p_draw = np.sum(np.diag(probs))
    p_opp_win = np.sum(np.triu(probs, 1))
    return p_team_win, p_draw, p_opp_win


def _merge_hth_asof(
    team_rows: pd.DataFrame,
    hth_rows: pd.DataFrame,
    hth_cols: list[str],
) -> pd.DataFrame:
    """Attach head-to-head rolling stats as-of fixture date for (Team, Opponent)."""
    if team_rows.empty or hth_rows.empty:
        return team_rows
    need = [c for c in hth_cols if c in hth_rows.columns]
    if not need:
        return team_rows
    base = [c for c in ("Team", "Opponent", "Date") if c in hth_rows.columns]
    right = hth_rows[base + [c for c in need if c not in base]].copy()
    left = team_rows.copy()
    left["_row_id"] = np.arange(len(left))
    merged_chunks: list[pd.DataFrame] = []
    for (team, opponent), left_grp in left.groupby(["Team", "Opponent"], sort=False):
        mask = (right["Team"] == team) & (right["Opponent"] == opponent)
        right_grp = right.loc[mask]
        if right_grp.empty:
            merged_chunks.append(left_grp)
            continue
        lg = left_grp.sort_values("Date")
        rg = right_grp.sort_values("Date").drop(
            columns=["Team", "Opponent"], errors="ignore"
        )
        chunk = pd.merge_asof(
            lg,
            rg,
            on="Date",
            direction="backward",
        )
        merged_chunks.append(chunk)
    return (
        pd.concat(merged_chunks, ignore_index=True)
        .sort_values("_row_id")
        .drop(columns="_row_id")
    )


def _merge_opponent_asof(
    team_rows: pd.DataFrame, opp_source: pd.DataFrame, drop_opp_cols: list[str]
) -> pd.DataFrame:
    """Attach opponent rolling stats using latest league row on or before fixture date."""
    if team_rows.empty or opp_source.empty:
        return team_rows
    id_cols = {
        "Date",
        "Team",
        "Comp",
        "Round",
        "GF",
        "GA",
        "Opponent",
        "xG",
        "xGA",
        "Season",
        "Team_code",
        "Opp_code",
        "Venue_code",
        "Hour",
        "Day_code",
        "Target",
        "days_diff",
        "is_win",
        "Venue",
    }
    stat_cols = [c for c in opp_source.columns if c not in id_cols]
    opp = opp_source[["Date", "Team"] + stat_cols].rename(
        columns={"Team": "Opponent", "Date": "Opp_Date"}
    )
    for col in stat_cols:
        opp = opp.rename(columns={col: f"Opp_{col}"})
    left = team_rows.copy()
    left["_row_id"] = np.arange(len(left))
    merged_chunks: list[pd.DataFrame] = []
    for opponent, left_grp in left.groupby("Opponent", sort=False):
        right_grp = opp[opp["Opponent"] == opponent]
        if right_grp.empty:
            merged_chunks.append(left_grp)
            continue
        lg = left_grp.sort_values("Date")
        rg = right_grp.sort_values("Opp_Date").drop(columns=["Opponent"], errors="ignore")
        chunk = pd.merge_asof(
            lg,
            rg,
            left_on="Date",
            right_on="Opp_Date",
            direction="backward",
        )
        merged_chunks.append(chunk)
    merged = (
        pd.concat(merged_chunks, ignore_index=True)
        .sort_values("_row_id")
        .drop(columns="_row_id")
    )
    return merged.drop(columns=drop_opp_cols, errors="ignore")


def get_relative_feature(team_matches: pd.DataFrame, cols: list[str]) -> list[str]:
    relative_cols = []
    for col in cols:
        opp_col = f"Opp_{col}"
        if col in team_matches.columns and opp_col in team_matches.columns:
            rel = f"Relative_{col}"
            team_matches[rel] = team_matches[col] - team_matches[opp_col]
            relative_cols.append(rel)
    return relative_cols


RESULT_TARGET = {"D": 0, "L": 1, "W": 2}


def prepare_raw_frames(
    matches: pd.DataFrame,
    next_matches: pd.DataFrame,
    history_matches: pd.DataFrame,
    league: str = DEFAULT_LEAGUE,
):
    from fplpredict.data.matches_clean import (
        clean_matches_for_modeling,
        clean_next_matches,
    )

    mapping = build_team_name_mapping(matches, league=league)

    matches = clean_matches_for_modeling(matches[matches["Comp"] == league])
    next_matches = clean_next_matches(next_matches[next_matches["Comp"] == league])
    league_teams = set(matches["Team"].dropna()) | set(matches["Opponent"].dropna())
    hist = history_matches.copy()
    if not hist.empty and "Comp" in hist.columns:
        comp_mask = hist["Comp"] == league
        if "Home" in hist.columns and "Away" in hist.columns:
            home_n = hist["Home"].map(lambda x: normalize_team_name(x) or x)
            away_n = hist["Away"].map(lambda x: normalize_team_name(x) or x)
            pair_mask = home_n.isin(league_teams) & away_n.isin(league_teams)
            history_matches = hist[comp_mask | pair_mask].copy()
        else:
            history_matches = hist[comp_mask].copy()
    else:
        history_matches = hist

    matches["Team"] = matches["Team"].map(mapping).apply(
        lambda x: normalize_team_name(x) or x
    )
    matches["Opponent"] = matches["Opponent"].map(mapping).apply(
        lambda x: normalize_team_name(x) or x
    )
    matches = matches[matches["Result"].isin(RESULT_TARGET)].copy()

    team_mapping_dict = {
        key: value
        for key, value in zip(
            matches["Team"], matches["Team"].astype("category").cat.codes
        )
    }
    matches["Team_code"] = matches["Team"].map(team_mapping_dict)
    matches["Opp_code"] = matches["Opponent"].map(team_mapping_dict)
    matches["Date"] = pd.to_datetime(matches["Date"])
    matches["Venue_code"] = matches["Venue"].astype("category").cat.codes
    matches["Hour"] = (
        matches["Time"].str.replace(r":.+", "", regex=True).astype("int")
    )
    matches["Day_code"] = matches["Date"].dt.dayofweek
    matches["Target"] = matches["Result"].map(RESULT_TARGET)
    matches["GF"] = matches["GF"].apply(
        lambda x: int(re.search(r"^[0-9]+", str(x)).group(0))
    )
    matches["GA"] = matches["GA"].apply(
        lambda x: int(re.search(r"^[0-9]+", str(x)).group(0))
    )
    matches = matches.drop(
        columns=[
            "Mirrored",
            "Time",
            "Day",
            "Result",
            "Attendance",
            "Match Report",
            "Notes",
            "Captain",
            "Formation",
            "Opp Formation",
            "Referee",
        ],
        errors="ignore",
    )

    next_matches["Team"] = next_matches["Team"].map(mapping).apply(
        lambda x: normalize_team_name(x) or x
    )
    next_matches["Opponent"] = next_matches["Opponent"].map(mapping).apply(
        lambda x: normalize_team_name(x) or x
    )
    matches = drop_invalid_fixtures(matches)
    next_matches = drop_invalid_fixtures(next_matches)
    next_matches["Team_code"] = next_matches["Team"].map(team_mapping_dict)
    next_matches["Opp_code"] = next_matches["Opponent"].map(team_mapping_dict)
    next_matches["Date"] = pd.to_datetime(next_matches["Date"])
    next_matches["Venue_code"] = next_matches["Venue"].astype("category").cat.codes
    next_matches["Hour"] = (
        next_matches["Time"].str.replace(r":.+", "", regex=True).astype("int")
    )
    next_matches["Day_code"] = next_matches["Date"].dt.dayofweek
    next_matches["Target"] = np.nan
    next_matches = next_matches.drop(
        columns=[
            "Time",
            "Day",
            "Result",
            "Attendance",
            "Match Report",
            "Notes",
            "Captain",
            "Formation",
            "Opp Formation",
            "Referee",
        ],
        errors="ignore",
    )

    if not history_matches.empty:
        history_matches["Team"] = history_matches["Team"].map(mapping).apply(
            lambda x: normalize_team_name(x) or x
        )
        history_matches["Home"] = history_matches["Home"].map(mapping).apply(
            lambda x: normalize_team_name(x) or x
        )
        history_matches["Away"] = history_matches["Away"].map(mapping).apply(
            lambda x: normalize_team_name(x) or x
        )
        history_matches["Date"] = pd.to_datetime(history_matches["Date"])
        history_matches["GF"] = history_matches["Score"].apply(
            lambda score: score.split("–")[0]
        )
        history_matches["GA"] = history_matches["Score"].apply(
            lambda score: score.split("–")[1]
        )
        history_matches["GF"] = history_matches["GF"].apply(
            lambda x: int(re.search(r"[0-9]+$", x).group(0))
        )
        history_matches["GA"] = history_matches["GA"].apply(
            lambda x: int(re.search(r"^[0-9]+", x).group(0))
        )
        history_matches["Target"] = history_matches.apply(get_target_from_goals, axis=1)

        for index, row in history_matches.iterrows():
            if row["Team"] == row["Away"]:
                history_matches.loc[index, "Home"] = row["Away"]
                history_matches.loc[index, "Away"] = row["Home"]
                history_matches.loc[index, "GF"] = row["GA"]
                history_matches.loc[index, "GA"] = row["GF"]

        history_matches["Opponent"] = history_matches["Away"]
        history_matches = history_matches.drop(
            columns=["Time", "Day", "Venue", "Attendance", "Notes", "Referee", "Home", "Away"],
            errors="ignore",
        )

    numeric_cols = [
        "Sh",
        "SoT",
        "Dist",
        "FK",
        "PK",
        "PKatt",
        "CrdY",
        "CrdR",
        "2CrdY",
        "Fls",
        "Fld",
        "Off",
        "Crs",
        "Int",
        "TklW",
        "Won%",
        "SCA",
        "GCA",
    ]
    # Unknown stats (no football-data xG, mirrored rows) stay blank; rolling means skip blanks.
    numeric_cols += ["xG", "xGA", "xG.1"]
    for frame in (matches, next_matches, history_matches):
        for col in numeric_cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return matches, next_matches, history_matches, team_mapping_dict


def build_feature_matrix(
    matches: pd.DataFrame,
    next_matches: pd.DataFrame,
    history_matches: pd.DataFrame,
    league: str = DEFAULT_LEAGUE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    matches, next_matches, history_matches, team_mapping_dict = prepare_raw_frames(
        matches, next_matches, history_matches, league=league
    )

    league_ravg_cols = [
        "GF",
        "GA",
        "Sh",
        "SoT",
        "CrdY",
        "CrdR",
        "xG",
        "xGA",
    ]
    league_ravg_new_cols = [f"{c}league_rolling" for c in league_ravg_cols]
    last_ravg_cols = league_ravg_cols + ["Target"]
    last_ravg_new_cols = [f"{c}last_rolling" for c in last_ravg_cols]
    hth_ravg_cols = ["GF", "GA"]
    hth_ravg_new_cols = [f"{c}hth_rolling" for c in hth_ravg_cols]

    all_matches = pd.concat([matches, next_matches], ignore_index=True)
    all_matches = (
        all_matches.groupby("Team", group_keys=True)
        .apply(lambda x: rolling_avg_match_rest(x), include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    all_matches = (
        all_matches.groupby("Team", group_keys=True)
        .apply(lambda x: rolling_avg_match_rest(x, 1), include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    league_matches = all_matches[all_matches["Comp"] == league]
    team_league_matches = (
        league_matches.groupby("Team", group_keys=True)
        .apply(
            lambda x: rolling_avgs(x, league_ravg_cols, league_ravg_new_cols),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    team_league_matches = team_league_matches.sort_values(["Team", "Date"])
    team_league_matches = (
        team_league_matches.groupby("Team", group_keys=True)
        .apply(lambda x: rolling_match_results(x, "league"), include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    team_last_matches = (
        all_matches.groupby("Team", group_keys=True)
        .apply(
            lambda x: rolling_avgs(x, last_ravg_cols, last_ravg_new_cols, 1),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    hth_and_next = pd.concat([history_matches, next_matches], ignore_index=True)
    team_hth_matches = (
        hth_and_next.groupby(["Team", "Opponent"], group_keys=True)
        .apply(
            lambda x: rolling_avgs(x, hth_ravg_cols, hth_ravg_new_cols),
            include_groups=True,
        )
        .reset_index(drop=True)
    )
    team_hth_matches = (
        team_hth_matches.groupby(["Team", "Opponent"], group_keys=True)
        .apply(lambda x: rolling_match_results(x, "hth"), include_groups=True)
        .reset_index(drop=True)
    )

    league_drop_opp = [
        "Opp_Date",
        "Opp_Comp",
        "Opp_Round",
        "Opp_GF",
        "Opp_GA",
        "Opp_Opponent",
        "Opp_xG",
        "Opp_xGA",
        "Opp_Season",
        "Opp_Team_code",
        "Opp_Opp_code",
        "Opp_Venue_code",
        "Opp_Hour",
        "Opp_Day_code",
        "Opp_Target",
        "Opp_days_diff",
        "Opp_is_win",
    ]
    last_drop_opp = [
        "Opp_Date",
        "Opp_Comp",
        "Opp_Round",
        "Opp_GF",
        "Opp_GA",
        "Opp_Opponent",
        "Opp_xG",
        "Opp_xGA",
        "Opp_Season",
        "Opp_Team_code",
        "Opp_Opp_code",
        "Opp_Venue_code",
        "Opp_Hour",
        "Opp_Day_code",
        "Opp_days_diff",
    ]
    team_league_matches = _merge_opponent_asof(
        team_league_matches, team_league_matches, league_drop_opp
    )
    team_last_matches = _merge_opponent_asof(
        team_last_matches, team_last_matches, last_drop_opp
    )

    last_merge = copy.deepcopy(last_ravg_new_cols)
    hth_merge = copy.deepcopy(hth_ravg_new_cols)
    last_merge += [f"Opp_{column}" for column in last_ravg_new_cols]
    last_merge += ["Date", "Team"]
    hth_merge += ["Date", "Team", "hthnum_wins_rolling"]

    team_matches = team_league_matches.merge(
        team_last_matches[last_merge], on=["Date", "Team"], how="left"
    )
    team_matches = _merge_hth_asof(team_matches, team_hth_matches, hth_merge)
    team_matches = team_matches[team_matches["Comp"] == league].copy()

    team_matches = team_matches.sort_values("Date").reset_index(drop=True)

    positions = pre_match_positions(team_matches)
    team_matches["Team_position"] = positions["Team_position"]
    team_matches["Opp_position"] = positions["Opp_position"]

    team_attack = defaultdict(lambda: 0.0)
    team_defense = defaultdict(lambda: 0.0)
    home_adv = 0.2
    learning_rate = 0.05

    def expected_goals(team, opp, venue):
        h = home_adv if venue == "Home" else 0.0
        exp_team = np.exp(team_attack[team] - team_defense[opp] + h)
        exp_opp = np.exp(team_attack[opp] - team_defense[team])
        return exp_team, exp_opp

    def update_strengths(team, opp, team_goals, opp_goals, team_xg, opp_xg):
        team_attack[team] += learning_rate * (team_goals - team_xg)
        team_defense[team] += learning_rate * (opp_xg - opp_goals)
        team_attack[opp] += learning_rate * (opp_goals - opp_xg)
        team_defense[opp] += learning_rate * (team_xg - team_goals)

    # Rows on one date read pre-date state; each fixture updates once (both perspectives exist).
    n_rows = len(team_matches)
    expected_goal_team = np.zeros(n_rows)
    expected_goal_opp = np.zeros(n_rows)
    relative_attack = np.zeros(n_rows)
    relative_defense = np.zeros(n_rows)
    for _date, day in team_matches.groupby("Date", sort=True):
        pending = []
        seen_fixtures: set[frozenset] = set()
        for pos, row in zip(day.index, day.itertuples(index=False)):
            team, opp = row.Team, row.Opponent
            ex_team, ex_opp = expected_goals(team, opp, getattr(row, "Venue", ""))
            expected_goal_team[pos] = ex_team
            expected_goal_opp[pos] = ex_opp
            relative_attack[pos] = team_attack[team] - team_attack[opp]
            relative_defense[pos] = team_defense[team] - team_defense[opp]
            fixture = frozenset((team, opp))
            if pd.notna(row.GF) and pd.notna(row.GA) and fixture not in seen_fixtures:
                seen_fixtures.add(fixture)
                pending.append((team, opp, row.GF, row.GA, ex_team, ex_opp))
        for args in pending:
            update_strengths(*args)

    team_matches["Expected_goal_team"] = expected_goal_team
    team_matches["Expected_goal_opp"] = expected_goal_opp
    team_matches["Relative_attack_strength"] = relative_attack
    team_matches["Relative_defense_strength"] = relative_defense
    team_matches[
        ["Predicted_team_win", "Predicted_team_draw", "Predicted_team_loss"]
    ] = team_matches.apply(
        lambda row: match_outcome_probs(row["Expected_goal_team"], row["Expected_goal_opp"]),
        axis=1,
        result_type="expand",
    )

    team_ratings = defaultdict(lambda: 1500.0)
    k_factor = 20
    home_rating_adv = 40
    scaling = 400

    def expected_score(team_rating, opp_rating, venue):
        h = home_rating_adv if venue == "Home" else 0
        return 1 / (1 + 10 ** (-(team_rating - opp_rating + h) / scaling))

    def goal_diff_multiplier(team_goal, opp_goal):
        goal_diff = abs(team_goal - opp_goal)
        return np.log((goal_diff + 1) * (2.2 / (goal_diff + 2.2)))

    def update_ratings(team, opp, team_goal, opp_goal, venue):
        if team_goal > opp_goal:
            team_result = 1
        elif team_goal == opp_goal:
            team_result = 0.5
        else:
            team_result = 0
        opp_result = 1 - team_result
        team_rating, opp_rating = team_ratings[team], team_ratings[opp]
        team_expected = expected_score(team_rating, opp_rating, venue)
        opp_expected = 1 - team_expected
        multiplier = goal_diff_multiplier(team_goal, opp_goal)
        team_ratings[team] += k_factor * multiplier * (team_result - team_expected)
        team_ratings[opp] += k_factor * multiplier * (opp_result - opp_expected)

    ratings = np.zeros(n_rows)
    for _date, day in team_matches.groupby("Date", sort=True):
        pending = []
        seen_fixtures = set()
        for pos, row in zip(day.index, day.itertuples(index=False)):
            team, opp = row.Team, row.Opponent
            ratings[pos] = team_ratings[team] - team_ratings[opp]
            fixture = frozenset((team, opp))
            if pd.notna(row.GF) and pd.notna(row.GA) and fixture not in seen_fixtures:
                seen_fixtures.add(fixture)
                pending.append((team, opp, row.GF, row.GA, getattr(row, "Venue", "")))
        for args in pending:
            update_ratings(*args)
    team_matches["Relative_ratings"] = ratings
    if "Venue" in team_matches.columns:
        team_matches["is_home"] = (team_matches["Venue"] == "Home").astype(int)
    else:
        team_matches["is_home"] = 0

    get_relative_feature(team_matches, league_ravg_new_cols)
    get_relative_feature(team_matches, last_ravg_new_cols)
    team_matches["Relative_pos"] = (
        team_matches["Team_position"] - team_matches["Opp_position"]
    )
    team_matches["Relative_rolling_rests_1"] = (
        team_matches["rolling_rests_1"] - team_matches["Opp_rolling_rests_1"]
    )
    team_matches["Relative_rolling_rests_3"] = (
        team_matches["rolling_rests_3"] - team_matches["Opp_rolling_rests_3"]
    )
    team_matches["Relative_leaguenum_wins_rolling"] = (
        team_matches["leaguenum_wins_rolling"]
        - team_matches["Opp_leaguenum_wins_rolling"]
    )

    columns_to_drop = [
        "Sh",
        "SoT",
        "Dist",
        "FK",
        "PK",
        "PKatt",
        "CrdY",
        "CrdR",
        "2CrdY",
        "Fls",
        "Fld",
        "Off",
        "Crs",
        "Int",
        "TklW",
        "Won%",
        "SCA",
        "GCA",
        "Venue",
    ]
    opp_columns_to_drop = [f"Opp_{column}" for column in columns_to_drop]
    team_matches = team_matches.drop(columns=opp_columns_to_drop, errors="ignore")
    team_matches["Match_key"] = team_matches.apply(
        lambda row: tuple(
            sorted(
                [
                    normalize_team_name(row["Team"]) or str(row["Team"]),
                    normalize_team_name(row["Opponent"]) or str(row["Opponent"]),
                ]
            )
        ),
        axis=1,
    )
    columns_to_drop.extend(
        [
            "Comp",
            "Round",
            "GF",
            "GA",
            "xG",
            "xGA",
            "days_diff",
            "is_win",
        ]
    )
    team_matches = team_matches.drop(columns=columns_to_drop, errors="ignore")
    # Keep rows with completed results or upcoming fixtures.
    has_result = team_matches["Target"].notna()
    has_hth = team_matches["hthnum_wins_rolling"].notna()
    team_matches = team_matches[has_result | has_hth].copy()
    team_matches = team_matches.drop_duplicates(subset=["Match_key", "Date"])
    team_matches = team_matches.drop(columns=["Match_key"], errors="ignore")

    team_matches = drop_invalid_fixtures(team_matches)

    future_matches = team_matches[team_matches["Target"].isna()].copy()
    past_matches = team_matches[team_matches["Target"].notna()].copy()

    if not future_matches.empty:
        future_matches["_fixture_key"] = future_matches.apply(
            lambda row: tuple(
                sorted([str(row["Team"]), str(row["Opponent"])])
            )
            + (row["Date"].date().isoformat(),),
            axis=1,
        )
        if "Venue" in future_matches.columns:
            future_matches["_home_first"] = (
                future_matches["Venue"].eq("Home").map({True: 0, False: 1})
            )
        else:
            future_matches["_home_first"] = 0
        future_matches = future_matches.sort_values(["Date", "_home_first"])
        future_matches = future_matches.drop_duplicates(
            subset=["_fixture_key"], keep="first"
        ).drop(columns=["_fixture_key", "_home_first"], errors="ignore")

    return past_matches, future_matches, team_mapping_dict
