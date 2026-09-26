import copy
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fplpredict.config import LEAGUE
from fplpredict.teams import TEAM_ALIASES, normalize_team_name


class MissingDict(dict):
    def __missing__(self, key):
        return key


def build_team_name_mapping(matches: pd.DataFrame) -> MissingDict:
    league = matches[matches["Comp"] == LEAGUE]
    team_names = sorted(league["Team"].unique())
    opp_names = sorted(league["Opponent"].unique())
    exclusive_team = [name for name in team_names if name not in opp_names]
    exclusive_opp = [name for name in opp_names if name not in team_names]
    mapping = dict(TEAM_ALIASES)
    for team_name, opp_name in zip(exclusive_team, exclusive_opp):
        canonical_team = normalize_team_name(team_name) or team_name
        canonical_opp = normalize_team_name(opp_name) or opp_name
        mapping[team_name] = canonical_opp
        if canonical_team != team_name:
            mapping[canonical_team] = canonical_opp
    return MissingDict(**mapping)


def get_target_from_goals(row: pd.Series) -> int:
    if row["GF"] > row["GA"]:
        return 2
    if row["GF"] < row["GA"]:
        return 1
    return 0


def rolling_avgs(group, cols, new_cols, num_past_matches_considered=3):
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


def rolling_match_results(group, name, num_past_matches_considered=3):
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


def get_relative_feature(team_matches: pd.DataFrame, cols: list[str]) -> list[str]:
    relative_cols = []
    for col in cols:
        opp_col = f"Opp_{col}"
        if col in team_matches.columns and opp_col in team_matches.columns:
            rel = f"Relative_{col}"
            team_matches[rel] = team_matches[col] - team_matches[opp_col]
            relative_cols.append(rel)
    return relative_cols


def prepare_raw_frames(
    matches: pd.DataFrame, next_matches: pd.DataFrame, history_matches: pd.DataFrame
):
    mapping = build_team_name_mapping(matches)

    matches = matches.copy()
    next_matches = next_matches.copy()
    history_matches = history_matches.copy()

    matches["Team"] = matches["Team"].map(mapping).apply(
        lambda x: normalize_team_name(x) or x
    )
    matches["Opponent"] = matches["Opponent"].apply(
        lambda x: normalize_team_name(x) or x
    )
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
    matches["Target"] = matches["Result"].astype("category").cat.codes
    matches["GF"] = matches["GF"].apply(
        lambda x: int(re.search(r"^[0-9]+", str(x)).group(0))
    )
    matches["GA"] = matches["GA"].apply(
        lambda x: int(re.search(r"^[0-9]+", str(x)).group(0))
    )
    matches = matches.drop(
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

    next_matches["Team"] = next_matches["Team"].map(mapping).apply(
        lambda x: normalize_team_name(x) or x
    )
    next_matches["Opponent"] = next_matches["Opponent"].apply(
        lambda x: normalize_team_name(x) or x
    )
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
            "Venue",
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
        "xG",
        "xGA",
        "xG.1",
    ]
    for frame in (matches, next_matches, history_matches):
        for col in numeric_cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    return matches, next_matches, history_matches, team_mapping_dict


def build_feature_matrix(
    matches: pd.DataFrame,
    next_matches: pd.DataFrame,
    history_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    matches, next_matches, history_matches, team_mapping_dict = prepare_raw_frames(
        matches, next_matches, history_matches
    )

    league_ravg_cols = [
        "GF",
        "GA",
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
        "xG",
        "xGA",
    ]
    league_ravg_new_cols = [f"{c}league_rolling" for c in league_ravg_cols]
    last_ravg_cols = league_ravg_cols + ["Target"]
    last_ravg_new_cols = [f"{c}last_rolling" for c in last_ravg_cols]
    hth_ravg_cols = ["xG", "xG.1", "GF", "GA"]
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

    league_matches = all_matches[all_matches["Comp"] == LEAGUE]
    team_league_matches = (
        league_matches.groupby("Team", group_keys=True)
        .apply(
            lambda x: rolling_avgs(x, league_ravg_cols, league_ravg_new_cols),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )
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
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    team_hth_matches = (
        team_hth_matches.groupby("Team", group_keys=True)
        .apply(lambda x: rolling_match_results(x, "hth"), include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    opp_team_league = team_league_matches.add_prefix("Opp_")
    opp_team_last = team_last_matches.add_prefix("Opp_")

    team_league_matches = team_league_matches.merge(
        opp_team_league,
        left_on=["Date", "Opponent"],
        right_on=["Opp_Date", "Opp_Team"],
        how="left",
    )
    team_league_matches = team_league_matches.drop(
        columns=[
            "Opp_Date",
            "Opp_Team",
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
        ],
        errors="ignore",
    )

    team_last_matches = team_last_matches.merge(
        opp_team_last,
        left_on=["Date", "Opponent"],
        right_on=["Opp_Date", "Opp_Team"],
        how="left",
    )
    team_last_matches = team_last_matches.drop(
        columns=[
            "Opp_Date",
            "Opp_Team",
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
        ],
        errors="ignore",
    )

    last_merge = copy.deepcopy(last_ravg_new_cols)
    hth_merge = copy.deepcopy(hth_ravg_new_cols)
    last_merge += [f"Opp_{column}" for column in last_ravg_new_cols]
    last_merge += ["Date", "Team"]
    hth_merge += ["Date", "Team", "hthnum_wins_rolling"]

    team_matches = team_league_matches.merge(
        team_last_matches[last_merge], on=["Date", "Team"], how="left"
    )
    team_matches = team_matches.merge(
        team_hth_matches[hth_merge], on=["Date", "Team"], how="left"
    )
    team_matches = team_matches[team_matches["Comp"] == LEAGUE].copy()

    sorted_round = sorted(
        team_matches["Round"].unique(),
        key=lambda rnd: int(re.search(r"[0-9]+$", rnd).group(0)),
    )
    round_mapping = {value: key for key, value in enumerate(sorted_round, start=1)}
    team_matches["Round"] = team_matches["Round"].map(round_mapping)
    team_matches = team_matches.sort_values("Date").reset_index(drop=True)

    standings = defaultdict(lambda: {"points": -1, "goal_diff": 0, "GF": 0})
    curr_season = 0
    for season_gameweek, stats in team_matches.groupby(["Season", "Round"], sort=True):
        if curr_season != season_gameweek[0]:
            curr_season = season_gameweek[0]
            standings = defaultdict(lambda: {"points": -1, "goal_diff": 0, "GF": 0})

        sorted_standings = sorted(
            standings.items(),
            key=lambda x: (-x[1]["points"], -x[1]["goal_diff"], -x[1]["GF"], x[0]),
        )
        ranking = {
            team: pos for pos, (team, _stats) in enumerate(sorted_standings, start=1)
        }

        for index, row in stats.iterrows():
            team_pos = ranking.get(row["Team"])
            opp_pos = ranking.get(row["Opponent"])
            team_matches.at[index, "Team_position"] = (
                team_pos if team_pos is not None else np.nan
            )
            team_matches.at[index, "Opp_position"] = (
                opp_pos if opp_pos is not None else np.nan
            )

        for index, row in stats.iterrows():
            goals_for = 0 if pd.isna(row["GF"]) else row["GF"]
            goals_against = 0 if pd.isna(row["GA"]) else row["GA"]
            standings[row["Team"]]["GF"] += goals_for
            standings[row["Team"]]["goal_diff"] += goals_for - goals_against
            if goals_for > goals_against:
                standings[row["Team"]]["points"] += 3
            elif goals_for == goals_against:
                standings[row["Team"]]["points"] += 1

    team_attack = defaultdict(lambda: 0.0)
    team_defense = defaultdict(lambda: 0.0)
    home_adv = 0.2
    learning_rate = 0.05

    def expected_goals(team, opp):
        exp_team = np.exp(team_attack[team] - team_defense[opp] + home_adv)
        exp_opp = np.exp(team_attack[opp] - team_defense[team])
        return exp_team, exp_opp

    def update_strengths(team, opp, team_goals, opp_goals, team_xg, opp_xg):
        team_attack[team] += learning_rate * (team_goals - team_xg)
        team_defense[team] += learning_rate * (opp_xg - opp_goals)
        team_attack[opp] += learning_rate * (opp_goals - opp_xg)
        team_defense[opp] += learning_rate * (team_xg - team_goals)

    expected_goal_team, expected_goal_opp = [], []
    relative_attack, relative_defense = [], []
    for _, row in team_matches.iterrows():
        ex_team, ex_opp = expected_goals(row["Team"], row["Opponent"])
        expected_goal_team.append(ex_team)
        expected_goal_opp.append(ex_opp)
        relative_attack.append(team_attack[row["Team"]] - team_attack[row["Opponent"]])
        relative_defense.append(
            team_defense[row["Team"]] - team_defense[row["Opponent"]]
        )
        if not pd.isna(row["GF"]) and not pd.isna(row["GA"]):
            update_strengths(
                row["Team"], row["Opponent"], row["GF"], row["GA"], ex_team, ex_opp
            )

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

    ratings = []
    for _, row in team_matches.iterrows():
        ratings.append(team_ratings[row["Team"]] - team_ratings[row["Opponent"]])
        if not pd.isna(row["GF"]) and not pd.isna(row["GA"]):
            update_ratings(row["Team"], row["Opponent"], row["GF"], row["GA"], row["Venue"])
    team_matches["Relative_ratings"] = ratings

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
        lambda row: tuple(sorted([row["Team"], row["Opponent"]])),
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

    future_matches = team_matches[team_matches["Target"].isna()].copy()
    past_matches = team_matches[team_matches["Target"].notna()].copy()
    return past_matches, future_matches, team_mapping_dict
