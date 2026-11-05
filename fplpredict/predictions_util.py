"""Merge and deduplicate prediction history without dropping distinct fixtures."""

from __future__ import annotations

import pandas as pd

from fplpredict.config import RESULT_LABELS
from fplpredict.teams import normalize_team_name

_OPPOSITE_OUTCOME = {"Win": "Loss", "Loss": "Win", "Draw": "Draw"}
_RESULT_TO_TARGET = {"D": 0, "L": 1, "W": 2}
_TARGET_FLIP = {0: 0, 1: 2, 2: 1}


def fixture_key(row) -> tuple | None:
    team = normalize_team_name(row.get("Team"))
    opp = normalize_team_name(row.get("Opponent"))
    if not team or not opp:
        return None
    date = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
    return (date, team, opp)


def symmetric_fixture_key(row) -> tuple | None:
    """One key per fixture regardless of which club is listed as Team."""
    team = normalize_team_name(row.get("Team"))
    opp = normalize_team_name(row.get("Opponent"))
    if not team or not opp:
        return None
    return (pd.Timestamp(row["Date"]).date(), tuple(sorted([team, opp])))


def _actual_is_set(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return False
    return str(val).strip() != ""


def propagate_fixture_actuals(df: pd.DataFrame) -> pd.DataFrame:
    """Copy Actual onto the reverse team/opponent row for the same fixture."""
    if df.empty or "Actual" not in df.columns:
        return df

    out = normalize_predictions_frame(df)
    scored: dict[tuple, tuple[str, str, str]] = {}
    for _, row in out.iterrows():
        actual = row.get("Actual")
        if not _actual_is_set(actual):
            continue
        key = symmetric_fixture_key(row)
        if key is None:
            continue
        scored[key] = (str(actual).strip(), str(row["Team"]), str(row["Opponent"]))

    if not scored:
        return out

    for idx, row in out.iterrows():
        if _actual_is_set(row.get("Actual")):
            continue
        key = symmetric_fixture_key(row)
        if key is None or key not in scored:
            continue
        actual, ref_team, ref_opp = scored[key]
        team = str(row["Team"])
        if team == ref_team:
            out.at[idx, "Actual"] = actual
        elif team == ref_opp:
            out.at[idx, "Actual"] = _OPPOSITE_OUTCOME.get(actual, actual)
    return out


def _has_opponent(val) -> bool:
    return normalize_team_name(val) is not None


def drop_invalid_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing or identical team / opponent (bad scrape or mapping)."""
    if df.empty:
        return df
    out = normalize_predictions_frame(df)
    mask = out["Team"].notna() & out["Opponent"].notna()
    mask &= out["Team"].astype(str).str.strip() != ""
    mask &= out["Opponent"].astype(str).str.strip() != ""
    mask &= out["Team"] != out["Opponent"]
    return out[mask].copy()


def normalize_predictions_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    for col in ("Team", "Opponent"):
        if col in out.columns:
            out[col] = out[col].apply(normalize_team_name)
    return out


def dedupe_true_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove only rows that are true duplicates:
    - Blank opponent when the same (Date, Team) already has a named opponent.
    - Exact same (Date, Team, Opponent) repeated (keep latest row).

    Does NOT collapse different opponents on the same date for the same team
    (would only happen if source data is wrong; those rows are kept).
    """
    if df.empty:
        return df

    out = normalize_predictions_frame(df)
    drop_indices: set = set()

    for (_date, _team), group in out.groupby(["Date", "Team"], sort=False):
        with_opp = group[group["Opponent"].apply(_has_opponent)]
        without_opp = group[~group["Opponent"].apply(_has_opponent)]
        if not with_opp.empty and not without_opp.empty:
            drop_indices.update(without_opp.index.tolist())
        elif with_opp.empty and len(without_opp) > 1:
            # Multiple junk rows with no opponent — keep the last one only.
            drop_indices.update(without_opp.index.tolist()[:-1])

    out = out.drop(index=list(drop_indices), errors="ignore")
    if "Predicted_at" in out.columns:
        out = out.sort_values(
            ["Date", "Predicted_at"],
            ascending=[False, True],
            na_position="last",
        )
    else:
        out = out.sort_values("Date", ascending=False)
    out = out.drop_duplicates(subset=["Date", "Team", "Opponent"], keep="first")
    return out.sort_values("Date", ascending=False).reset_index(drop=True)


def completed_match_results(matches: pd.DataFrame, league: str) -> pd.DataFrame:
    """Played fixtures from ``matches.csv`` as (Date, Team, Opponent, Target)."""
    from fplpredict.data.matches_clean import clean_matches_for_modeling
    from fplpredict.model.features import RESULT_TARGET, build_team_name_mapping

    if matches.empty or "Comp" not in matches.columns:
        return pd.DataFrame(columns=["Date", "Team", "Opponent", "Target"])

    mapping = build_team_name_mapping(matches, league=league)
    sub = clean_matches_for_modeling(matches.loc[matches["Comp"] == league].copy())
    sub["Team"] = sub["Team"].map(mapping).apply(lambda x: normalize_team_name(x) or x)
    sub["Opponent"] = sub["Opponent"].map(mapping).apply(
        lambda x: normalize_team_name(x) or x
    )
    sub = sub[sub["Result"].isin(RESULT_TARGET)]
    sub["Date"] = pd.to_datetime(sub["Date"])
    sub["Target"] = sub["Result"].map(RESULT_TARGET)
    return (
        sub[["Date", "Team", "Opponent", "Target"]]
        .drop_duplicates(subset=["Date", "Team", "Opponent"], keep="last")
        .reset_index(drop=True)
    )


def _result_lookup(past_matches: pd.DataFrame) -> pd.DataFrame:
    pm = past_matches.copy()
    pm["Date"] = pd.to_datetime(pm["Date"])
    pm["Team"] = pm["Team"].apply(normalize_team_name)
    pm["Opponent"] = pm["Opponent"].apply(normalize_team_name)
    if "Target" not in pm.columns:
        if "Result" not in pm.columns:
            return pd.DataFrame(columns=["Date", "Team", "Opponent", "_actual_label"])
        pm["Target"] = pm["Result"].map(_RESULT_TO_TARGET)
    pm = pm.dropna(subset=["Target"])
    direct = pm[["Date", "Team", "Opponent", "Target"]].drop_duplicates(
        subset=["Date", "Team", "Opponent"], keep="last"
    )
    reverse_rows = []
    for _, row in direct.iterrows():
        target = int(row["Target"])
        flipped = _TARGET_FLIP[target]
        reverse_rows.append(
            {
                "Date": row["Date"],
                "Team": row["Opponent"],
                "Opponent": row["Team"],
                "Target": flipped,
            }
        )
    if reverse_rows:
        direct = pd.concat([direct, pd.DataFrame(reverse_rows)], ignore_index=True)
    direct = direct.drop_duplicates(subset=["Date", "Team", "Opponent"], keep="last")
    direct["_actual_label"] = direct["Target"].map(RESULT_LABELS)
    return direct[["Date", "Team", "Opponent", "_actual_label"]]


def backfill_actuals(df: pd.DataFrame, past_matches: pd.DataFrame) -> pd.DataFrame:
    """Set Actual from played results using (Date, Team, Opponent)."""
    if df.empty or past_matches.empty:
        return df

    out = normalize_predictions_frame(df)
    lookup = _result_lookup(past_matches)
    if lookup.empty:
        return out

    out = out.merge(lookup, on=["Date", "Team", "Opponent"], how="left")
    if "Actual" in out.columns:
        out["Actual"] = out["Actual"].combine_first(out["_actual_label"])
    else:
        out["Actual"] = out["_actual_label"]
    out = out.drop(columns=["_actual_label"], errors="ignore")
    return propagate_fixture_actuals(out)
