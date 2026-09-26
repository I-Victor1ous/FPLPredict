"""Merge and deduplicate prediction history without dropping distinct fixtures."""

from __future__ import annotations

import pandas as pd

from fplpredict.config import RESULT_LABELS
from fplpredict.teams import normalize_team_name


def _has_opponent(val) -> bool:
    return normalize_team_name(val) is not None


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
    out = out.sort_values("Date", ascending=False)
    out = out.drop_duplicates(subset=["Date", "Team", "Opponent"], keep="first")
    return out.sort_values("Date", ascending=False).reset_index(drop=True)


def backfill_actuals(df: pd.DataFrame, past_matches: pd.DataFrame) -> pd.DataFrame:
    """Set Actual from played results using (Date, Team, Opponent)."""
    if df.empty or past_matches.empty:
        return df

    out = normalize_predictions_frame(df)
    pm = past_matches.copy()
    pm["Date"] = pd.to_datetime(pm["Date"])
    pm["Team"] = pm["Team"].apply(normalize_team_name)
    pm["Opponent"] = pm["Opponent"].apply(normalize_team_name)

    lookup = pm[["Date", "Team", "Opponent", "Target"]].drop_duplicates(
        subset=["Date", "Team", "Opponent"], keep="last"
    )
    lookup = lookup.rename(
        columns={"Target": "_target"},
    )
    lookup["_actual_label"] = lookup["_target"].map(RESULT_LABELS)

    out = out.merge(
        lookup[["Date", "Team", "Opponent", "_actual_label"]],
        on=["Date", "Team", "Opponent"],
        how="left",
    )
    if "Actual" in out.columns:
        out["Actual"] = out["Actual"].combine_first(out["_actual_label"])
    else:
        out["Actual"] = out["_actual_label"]
    return out.drop(columns=["_actual_label"], errors="ignore")
