"""Gameweek live accuracy and model-strategy timeline for site export."""

from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from fplpredict.config import SEASON_ROLLOVER_MONTH
from fplpredict.data_io import read_csv
from fplpredict.model.ensemble import load_selection_meta, model_display_name
from fplpredict.paths import league_prediction_log
from fplpredict.teams import normalize_team_name

PL_LEGACY_STRATEGY = "soft_vote_fixed_4_4_1_1"
PL_LEGACY_CUTOFF = pd.Timestamp("2026-09-30")


def _gameweek_sort_key(label: str) -> tuple[int, int | str]:
    text = str(label).strip()
    match = re.search(r"(\d+)\s*$", text)
    if match:
        return (0, int(match.group(1)))
    return (1, text.lower())


def _is_scored(row: pd.Series) -> bool:
    actual = row.get("Actual")
    if actual is None or (isinstance(actual, float) and pd.isna(actual)):
        return False
    return str(actual).strip() != ""


def _pick_correct(row: pd.Series) -> bool:
    return str(row.get("Pred")).strip() == str(row.get("Actual")).strip()


def _prediction_as_of(row: pd.Series) -> pd.Timestamp:
    raw = row.get("Predicted_at")
    if raw is not None and not (isinstance(raw, float) and pd.isna(raw)) and str(raw).strip():
        ts = pd.Timestamp(raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.normalize()
    return pd.Timestamp(row["Date"]).normalize()


def _effective_strategy(row: pd.Series, league: str) -> str | None:
    if league == "Premier League" and _prediction_as_of(row) < PL_LEGACY_CUTOFF:
        return PL_LEGACY_STRATEGY
    strat = row.get("Model_strategy")
    if strat is None or (isinstance(strat, float) and pd.isna(strat)):
        return None
    text = str(strat).strip()
    return text or None


def season_end_year(d: pd.Timestamp) -> int:
    ts = pd.Timestamp(d).normalize()
    return ts.year + 1 if ts.month >= SEASON_ROLLOVER_MONTH else ts.year


def _attach_gameweeks(frame: pd.DataFrame, league: str) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out["_team"] = out["Team"].map(lambda x: normalize_team_name(x) or x)
    out["_opp"] = out["Opponent"].map(lambda x: normalize_team_name(x) or x)
    if "Season" in out.columns:
        out["_season"] = pd.to_numeric(out["Season"], errors="coerce")
    else:
        out["_season"] = pd.NA
    out["_season"] = out["_season"].fillna(out["Date"].map(season_end_year)).astype(int)

    try:
        matches = read_csv("matches.csv")
    except FileNotFoundError:
        out["Round"] = pd.NA
        return out

    sub = matches[matches["Comp"] == league].copy()
    if sub.empty or "Round" not in sub.columns:
        out["Round"] = pd.NA
        return out

    sub["Date"] = pd.to_datetime(sub["Date"]).dt.normalize()
    sub["_team"] = sub["Team"].map(lambda x: normalize_team_name(x) or x)
    sub["_opp"] = sub["Opponent"].map(lambda x: normalize_team_name(x) or x)
    sub["_season"] = pd.to_numeric(sub["Season"], errors="coerce")
    sub["_season"] = sub["_season"].fillna(sub["Date"].map(season_end_year)).astype(int)
    rounds = sub.drop_duplicates(subset=["Date", "_team", "_opp", "_season"])[
        ["Date", "_team", "_opp", "_season", "Round"]
    ]
    out = out.merge(rounds, on=["Date", "_team", "_opp", "_season"], how="left")
    if "Season" not in out.columns:
        out["Season"] = out["_season"]
    return out


def weekly_performance_by_gameweek(history_df: pd.DataFrame, league: str) -> list[dict]:
    """One row per competition gameweek (from matches.csv ``Round``), deduped history rows."""
    if history_df.empty:
        return []
    frame = _attach_gameweeks(history_df, league)
    frame = frame[frame.apply(_is_scored, axis=1)]
    if frame.empty:
        return []

    frame["correct"] = frame.apply(_pick_correct, axis=1)
    frame["_gw"] = frame["Round"].astype(str).str.strip()
    frame.loc[frame["_gw"].isin(("", "nan", "None")), "_gw"] = pd.NA
    missing = frame["_gw"].isna()
    if missing.any():
        fallback = frame.loc[missing, "Date"] - pd.to_timedelta(
            frame.loc[missing, "Date"].dt.dayofweek, unit="D"
        )
        frame.loc[missing, "_gw"] = "Week of " + fallback.dt.strftime("%Y-%m-%d")

    out: list[dict] = []
    group_cols = ["_season", "_gw"]
    for (_season, gw), grp in frame.groupby(group_cols, sort=False):
        grp = grp.sort_values("Date")
        n = len(grp)
        correct = int(grp["correct"].sum())
        strategies = [_effective_strategy(row, league) for _, row in grp.iterrows()]
        strategies = [s for s in strategies if s]
        dominant = Counter(strategies).most_common(1)[0][0] if strategies else None
        label = model_display_name(dominant) if dominant else None
        week_start = pd.Timestamp(grp["Date"].min())
        season_end = int(_season) if pd.notna(_season) else None
        out.append(
            {
                "gameweek": str(gw),
                "weekStart": week_start.strftime("%Y-%m-%d"),
                "seasonEnd": season_end,
                "n": n,
                "correct": correct,
                "accuracy": round(correct / n, 4) if n else None,
                "modelStrategy": dominant,
                "modelLabel": label,
            }
        )
    out.sort(
        key=lambda row: (
            row["seasonEnd"] or 0,
            _gameweek_sort_key(row["gameweek"]),
        )
    )
    return out


def _apply_premier_league_legacy_timeline(segments: list[dict]) -> list[dict]:
    cutoff_s = "2026-09-30T00:00:00Z"
    cutoff = pd.Timestamp(cutoff_s.replace("Z", ""))
    legacy = {
        "from": "2018-08-01T00:00:00Z",
        "to": cutoff_s,
        "strategy": PL_LEGACY_STRATEGY,
        "label": model_display_name(PL_LEGACY_STRATEGY),
        "note": (
            "All Premier League predictions before 30 Sep 2026 used the soft-voting "
            "ensemble with fixed weights (4,4,1,1)."
        ),
    }
    rest: list[dict] = []
    for seg in segments:
        seg_from = pd.Timestamp(seg["from"].replace("Z", ""))
        if seg_from >= cutoff:
            rest.append(seg)
            continue
        seg_to = seg.get("to")
        if seg_to:
            seg_to_ts = pd.Timestamp(str(seg_to).replace("Z", ""))
            if seg_to_ts <= cutoff:
                continue
        trimmed = {**seg, "from": cutoff_s}
        rest.append(trimmed)
    return [legacy] + rest


def model_strategy_timeline(league: str) -> list[dict]:
    """Step timeline from prediction log + current model_selection.json."""
    points: list[tuple[pd.Timestamp, str]] = []

    log_path = league_prediction_log(league)
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            strat = rec.get("modelStrategy")
            ev = rec.get("modelEvaluatedAt")
            if not strat or not ev:
                continue
            ts = pd.Timestamp(ev)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            points.append((ts, str(strat)))

    meta = load_selection_meta(league) or {}
    strat = meta.get("selected_model")
    ev = meta.get("evaluated_at")
    if strat and ev:
        ts = pd.Timestamp(ev)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        points.append((ts, str(strat)))

    if not points and league != "Premier League":
        return []

    segments: list[dict] = []
    if points:
        points.sort(key=lambda x: x[0])
        for ts, strategy in points:
            if segments and segments[-1]["strategy"] == strategy:
                continue
            if segments:
                segments[-1]["to"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            segments.append(
                {
                    "from": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": None,
                    "strategy": strategy,
                    "label": model_display_name(strategy),
                }
            )

    if league == "Premier League":
        return _apply_premier_league_legacy_timeline(segments)
    return segments


# Backwards-compatible alias for tests / imports
weekly_performance_by_fixture_date = weekly_performance_by_gameweek
