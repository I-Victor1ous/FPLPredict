"""Append-only prediction log (integrity audit trail)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fplpredict.paths import league_prediction_log
from fplpredict.predictions_util import fixture_key, normalize_predictions_frame

log = logging.getLogger(__name__)


def prediction_log_path(league: str) -> Path:
    return league_prediction_log(league)


def _read_log(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    records: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            key = tuple(rec["key"])
            records[key] = rec
        except (json.JSONDecodeError, KeyError):
            continue
    return records


def append_prediction_records(
    league: str,
    rows: pd.DataFrame,
    *,
    model_strategy: str | None,
    model_evaluated_at: str | None,
) -> None:
    """Append first-seen fixtures to JSONL; never rewrite existing keys."""
    path = prediction_log_path(league)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_log(path)
    stamped = datetime.now(timezone.utc).isoformat()
    new_lines: list[str] = []

    frame = normalize_predictions_frame(rows)
    for _, row in frame.iterrows():
        key = fixture_key(row)
        if not key or key in existing:
            continue
        rec = {
            "key": list(key),
            "predictedAt": stamped,
            "modelStrategy": model_strategy,
            "modelEvaluatedAt": model_evaluated_at,
            "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
            "team": row["Team"],
            "opponent": row["Opponent"],
            "pred": row.get("Pred"),
            "teamWinPct": row.get("Team_win%"),
            "drawPct": row.get("Draw%"),
            "oppWinPct": row.get("Opp_win%"),
        }
        new_lines.append(json.dumps(rec, allow_nan=False))
        existing[key] = rec

    if new_lines:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(new_lines) + "\n")
        log.info("[%s] Appended %d prediction log entries.", league, len(new_lines))


def _row_has_actual(row) -> bool:
    actual = row.get("Actual") if hasattr(row, "get") else None
    if actual is None or (isinstance(actual, float) and pd.isna(actual)) or pd.isna(actual):
        return False
    return str(actual).strip() != ""


def apply_log_to_frame(
    df: pd.DataFrame,
    league: str,
    *,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fill missing prediction metadata from the append-only log."""
    if df.empty:
        return df
    records = _read_log(prediction_log_path(league))
    if not records:
        return df

    today = today or pd.Timestamp.now().normalize()
    out = normalize_predictions_frame(df)
    for col in (
        "Predicted_at",
        "Model_strategy",
        "Model_evaluated_at",
        "Pred",
        "Team_win%",
        "Draw%",
        "Opp_win%",
    ):
        if col not in out.columns:
            out[col] = None

    for idx, row in out.iterrows():
        key = fixture_key(row)
        rec = records.get(key)
        if not rec:
            continue
        started = pd.Timestamp(row["Date"]).normalize() < today
        locked = started and not _row_has_actual(row)
        if locked:
            if rec.get("pred") is not None:
                out.at[idx, "Pred"] = rec.get("pred")
            if rec.get("teamWinPct") is not None:
                out.at[idx, "Team_win%"] = rec.get("teamWinPct")
            if rec.get("drawPct") is not None:
                out.at[idx, "Draw%"] = rec.get("drawPct")
            if rec.get("oppWinPct") is not None:
                out.at[idx, "Opp_win%"] = rec.get("oppWinPct")
            out.at[idx, "Predicted_at"] = rec.get("predictedAt")
            out.at[idx, "Model_strategy"] = rec.get("modelStrategy")
            out.at[idx, "Model_evaluated_at"] = rec.get("modelEvaluatedAt")
            continue
        if pd.isna(out.at[idx, "Predicted_at"]) or out.at[idx, "Predicted_at"] in ("", None):
            out.at[idx, "Predicted_at"] = rec.get("predictedAt")
        if pd.isna(out.at[idx, "Model_strategy"]) or out.at[idx, "Model_strategy"] in ("", None):
            out.at[idx, "Model_strategy"] = rec.get("modelStrategy")
        if pd.isna(out.at[idx, "Model_evaluated_at"]) or out.at[idx, "Model_evaluated_at"] in (
            "",
            None,
        ):
            out.at[idx, "Model_evaluated_at"] = rec.get("modelEvaluatedAt")
    return out


def restore_missing_predictions_from_log(
    df: pd.DataFrame, league: str
) -> pd.DataFrame:
    """Re-insert prediction rows that exist in the log but were dropped from the CSV."""
    records = _read_log(prediction_log_path(league))
    if not records:
        return df

    out = normalize_predictions_frame(df) if not df.empty else pd.DataFrame()
    existing_keys: set[tuple] = set()
    existing_fixtures: set[tuple] = set()
    if not out.empty:
        for _, row in out.iterrows():
            key = fixture_key(row)
            if key:
                existing_keys.add(key)
            date = pd.Timestamp(row["Date"]).date()
            team = row.get("Team")
            opp = row.get("Opponent")
            if team and opp:
                existing_fixtures.add((date, tuple(sorted([str(team), str(opp)]))))

    restored: list[dict] = []
    for key, rec in records.items():
        if key in existing_keys:
            continue
        date = pd.Timestamp(rec["date"]).date()
        teams = tuple(sorted([str(key[1]), str(key[2])]))
        if (date, teams) in existing_fixtures:
            continue
        restored.append(
            {
                "Date": pd.Timestamp(rec["date"]),
                "Team": rec["team"],
                "Opponent": rec["opponent"],
                "Pred": rec.get("pred"),
                "Team_win%": rec.get("teamWinPct"),
                "Draw%": rec.get("drawPct"),
                "Opp_win%": rec.get("oppWinPct"),
                "Predicted_at": rec.get("predictedAt"),
                "Model_strategy": rec.get("modelStrategy"),
                "Model_evaluated_at": rec.get("modelEvaluatedAt"),
            }
        )
        existing_fixtures.add((date, teams))

    if not restored:
        return out
    merged = pd.concat([out, pd.DataFrame(restored)], ignore_index=True)
    log.info("[%s] Restored %d prediction row(s) from log.", league, len(restored))
    return merged.sort_values("Date", ascending=False).reset_index(drop=True)


def merge_with_integrity(
    new_predictions: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Keep stored picks/probs for started or scored fixtures; allow updates only
    for fixtures still in the future with no result.
    """
    today = today or pd.Timestamp.now().normalize()
    new_predictions = normalize_predictions_frame(new_predictions)
    if existing.empty:
        out = new_predictions.copy()
        out["Predicted_at"] = datetime.now(timezone.utc).isoformat()
        return out

    existing = normalize_predictions_frame(existing)
    merge_keys = ["Date", "Team", "Opponent"]
    existing_idx = existing.set_index(merge_keys, drop=False)

    rows = []
    for _, row in new_predictions.iterrows():
        key = (row["Date"], row["Team"], row["Opponent"])
        if key not in existing_idx.index:
            fresh = row.copy()
            fresh["Predicted_at"] = datetime.now(timezone.utc).isoformat()
            rows.append(fresh)
            continue

        old = existing_idx.loc[key]
        if isinstance(old, pd.DataFrame):
            old = old.iloc[0].copy()
        else:
            old = old.copy()
        has_actual = pd.notna(old.get("Actual")) and str(old.get("Actual")).strip() != ""
        started = pd.Timestamp(row["Date"]).normalize() < today
        if has_actual or started:
            rows.append(old)
        else:
            updated = row.copy()
            for meta in ("Predicted_at", "Model_strategy", "Model_evaluated_at"):
                if meta in old.index and pd.notna(old.get(meta)):
                    updated[meta] = old[meta]
            rows.append(updated)

    untouched = existing[
        ~existing.set_index(merge_keys).index.isin(
            new_predictions.set_index(merge_keys).index
        )
    ]
    merged = pd.concat([pd.DataFrame(rows), untouched], ignore_index=True)
    return merged.sort_values("Date", ascending=False).reset_index(drop=True)
