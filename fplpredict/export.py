import json
import logging
from datetime import datetime, timezone

import pandas as pd

from fplpredict.data_io import read_csv
from fplpredict.model.features import build_feature_matrix
from fplpredict.config import MODEL_RESELECT_DAYS
from fplpredict.model.ensemble import load_selection_meta, model_display_name
from fplpredict.paths import PREDICTIONS_CSV, PREDICTIONS_JSON, WEB_DATA_DIR
from fplpredict.predictions_util import backfill_actuals, dedupe_true_duplicates
from fplpredict.teams import normalize_team_name

log = logging.getLogger(__name__)


def _clean_str(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if pd.isna(val):
        return None
    text = str(val).strip()
    return text if text else None


def _clean_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return None
    return float(val)


def _clean_position(val) -> int | None:
    if val is None or pd.isna(val):
        return None
    pos = int(float(val))
    return pos if pos > 0 else None


def _outcome_label(team: str, opponent: str, outcome: str) -> str:
    if outcome == "Draw":
        return "Draw"
    if outcome == "Win":
        return f"{team} to win"
    if outcome == "Loss":
        return f"{opponent} to win"
    return outcome


def _fixture_key(row) -> tuple:
    team = normalize_team_name(_clean_str(row.get("Team")))
    opp = normalize_team_name(_clean_str(row.get("Opponent")))
    if not team or not opp:
        return ()
    return tuple(sorted([team, opp])) + (str(row["Date"].date()),)


def _row_rank(row) -> tuple:
    """Higher is better when choosing one row per fixture."""
    pos_score = 0
    if _clean_position(row.get("Team_position")) is not None:
        pos_score += 2
    if _clean_position(row.get("Opp_position")) is not None:
        pos_score += 2
    try:
        conf = max(
            float(row["Team_win%"]),
            float(row["Draw%"]),
            float(row["Opp_win%"]),
        )
    except (TypeError, ValueError, KeyError):
        conf = 0.0
    return (pos_score, conf)


def _dedupe_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture (normalized team names); keep the best row."""
    if df.empty:
        return df
    best: dict[tuple, pd.Series] = {}
    for _, row in df.iterrows():
        key = _fixture_key(row)
        if not key:
            continue
        if key not in best or _row_rank(row) > _row_rank(best[key]):
            best[key] = row
    if not best:
        return pd.DataFrame()
    return pd.DataFrame(list(best.values()))


def _row_to_item(row) -> dict | None:
    team = normalize_team_name(_clean_str(row.get("Team")))
    opponent = normalize_team_name(_clean_str(row.get("Opponent")))
    pred = _clean_str(row.get("Pred"))
    if not team or not opponent or not pred:
        return None
    actual = _clean_str(row.get("Actual"))
    pick_label = _outcome_label(team, opponent, pred)
    actual_label = (
        _outcome_label(team, opponent, actual) if actual is not None else None
    )
    correct = (
        pick_label == actual_label if actual_label is not None else None
    )
    return {
        "date": row["Date"].strftime("%Y-%m-%d"),
        "team": team,
        "opponent": opponent,
        "pred": pred,
        "pickLabel": pick_label,
        "actual": actual,
        "actualLabel": actual_label,
        "teamPosition": _clean_position(row.get("Team_position")),
        "oppPosition": _clean_position(row.get("Opp_position")),
        "probs": {
            "teamWin": _clean_float(row["Team_win%"]),
            "draw": _clean_float(row["Draw%"]),
            "oppWin": _clean_float(row["Opp_win%"]),
        },
        "correct": correct,
    }


def _build_items(frame: pd.DataFrame) -> list[dict]:
    items = []
    for _, row in frame.iterrows():
        item = _row_to_item(row)
        if item is None:
            continue
        if all(v is not None for v in item["probs"].values()):
            items.append(item)
    return items


def _metrics_from_history(items: list[dict]) -> dict | None:
    scored = [i for i in items if i.get("actualLabel")]
    if not scored:
        return None
    correct = sum(1 for i in scored if i.get("correct") is True)
    return {
        "n": len(scored),
        "accuracy": round(correct / len(scored), 4),
    }


def _model_meta_for_site() -> dict:
    selection = load_selection_meta() or {}
    strategy = selection.get("selected_model")
    reselect_days = selection.get("reselect_days", MODEL_RESELECT_DAYS)
    return {
        "selectedModel": strategy,
        "modelLabel": model_display_name(strategy),
        "modelEvaluatedAt": selection.get("evaluated_at"),
        "modelReselectDays": reselect_days,
        "modelTestAccuracy": selection.get("best_accuracy"),
        "pipelineNote": (
            f"Weekly scrape and predict; model strategy re-evaluated about every "
            f"{reselect_days} days on a scheduled run."
        ),
    }


def export_predictions_json() -> dict:
    df = dedupe_true_duplicates(pd.read_csv(PREDICTIONS_CSV, index_col=0))
    df["Date"] = pd.to_datetime(df["Date"])
    try:
        past_matches, _, _ = build_feature_matrix(
            read_csv("matches.csv"),
            read_csv("next_matches.csv"),
            read_csv("history.csv"),
        )
        df = backfill_actuals(df, past_matches)
        df = dedupe_true_duplicates(df)
    except Exception:
        log.warning("Could not backfill Actual from match results", exc_info=True)
    today = pd.Timestamp.now().normalize()

    upcoming = df[(df["Actual"].isna()) | (df["Actual"] == "")].copy()
    upcoming = upcoming[upcoming["Date"] >= today]
    upcoming = upcoming.sort_values("Date", ascending=True)
    upcoming = _dedupe_fixtures(upcoming)

    history_df = df[df["Actual"].notna() & (df["Actual"] != "")].copy()
    history_df = history_df[history_df["Date"] < today]
    history_df = history_df.sort_values("Date", ascending=False)
    history_df = _dedupe_fixtures(history_df)

    upcoming_items = _build_items(upcoming)
    history_items = _build_items(history_df)
    history_items.sort(key=lambda x: x["date"], reverse=True)

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "league": "Premier League",
        "upcoming": upcoming_items,
        "history": history_items,
        "metrics": _metrics_from_history(history_items),
        "meta": {
            "upcomingCount": len(upcoming_items),
            "historyCount": len(history_items),
            "recentPreviewSize": 10,
            "historyPageSize": 20,
            **_model_meta_for_site(),
            "positionsNote": (
                "Table positions are computed from scraped match results in matches.csv. "
                "Run the weekly scrape to refresh; missing # means the team is not in the "
                "simulated table yet (often a naming or incomplete data issue)."
            ),
        },
    }

    encoded = json.dumps(payload, indent=2, allow_nan=False)
    PREDICTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_JSON.write_text(encoded, encoding="utf-8")
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DATA_DIR / "predictions.json").write_text(encoded, encoding="utf-8")
    return payload
