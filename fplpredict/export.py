import json
import logging
from datetime import datetime, timezone

import pandas as pd

from fplpredict.config import LEAGUE as DEFAULT_LEAGUE, MODEL_RESELECT_DAYS, active_leagues, league_slug
from fplpredict.data_io import read_csv
from fplpredict.model.features import build_feature_matrix
from fplpredict.model.ensemble import load_selection_meta, model_display_name
from fplpredict.paths import WEB_DATA_DIR, league_predictions_csv
from fplpredict.predictions_util import (
    backfill_actuals,
    dedupe_true_duplicates,
    drop_invalid_fixtures,
)
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
    if not team or not opponent or not pred or team == opponent:
        return None
    actual = _clean_str(row.get("Actual"))
    pick_label = _outcome_label(team, opponent, pred)
    actual_label = (
        _outcome_label(team, opponent, actual) if actual is not None else None
    )
    correct = pick_label == actual_label if actual_label is not None else None
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


def _model_meta_for_site(league: str) -> dict:
    selection = load_selection_meta(league) or {}
    strategy = selection.get("selected_model")
    reselect_days = selection.get("reselect_days", MODEL_RESELECT_DAYS)
    split = selection.get("split") or {}
    return {
        "selectedModel": strategy,
        "modelLabel": model_display_name(strategy),
        "modelEvaluatedAt": selection.get("evaluated_at"),
        "modelReselectDays": reselect_days,
        "modelTestAccuracy": selection.get("best_accuracy"),
        "modelSplit": split,
        "pipelineNote": (
            f"Weekly scrape and predict; model strategy re-evaluated about every "
            f"{reselect_days} days on a scheduled run."
        ),
    }


def _has_actual(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def _partition_predictions(df: pd.DataFrame, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Upcoming = no result and fixture date today or later; history = scored, before today."""
    scored = _has_actual(df["Actual"])
    upcoming = df[~scored & (df["Date"] >= today)].copy()
    history_df = df[scored & (df["Date"] < today)].copy()
    return upcoming, history_df


def _export_league_payload(league: str) -> dict:
    csv_path = league_predictions_csv(league)
    if not csv_path.exists():
        raise FileNotFoundError(f"No predictions CSV for {league}: {csv_path}")
    df = drop_invalid_fixtures(dedupe_true_duplicates(pd.read_csv(csv_path, index_col=0)))
    df["Date"] = pd.to_datetime(df["Date"])
    try:
        past_matches, _, _ = build_feature_matrix(
            read_csv("matches.csv"),
            read_csv("next_matches.csv"),
            read_csv("history.csv"),
            league=league,
        )
        df = backfill_actuals(df, past_matches)
        df = dedupe_true_duplicates(df)
    except Exception:
        log.warning("[%s] Could not backfill Actual from match results", league, exc_info=True)

    today = pd.Timestamp.now().normalize()
    upcoming, history_df = _partition_predictions(df, today)
    upcoming = _dedupe_fixtures(upcoming.sort_values("Date"))
    history_df = _dedupe_fixtures(history_df.sort_values("Date", ascending=False))

    upcoming_items = _build_items(upcoming)
    history_items = _build_items(history_df)
    history_items.sort(key=lambda x: x["date"], reverse=True)

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "league": league,
        "upcoming": upcoming_items,
        "history": history_items,
        "metrics": _metrics_from_history(history_items),
        "meta": {
            "upcomingCount": len(upcoming_items),
            "historyCount": len(history_items),
            "recentPreviewSize": 10,
            "historyPageSize": 20,
            **_model_meta_for_site(league),
            "positionsNote": (
                "Table positions are computed from scraped match results in matches.csv. "
                "Run the weekly scrape to refresh; missing # means the team is not in the "
                "simulated table yet (often a naming or incomplete data issue)."
            ),
        },
    }


def export_predictions_json() -> dict:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    leagues = active_leagues()
    manifest_leagues = []
    exported: dict[str, dict] = {}

    for league in leagues:
        slug = league_slug(league)
        manifest_leagues.append({"name": league, "slug": slug})
        try:
            payload = _export_league_payload(league)
        except FileNotFoundError:
            log.warning("[%s] Skipping JSON export (no predictions CSV yet).", league)
            continue
        encoded = json.dumps(payload, indent=2, allow_nan=False)
        (WEB_DATA_DIR / f"{slug}.json").write_text(encoded, encoding="utf-8")
        exported[league] = payload

    manifest = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "defaultLeague": DEFAULT_LEAGUE,
        "leagues": manifest_leagues,
    }
    (WEB_DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if exported:
        return exported.get(leagues[0]) or next(iter(exported.values()))
    return {}
