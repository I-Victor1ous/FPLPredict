import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from fplpredict.config import LEAGUE as DEFAULT_LEAGUE
from fplpredict.config import MODEL_RESELECT_DAYS, active_leagues
from fplpredict.data_io import read_csv
from fplpredict.model.ensemble import load_selection_meta, model_display_name
from fplpredict.model.features import build_feature_matrix
from fplpredict.model.market import combined_market_lookup, market_probs_dict
from fplpredict.model.metrics import accuracy_with_interval
from fplpredict.model.rolling_backtest import load_rolling_backtest
from fplpredict.paths import WEB_DATA_DIR, league_predictions_csv, league_slug, league_web_json
from fplpredict.performance_export import (
    model_strategy_timeline,
    weekly_performance_by_gameweek,
)
from fplpredict.predictions_util import (
    backfill_actuals,
    completed_match_results,
    dedupe_true_duplicates,
    drop_invalid_fixtures,
    propagate_fixture_actuals,
)
from fplpredict.teams import normalize_team_name

log = logging.getLogger(__name__)

# Played fixtures without a scraped result stay visible this long (then assumed postponed).
AWAITING_RESULT_DAYS = 14
CONFIDENT_PICK_MIN = float(os.environ.get("FPL_CONFIDENT_PICK_MIN", "55"))


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


def _item_fixture_key(item: dict) -> tuple:
    team = normalize_team_name(_clean_str(item.get("team")))
    opp = normalize_team_name(_clean_str(item.get("opponent")))
    if not team or not opp:
        return ()
    return tuple(sorted([team, opp])) + (str(item["date"])[:10],)


def _drop_items_already_scored(
    items: list[dict], scored_keys: set[tuple]
) -> list[dict]:
    if not scored_keys:
        return items
    return [i for i in items if _item_fixture_key(i) not in scored_keys]


def _predicted_at(row) -> pd.Timestamp:
    raw = row.get("Predicted_at")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return pd.Timestamp.max
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts
    except (TypeError, ValueError):
        return pd.Timestamp.max


def _dedupe_fixtures(
    df: pd.DataFrame, *, keep: str = "earliest"
) -> pd.DataFrame:
    """One row per fixture; earliest for history integrity, latest for upcoming."""
    if df.empty:
        return df
    best: dict[tuple, pd.Series] = {}
    for _, row in df.iterrows():
        key = _fixture_key(row)
        if not key:
            continue
        if key not in best:
            best[key] = row
            continue
        at_row, at_best = _predicted_at(row), _predicted_at(best[key])
        if keep == "latest":
            if at_row > at_best:
                best[key] = row
        elif at_row < at_best:
            best[key] = row
    if not best:
        return pd.DataFrame()
    return pd.DataFrame(list(best.values()))


def _fixture_home_map(
    past_matches: pd.DataFrame, future_matches: pd.DataFrame
) -> dict[tuple, int]:
    home: dict[tuple, int] = {}
    for frame in (past_matches, future_matches):
        if frame is None or frame.empty or "is_home" not in frame.columns:
            continue
        for _, r in frame.iterrows():
            team = normalize_team_name(_clean_str(r.get("Team")))
            opp = normalize_team_name(_clean_str(r.get("Opponent")))
            if not team or not opp:
                continue
            key = (pd.Timestamp(r["Date"]).date(), team, opp)
            home[key] = int(r.get("is_home", 0))
    return home


def _row_to_item(
    row,
    *,
    market_lookup: dict | None = None,
    home_map: dict[tuple, int] | None = None,
) -> dict | None:
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
    if market_lookup is not None and home_map is not None:
        team_k = normalize_team_name(team)
        opp_k = normalize_team_name(opponent)
        key = (row["Date"].date(), team_k, opp_k)
        series = pd.Series(
            {
                "Date": row["Date"],
                "Team": team_k,
                "Opponent": opp_k,
                "is_home": home_map.get(key, 1),
            }
        )
        market = market_probs_dict(series, market_lookup)
        if market:
            item["marketProbs"] = market
    return item


def _build_items(
    frame: pd.DataFrame,
    *,
    market_lookup: dict | None = None,
    home_map: dict[tuple, int] | None = None,
) -> list[dict]:
    items = []
    for _, row in frame.iterrows():
        item = _row_to_item(row, market_lookup=market_lookup, home_map=home_map)
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
    stats = accuracy_with_interval(correct, len(scored))
    return stats


def _confidence_tiers(items: list[dict]) -> list[dict]:
    scored = [i for i in items if i.get("actualLabel") and i.get("probs")]
    tiers: list[dict] = []
    for threshold in (50.0, 55.0, 60.0, 65.0):
        subset = [
            i
            for i in scored
            if max(float(i["probs"]["teamWin"]), float(i["probs"]["draw"]), float(i["probs"]["oppWin"]))
            >= threshold
        ]
        if not subset:
            continue
        correct = sum(1 for i in subset if i.get("correct") is True)
        tiers.append(
            {
                "minTopProb": threshold,
                "n": len(subset),
                "accuracy": round(correct / len(subset), 4),
            }
        )
    return tiers


def _evaluation_for_site(league: str) -> dict:
    selection = load_selection_meta(league) or {}
    rolling = load_rolling_backtest(league)
    return {
        "testMetrics": selection.get("test_metrics"),
        "rollingBacktest": rolling or None,
    }


def _confident_picks(items: list[dict]) -> list[dict]:
    picks = []
    for item in items:
        probs = item.get("probs") or {}
        top = max(
            float(probs.get("teamWin", 0)),
            float(probs.get("draw", 0)),
            float(probs.get("oppWin", 0)),
        )
        if top >= CONFIDENT_PICK_MIN:
            picks.append({**item, "topProb": round(top, 2)})
    picks.sort(key=lambda x: (-x["topProb"], x["date"]))
    return picks[:20]


def _model_meta_for_site(league: str) -> dict:
    selection = load_selection_meta(league) or {}
    strategy = selection.get("selected_model")
    reselect_days = selection.get("reselect_days", MODEL_RESELECT_DAYS)
    split = selection.get("split") or {}
    return {
        "selectedModel": strategy,
        "modelLabel": model_display_name(strategy),
        "modelEvaluatedAt": selection.get("evaluated_at"),
        "modelFittedAt": selection.get("fitted_at"),
        "modelReselectDays": reselect_days,
        "modelTestAccuracy": selection.get("best_accuracy"),
        "modelTestRps": (selection.get("test_metrics") or {}).get("rps"),
        "modelSplit": split,
        "pipelineNote": (
            f"Weekly scrape and refit on the latest results; strategy comparison "
            f"re-runs about every {reselect_days} days."
        ),
    }


def _has_actual(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def _partition_predictions(
    df: pd.DataFrame, today: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into upcoming, awaiting result (played, not scraped yet) and scored history."""
    scored = _has_actual(df["Actual"])
    unscored_past = ~scored & (df["Date"] < today)
    recent = df["Date"] >= today - pd.Timedelta(days=AWAITING_RESULT_DAYS)
    upcoming = df[~scored & (df["Date"] >= today)].copy()
    awaiting = df[unscored_past & recent].copy()
    history_df = df[scored & (df["Date"] < today)].copy()
    return upcoming, awaiting, history_df


def _league_next_fixtures_for_export(
    matches: pd.DataFrame, next_matches: pd.DataFrame, league: str
) -> pd.DataFrame:
    from fplpredict.data.matches_clean import clean_next_matches
    from fplpredict.model.features import build_team_name_mapping

    mapping = build_team_name_mapping(matches, league=league)
    nm = clean_next_matches(next_matches.loc[next_matches["Comp"] == league].copy())
    if nm.empty:
        return nm
    nm["Team"] = nm["Team"].map(mapping).apply(lambda x: normalize_team_name(x) or x)
    nm["Opponent"] = nm["Opponent"].map(mapping).apply(
        lambda x: normalize_team_name(x) or x
    )
    nm["Date"] = pd.to_datetime(nm["Date"])
    if "Venue" in nm.columns:
        nm["is_home"] = (nm["Venue"] == "Home").astype(int)
    else:
        nm["is_home"] = 1
    return nm


def _export_league_payload(league: str) -> dict:
    csv_path = league_predictions_csv(league)
    if not csv_path.exists():
        raise FileNotFoundError(f"No predictions CSV for {league}: {csv_path}")
    df = drop_invalid_fixtures(dedupe_true_duplicates(pd.read_csv(csv_path, index_col=0)))
    df["Date"] = pd.to_datetime(df["Date"])
    valid_upcoming_keys: set[tuple] = set()
    past_matches = future_matches = pd.DataFrame()
    matches_raw = read_csv("matches.csv")
    next_raw = read_csv("next_matches.csv")
    from fplpredict.data.football_data import completed_results_from_football_data

    completed = completed_match_results(matches_raw, league)
    fd_completed = completed_results_from_football_data(matches_raw, league)
    df = backfill_actuals(df, completed)
    df = backfill_actuals(df, fd_completed)
    df = dedupe_true_duplicates(df)
    try:
        past_matches, future_matches, _ = build_feature_matrix(
            matches_raw,
            next_raw,
            read_csv("history.csv"),
            league=league,
        )
        df = backfill_actuals(df, past_matches)
        df = propagate_fixture_actuals(df)
        df = dedupe_true_duplicates(df)
        if not future_matches.empty:
            for _, row in future_matches.iterrows():
                team = normalize_team_name(_clean_str(row.get("Team")))
                opp = normalize_team_name(_clean_str(row.get("Opponent")))
                if team and opp:
                    valid_upcoming_keys.add(
                        (team, opp, pd.Timestamp(row["Date"]).date())
                    )
    except Exception:
        log.warning("[%s] Feature matrix unavailable for export extras", league, exc_info=True)
        past_matches = completed
        future_matches = _league_next_fixtures_for_export(matches_raw, next_raw, league)

    today = pd.Timestamp.now().normalize()
    upcoming, awaiting, history_df = _partition_predictions(df, today)
    if valid_upcoming_keys:

        def _upcoming_row_key(r) -> tuple:
            team = normalize_team_name(_clean_str(r.get("Team")))
            opp = normalize_team_name(_clean_str(r.get("Opponent")))
            if not team or not opp:
                return ()
            return (team, opp, pd.Timestamp(r["Date"]).date())

        upcoming = upcoming[
            upcoming.apply(lambda r: _upcoming_row_key(r) in valid_upcoming_keys, axis=1)
        ]
    if not upcoming.empty and "Date" in upcoming.columns:
        upcoming = _dedupe_fixtures(upcoming.sort_values("Date"), keep="latest")
    else:
        upcoming = upcoming.iloc[0:0]
    # Same choice as history, so the pick shown now is the one that gets scored.
    if not awaiting.empty and "Date" in awaiting.columns:
        awaiting = _dedupe_fixtures(awaiting.sort_values("Date"), keep="earliest")
    else:
        awaiting = awaiting.iloc[0:0]
    if not history_df.empty and "Date" in history_df.columns:
        history_df = _dedupe_fixtures(
            history_df.sort_values("Date", ascending=False), keep="earliest"
        )
    else:
        history_df = history_df.iloc[0:0]

    seasons = set()
    if not past_matches.empty and "Season" in past_matches.columns:
        seasons |= {int(s) for s in past_matches["Season"].dropna().unique()}
    market_lookup = combined_market_lookup(league, seasons) if seasons else {}
    home_map = _fixture_home_map(past_matches, future_matches)

    upcoming_items = _build_items(
        upcoming, market_lookup=market_lookup, home_map=home_map
    )
    awaiting_items = _build_items(
        awaiting, market_lookup=market_lookup, home_map=home_map
    )
    history_items = _build_items(
        history_df, market_lookup=market_lookup, home_map=home_map
    )
    history_items.sort(key=lambda x: x["date"], reverse=True)
    scored_keys = {
        _item_fixture_key(i)
        for i in history_items
        if i.get("actualLabel")
    }
    scored_keys.discard(())
    upcoming_items = _drop_items_already_scored(upcoming_items, scored_keys)
    awaiting_items = _drop_items_already_scored(awaiting_items, scored_keys)
    confident = _confident_picks(upcoming_items + awaiting_items)

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "league": league,
        "upcoming": upcoming_items,
        "awaitingResult": awaiting_items,
        "history": history_items,
        "metrics": _metrics_from_history(history_items),
        "confidenceTiers": _confidence_tiers(history_items),
        "weeklyPerformance": weekly_performance_by_gameweek(history_df, league),
        "modelTimeline": model_strategy_timeline(league),
        "evaluation": _evaluation_for_site(league),
        "confidentPicks": confident,
        "meta": {
            "upcomingCount": len(upcoming_items),
            "awaitingResultCount": len(awaiting_items),
            "historyCount": len(history_items),
            "recentPreviewSize": 10,
            "historyPageSize": 20,
            **_model_meta_for_site(league),
            "positionsNote": (
                "Table positions are computed from this season's completed matches before "
                "each fixture. A missing # means the season has no completed matches yet."
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
        league_web_json(league).write_text(encoded, encoding="utf-8")
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
