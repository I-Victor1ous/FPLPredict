import json

import pandas as pd

from fplpredict.predictions_log import (
    apply_log_to_frame,
    merge_with_integrity,
    restore_missing_predictions_from_log,
)
from fplpredict.predictions_util import drop_invalid_fixtures


def test_merge_does_not_replace_scored_row_probs():
    today = pd.Timestamp("2026-10-01")
    existing = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-09-20"),
                "Team": "Arsenal",
                "Opponent": "Chelsea",
                "Pred": "Win",
                "Team_win%": 55.0,
                "Draw%": 25.0,
                "Opp_win%": 20.0,
                "Actual": "Win",
                "Predicted_at": "2026-09-15T00:00:00+00:00",
            }
        ]
    )
    new = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-09-20"),
                "Team": "Arsenal",
                "Opponent": "Chelsea",
                "Pred": "Draw",
                "Team_win%": 10.0,
                "Draw%": 80.0,
                "Opp_win%": 10.0,
            }
        ]
    )
    merged = merge_with_integrity(new, existing, today=today)
    row = merged.iloc[0]
    assert row["Team_win%"] == 55.0
    assert row["Pred"] == "Win"


def test_merge_keeps_started_fixture_without_result():
    today = pd.Timestamp("2026-10-18")
    existing = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-17"),
                "Team": "Newcastle United",
                "Opponent": "Aston Villa",
                "Pred": "Win",
                "Team_win%": 39.0,
                "Draw%": 30.0,
                "Opp_win%": 31.0,
                "Predicted_at": "2026-10-14T15:42:00+00:00",
            }
        ]
    )
    new = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-17"),
                "Team": "Newcastle United",
                "Opponent": "Aston Villa",
                "Pred": "Loss",
                "Team_win%": 20.0,
                "Draw%": 30.0,
                "Opp_win%": 50.0,
            }
        ]
    )
    merged = merge_with_integrity(new, existing, today=today)
    merged = drop_invalid_fixtures(merged)
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["Team"] == "Newcastle United"
    assert row["Pred"] == "Win"
    assert row["Team_win%"] == 39.0


def test_restore_missing_predictions_from_log(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions-log-premier-league.jsonl"
    rec = {
        "key": ["2026-10-17", "Newcastle United", "Aston Villa"],
        "predictedAt": "2026-10-14T15:42:00+00:00",
        "modelStrategy": "lr",
        "modelEvaluatedAt": "2026-10-01T00:00:00+00:00",
        "date": "2026-10-17",
        "team": "Newcastle United",
        "opponent": "Aston Villa",
        "pred": "Win",
        "teamWinPct": 39.0,
        "drawPct": 30.0,
        "oppWinPct": 31.0,
    }
    log_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "fplpredict.predictions_log.prediction_log_path",
        lambda _league: log_path,
    )

    restored = restore_missing_predictions_from_log(pd.DataFrame(), "Premier League")
    assert len(restored) == 1
    assert restored.iloc[0]["Pred"] == "Win"


def test_apply_log_restores_pick_for_started_unscored(tmp_path, monkeypatch):
    log_path = tmp_path / "log.jsonl"
    rec = {
        "key": ["2026-10-17", "Newcastle United", "Aston Villa"],
        "predictedAt": "2026-10-14T15:42:00+00:00",
        "modelStrategy": "lr",
        "modelEvaluatedAt": None,
        "date": "2026-10-17",
        "team": "Newcastle United",
        "opponent": "Aston Villa",
        "pred": "Win",
        "teamWinPct": 39.0,
        "drawPct": 30.0,
        "oppWinPct": 31.0,
    }
    log_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "fplpredict.predictions_log.prediction_log_path",
        lambda _league: log_path,
    )
    df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-10-17"),
                "Team": "Newcastle United",
                "Opponent": "Aston Villa",
                "Pred": "Loss",
                "Team_win%": 20.0,
                "Draw%": 30.0,
                "Opp_win%": 50.0,
                "Predicted_at": "2026-10-18T01:00:00+00:00",
            }
        ]
    )
    fixed = apply_log_to_frame(df, "Premier League", today=pd.Timestamp("2026-10-18"))
    assert fixed.iloc[0]["Pred"] == "Win"
    assert fixed.iloc[0]["Team_win%"] == 39.0
