import pandas as pd

from fplpredict.performance_export import (
    PL_LEGACY_STRATEGY,
    model_strategy_timeline,
    weekly_performance_by_gameweek,
)


def test_weekly_performance_groups_by_gameweek(monkeypatch):
    def fake_read(name: str):
        if name != "matches.csv":
            raise FileNotFoundError
        return pd.DataFrame(
            [
                {
                    "Comp": "Test League",
                    "Date": "2025-09-28",
                    "Team": "A",
                    "Opponent": "B",
                    "Round": "Matchweek 4",
                    "Season": 2026,
                },
                {
                    "Comp": "Test League",
                    "Date": "2025-09-27",
                    "Team": "C",
                    "Opponent": "D",
                    "Round": "Matchweek 4",
                    "Season": 2026,
                },
                {
                    "Comp": "Test League",
                    "Date": "2025-10-06",
                    "Team": "E",
                    "Opponent": "F",
                    "Round": "Matchweek 5",
                    "Season": 2026,
                },
            ]
        )

    monkeypatch.setattr("fplpredict.performance_export.read_csv", fake_read)

    df = pd.DataFrame(
        [
            {
                "Date": "2025-09-28",
                "Team": "A",
                "Opponent": "B",
                "Pred": "Win",
                "Actual": "Win",
                "Model_strategy": "lr",
            },
            {
                "Date": "2025-09-27",
                "Team": "C",
                "Opponent": "D",
                "Pred": "Loss",
                "Actual": "Win",
                "Model_strategy": "lr",
            },
            {
                "Date": "2025-10-06",
                "Team": "E",
                "Opponent": "F",
                "Pred": "Draw",
                "Actual": "Draw",
                "Model_strategy": "svc",
            },
        ]
    )
    weeks = weekly_performance_by_gameweek(df, "Test League")
    assert len(weeks) == 2
    assert weeks[0]["gameweek"] == "Matchweek 4"
    assert weeks[0]["n"] == 2
    assert weeks[0]["correct"] == 1
    assert weeks[1]["gameweek"] == "Matchweek 5"


def test_pl_legacy_strategy_on_old_predictions():
    df = pd.DataFrame(
        [
            {
                "Date": "2026-09-20",
                "Team": "A",
                "Opponent": "B",
                "Pred": "Win",
                "Actual": "Win",
                "Model_strategy": "lr",
                "Predicted_at": "2026-09-15T10:00:00+00:00",
            },
        ]
    )
    weeks = weekly_performance_by_gameweek(df, "Premier League")
    assert weeks[0]["modelStrategy"] == PL_LEGACY_STRATEGY


def test_model_timeline_collapses_consecutive_same_strategy(tmp_path, monkeypatch):
    log = tmp_path / "predictions-log-test-league.jsonl"
    log.write_text(
        "\n".join(
            [
                '{"modelStrategy":"lr","modelEvaluatedAt":"2026-01-01T10:00:00+00:00"}',
                '{"modelStrategy":"lr","modelEvaluatedAt":"2026-01-08T10:00:00+00:00"}',
                '{"modelStrategy":"svc","modelEvaluatedAt":"2026-02-01T10:00:00+00:00"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_meta(_league: str):
        return None

    monkeypatch.setattr(
        "fplpredict.performance_export.load_selection_meta",
        fake_meta,
    )
    monkeypatch.setattr(
        "fplpredict.performance_export.league_prediction_log",
        lambda _league: log,
    )

    segments = model_strategy_timeline("Test League")
    assert len(segments) == 2
    assert segments[0]["strategy"] == "lr"
    assert segments[1]["strategy"] == "svc"


def test_premier_league_legacy_timeline_segment(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(
        "fplpredict.performance_export.league_prediction_log",
        lambda _league: Path("/nonexistent/predictions-log.jsonl"),
    )
    monkeypatch.setattr(
        "fplpredict.performance_export.load_selection_meta",
        lambda _league: {
            "selected_model": "lr",
            "evaluated_at": "2026-10-01T12:00:00+00:00",
        },
    )
    segments = model_strategy_timeline("Premier League")
    assert segments[0]["strategy"] == PL_LEGACY_STRATEGY
    assert segments[0]["note"]
    assert segments[0]["to"] == "2026-09-30T00:00:00Z"
    assert segments[-1]["strategy"] == "lr"
