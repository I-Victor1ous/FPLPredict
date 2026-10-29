"""End to end on a synthetic league: features → selection → comparison → fit → predict → export."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from fplpredict import export
from fplpredict.config import CANDIDATE_FEATURES, RESULT_LABELS
from fplpredict.model import ensemble
from fplpredict.model.feature_selection import forward_select_features
from fplpredict.model.features import build_feature_matrix
from fplpredict.model.preprocess import transform_features
from fplpredict.model.splits import completed_past_matches, restrict_last_n_seasons
from fplpredict.predictions_util import normalize_predictions_frame
from tests.contract import league_payload_problems
from tests.synthetic import LEAGUE, make_league


def _tiny_estimators():
    return [
        ("rf", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)),
        ("xgb", XGBClassifier(n_estimators=10, max_depth=2, verbosity=0, random_state=0)),
        ("svc", SVC(C=0.2, kernel="linear", probability=True, random_state=0)),
        ("lr", LogisticRegression(max_iter=500)),
    ]


@pytest.fixture(scope="module")
def league_frames():
    matches, upcoming, history = make_league(seed=1)
    past, future, _ = build_feature_matrix(matches, upcoming, history, league=LEAGUE)
    past["Date"] = pd.to_datetime(past["Date"])
    return matches, upcoming, history, past, future


def _predictions_frame(rows: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    return normalize_predictions_frame(
        pd.DataFrame(
            {
                "Date": rows["Date"].to_numpy(),
                "Team_position": rows["Team_position"].to_numpy(),
                "Team": rows["Team"].to_numpy(),
                "Pred": [RESULT_LABELS[int(i)] for i in proba.argmax(axis=1)],
                "Opponent": rows["Opponent"].to_numpy(),
                "Opp_position": rows["Opp_position"].to_numpy(),
                "Team_win%": (proba[:, 2] * 100).round(2),
                "Opp_win%": (proba[:, 1] * 100).round(2),
                "Draw%": (proba[:, 0] * 100).round(2),
                "Model_strategy": "lr",
                "Model_evaluated_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )


def test_pipeline_end_to_end(league_frames, monkeypatch, tmp_path):
    monkeypatch.setattr(ensemble, "build_base_estimators", _tiny_estimators)
    matches, upcoming, history, past, future = league_frames

    modeling = restrict_last_n_seasons(completed_past_matches(past))
    train_df, val_df, test_df, _, _ = ensemble.partition_frames(modeling)
    candidates = [c for c in CANDIDATE_FEATURES if c in modeling.columns][:12]
    features, history_log = forward_select_features(train_df, val_df, candidates, max_features=3)
    assert 1 <= len(features) <= 3 and len(history_log) == len(features)

    comparison = ensemble.compare_candidates(ensemble.build_eval_split(modeling, features))
    names = {row["model"] for row in comparison.val_scores}
    assert {"rf", "lr", "soft_vote_val_log_loss", "stacking_lr_meta_oof"} <= names
    assert comparison.best_model in names
    for row in comparison.scores:
        assert np.isfinite(row["log_loss"]) and 0 <= row["accuracy"] <= 1

    predictor, prep = ensemble.fit_production_predictor(
        comparison.best_model, past, features, comparison.tuned_weights
    )
    proba = predictor.predict_proba(transform_features(prep, future, features))
    assert proba.shape == (len(future), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    test_proba = predictor.predict_proba(transform_features(prep, test_df, features))
    predictions = pd.concat(
        [_predictions_frame(test_df, test_proba), _predictions_frame(future, proba)],
        ignore_index=True,
    )
    csv_path = tmp_path / "predictions.csv"
    predictions.to_csv(csv_path)

    frames = {"matches.csv": matches, "next_matches.csv": upcoming, "history.csv": history}
    monkeypatch.setattr(export, "league_predictions_csv", lambda league: csv_path)
    monkeypatch.setattr(export, "read_csv", lambda name: frames[name].copy())
    monkeypatch.setattr(export, "build_feature_matrix", lambda *a, **k: (past, future, {}))
    monkeypatch.setattr(export, "load_selection_meta", lambda league: {"selected_model": "lr"})
    monkeypatch.setattr(export, "model_strategy_timeline", lambda league: [])
    monkeypatch.setattr(export, "weekly_performance_by_gameweek", lambda df, league: [])

    payload = export._export_league_payload(LEAGUE)
    assert league_payload_problems(payload) == []
    assert len(payload["upcoming"]) == len(future)
    assert len(payload["history"]) == len(test_df)
    assert all(item["actualLabel"] for item in payload["history"])
