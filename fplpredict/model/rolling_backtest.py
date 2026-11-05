"""Walk-forward evaluation: each season scored using only prior seasons."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from fplpredict.config import MODEL_DATA_SEASONS, VAL_SEASON_COUNT
from fplpredict.model.calibration import apply_probability_calibration
from fplpredict.model.draw_tuning import predict_with_draw_threshold
from fplpredict.model.ensemble import (
    _production_training_flags,
    _score,
    fit_production_predictor,
    load_past_matches,
    load_selection_meta,
    selected_features,
)
from fplpredict.model.market import market_baseline
from fplpredict.model.metrics import ranked_probability_score
from fplpredict.model.preprocess import transform_features
from fplpredict.model.splits import completed_past_matches, sorted_seasons
from fplpredict.paths import league_artifacts_dir

log = logging.getLogger(__name__)


def rolling_backtest_rows(league: str) -> list[dict[str, Any]]:
    meta = load_selection_meta(league)
    if not meta or not meta.get("selected_model"):
        return []
    strategy, tuned_weights = _production_training_flags(meta)
    features = selected_features(league, load_past_matches(league).columns)
    t_draw = (meta.get("decision_thresholds") or {}).get("t_draw")
    past = completed_past_matches(load_past_matches(league))
    seasons = sorted_seasons(past)
    min_train = MODEL_DATA_SEASONS - VAL_SEASON_COUNT - 1
    rows: list[dict[str, Any]] = []
    for test_season in seasons:
        if test_season not in seasons or seasons.index(test_season) < min_train:
            continue
        prior = [s for s in seasons if s < test_season]
        if len(prior) < min_train:
            continue
        window = prior[-MODEL_DATA_SEASONS:]
        train_df = past[past["Season"].isin(window)]
        test_df = past[past["Season"] == test_season]
        if train_df.empty or test_df.empty:
            continue
        try:
            predictor, scaler = fit_production_predictor(
                strategy, train_df, features, tuned_weights
            )
        except Exception:
            log.warning("[%s] Rolling backtest skip season %s", league, test_season, exc_info=True)
            continue
        proba = apply_probability_calibration(
            np.clip(
                predictor.predict_proba(transform_features(scaler, test_df, features)),
                0,
                1,
            ),
            meta,
        )
        y = test_df["Target"].astype(int).to_numpy()
        if t_draw is not None:
            pred = predict_with_draw_threshold(proba, float(t_draw))
            acc = float(accuracy_score(y, pred))
        else:
            acc = float(accuracy_score(y, np.argmax(proba, axis=1)))
        row = {
            "seasonEnd": int(test_season),
            "n": int(len(y)),
            "accuracy": acc,
            "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
            "rps": ranked_probability_score(y, proba, sklearn_order=True),
        }
        market = market_baseline(test_df, proba, league)
        if market:
            row["market_accuracy"] = market["accuracy"]
            row["market_log_loss"] = market["log_loss"]
        rows.append(row)
    return rows


def write_rolling_backtest(league: str) -> list[dict[str, Any]]:
    rows = rolling_backtest_rows(league)
    path = league_artifacts_dir(league) / "rolling_backtest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seasons": rows}, indent=2), encoding="utf-8")
    return rows


def load_rolling_backtest(league: str) -> list[dict[str, Any]]:
    path = league_artifacts_dir(league) / "rolling_backtest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("seasons") or []
