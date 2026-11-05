import json
import logging

import numpy as np
import pandas as pd

from fplpredict.config import RESULT_LABELS, active_leagues
from fplpredict.data_io import read_csv
from fplpredict.model.calibration import apply_probability_calibration
from fplpredict.model.draw_tuning import predict_with_draw_threshold
from fplpredict.model.ensemble import ensure_production_model, load_selection_meta
from fplpredict.model.features import build_feature_matrix
from fplpredict.model.preprocess import transform_features
from fplpredict.paths import league_artifacts_dir, league_predictions_csv
from fplpredict.data.football_data import completed_results_from_football_data
from fplpredict.predictions_log import (
    append_prediction_records,
    apply_log_to_frame,
    merge_with_integrity,
    restore_missing_predictions_from_log,
)
from fplpredict.predictions_store import migrate_legacy_prediction_files
from fplpredict.predictions_util import (
    backfill_actuals,
    completed_match_results,
    dedupe_true_duplicates,
    drop_invalid_fixtures,
    normalize_predictions_frame,
)

log = logging.getLogger(__name__)


def run_prediction_pipeline(
    league: str,
    save_artifacts: bool = True,
    force_model_reselect: bool = False,
) -> pd.DataFrame:
    matches = read_csv("matches.csv")
    next_matches = read_csv("next_matches.csv")
    history = read_csv("history.csv")
    completed_results = completed_match_results(matches, league)
    fd_results = completed_results_from_football_data(matches, league)

    past_matches, future_matches, _team_mapping_dict = build_feature_matrix(
        matches, next_matches, history, league=league
    )
    past_matches["Date"] = pd.to_datetime(past_matches["Date"])

    predictor, scaler, features = ensure_production_model(
        past_matches,
        league=league,
        force_reselect=force_model_reselect,
    )
    missing = [f for f in features if f not in future_matches.columns]
    if missing:
        raise RuntimeError(f"Missing model features for upcoming fixtures: {missing}")

    selection = load_selection_meta(league) or {}
    model_strategy = selection.get("selected_model")
    model_evaluated_at = selection.get("evaluated_at")

    predictions_path = league_predictions_csv(league)
    existing = pd.DataFrame()
    if predictions_path.exists():
        try:
            existing = dedupe_true_duplicates(
                pd.read_csv(predictions_path, index_col=0)
            )
            existing = apply_log_to_frame(existing, league)
            existing = restore_missing_predictions_from_log(existing, league)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()

    if future_matches.empty:
        log.warning("[%s] No upcoming fixtures in next_matches.csv", league)
        if predictions_path.exists():
            merged = restore_missing_predictions_from_log(existing, league)
            merged = backfill_actuals(merged, past_matches)
            merged = backfill_actuals(merged, completed_results)
            merged = backfill_actuals(merged, fd_results)
            merged = drop_invalid_fixtures(dedupe_true_duplicates(merged))
            merged.to_csv(predictions_path)
            return merged
        raise RuntimeError(f"No upcoming fixtures found for {league}")

    predict_scaled = transform_features(scaler, future_matches, features)
    voter_proba = apply_probability_calibration(
        np.clip(predictor.predict_proba(predict_scaled), 0, 1),
        selection,
    )
    thresholds = (selection.get("decision_thresholds") or {}) if selection else {}
    t_draw = thresholds.get("t_draw")
    if t_draw is not None:
        voter_preds = predict_with_draw_threshold(voter_proba, float(t_draw))
    else:
        voter_preds = predictor.predict(predict_scaled)
    preds = [RESULT_LABELS[int(x)] for x in voter_preds]

    result = pd.DataFrame(
        {
            "Date": future_matches["Date"],
            "Team_position": future_matches["Team_position"],
            "Team": future_matches["Team"],
            "Pred": preds,
            "Opponent": future_matches["Opponent"],
            "Opp_position": future_matches["Opp_position"],
            "Team_win%": (voter_proba[:, 2] * 100).round(2),
            "Opp_win%": (voter_proba[:, 1] * 100).round(2),
            "Draw%": (voter_proba[:, 0] * 100).round(2),
            "Model_strategy": model_strategy,
            "Model_evaluated_at": model_evaluated_at,
        }
    )
    result = normalize_predictions_frame(result)

    append_prediction_records(
        league,
        result,
        model_strategy=model_strategy,
        model_evaluated_at=model_evaluated_at,
    )

    merged = merge_with_integrity(result, existing)
    merged = restore_missing_predictions_from_log(merged, league)
    merged = backfill_actuals(merged, past_matches)
    merged = backfill_actuals(merged, completed_results)
    merged = backfill_actuals(merged, fd_results)
    merged = apply_log_to_frame(merged, league)
    merged = drop_invalid_fixtures(dedupe_true_duplicates(merged))
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(predictions_path)

    if save_artifacts:
        art_dir = league_artifacts_dir(league)
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "features.json").write_text(
            json.dumps(features, indent=2), encoding="utf-8"
        )

    return merged


def run_all_league_predictions(
    force_model_reselect: bool = False,
    errors: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Predict every active league; failures are logged and recorded in ``errors``."""
    migrate_legacy_prediction_files()
    results: dict[str, pd.DataFrame] = {}
    for league in active_leagues():
        log.info("Prediction pipeline for %s", league)
        try:
            results[league] = run_prediction_pipeline(
                league=league, force_model_reselect=force_model_reselect
            )
        except Exception as exc:
            log.exception("[%s] Prediction pipeline failed; skipping league.", league)
            if errors is not None:
                errors[league] = f"{type(exc).__name__}: {exc}"
    if not results:
        raise RuntimeError("No league completed the prediction pipeline.")
    return results
