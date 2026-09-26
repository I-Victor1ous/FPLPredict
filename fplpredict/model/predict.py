import json

import numpy as np
import pandas as pd

from fplpredict.config import MODEL_FEATURES, RESULT_LABELS
from fplpredict.model.ensemble import ensure_production_model
from fplpredict.model.features import build_feature_matrix
from fplpredict.data_io import read_csv
from fplpredict.paths import ARTIFACTS_DIR, PREDICTIONS_CSV
from fplpredict.predictions_util import (
    backfill_actuals,
    dedupe_true_duplicates,
    normalize_predictions_frame,
)


def _merge_predictions(
    result: pd.DataFrame, past_matches: pd.DataFrame, predictions_path
) -> pd.DataFrame:
    frames = []
    for path in (predictions_path,):
        if path.exists():
            try:
                frames.append(pd.read_csv(path, index_col=0))
            except pd.errors.EmptyDataError:
                pass
    if frames:
        existing = dedupe_true_duplicates(pd.concat(frames, ignore_index=True))
    else:
        existing = pd.DataFrame()

    past_home = past_matches[["Date", "Team_position", "Opp_position", "Target"]]
    past_away = past_home.rename(
        columns={"Team_position": "Opp_position", "Opp_position": "Team_position"}
    )
    swap = {1: 2, 2: 1, 0: 0}
    past_away["Target"] = past_away["Target"].map(swap)
    past_selected = pd.concat([past_home, past_away], ignore_index=True)

    if not existing.empty:
        existing["Date"] = pd.to_datetime(existing["Date"])
        existing = existing.merge(
            past_selected[["Date", "Team_position", "Opp_position", "Target"]],
            on=["Date", "Team_position", "Opp_position"],
            how="left",
        )
        if "Actual" in existing.columns:
            existing["Actual"] = existing["Actual"].combine_first(existing["Target"])
        else:
            existing["Actual"] = existing["Target"]
        existing = existing.drop(columns=["Target"], errors="ignore")
        label_map = {**RESULT_LABELS, **{v: v for v in RESULT_LABELS.values()}}
        existing["Actual"] = existing["Actual"].map(label_map)

    result = normalize_predictions_frame(result)

    if not existing.empty:
        existing_norm = normalize_predictions_frame(existing)
        merge_keys = ["Date", "Team", "Opponent"]
        mask = ~result.set_index(merge_keys).index.isin(
            existing_norm.set_index(merge_keys).index
        )
        new_rows = result[mask]
        merged = pd.concat(
            [df for df in [new_rows, existing_norm] if not df.empty],
            ignore_index=True,
        )
    else:
        merged = result.copy()
        merged["Actual"] = None

    merged = dedupe_true_duplicates(merged)
    return merged


def run_prediction_pipeline(
    save_artifacts: bool = True, force_model_reselect: bool = False
) -> pd.DataFrame:
    matches = read_csv("matches.csv")
    next_matches = read_csv("next_matches.csv")
    history = read_csv("history.csv")

    past_matches, future_matches, _team_mapping_dict = build_feature_matrix(
        matches, next_matches, history
    )
    past_matches["Date"] = pd.to_datetime(past_matches["Date"])

    features = [f for f in MODEL_FEATURES if f in past_matches.columns]
    missing = set(MODEL_FEATURES) - set(features)
    if missing:
        raise RuntimeError(f"Missing model features in data: {sorted(missing)}")

    predictor, scaler = ensure_production_model(
        past_matches,
        features,
        force_reselect=force_model_reselect,
    )

    predict_x = future_matches[features].fillna(0)
    if predict_x.empty:
        raise RuntimeError("No upcoming fixtures found in next_matches.csv")

    predict_scaled = scaler.transform(predict_x)
    voter_preds = predictor.predict(predict_scaled)
    voter_proba = np.clip(predictor.predict_proba(predict_scaled), 0, 1)
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
        }
    )

    merged = _merge_predictions(result, past_matches, PREDICTIONS_CSV)
    merged = backfill_actuals(merged, past_matches)
    merged = dedupe_true_duplicates(merged)
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PREDICTIONS_CSV)

    if save_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS_DIR / "features.json").write_text(
            json.dumps(features, indent=2), encoding="utf-8"
        )

    return merged
