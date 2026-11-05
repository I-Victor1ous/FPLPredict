"""Walk-forward strategy comparison across full seasons (not a single val year)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from fplpredict.config import (
    WF_MIN_TEST_ROWS,
    WF_TRAIN_WINDOW_SEASONS,
    current_season_end,
)
from fplpredict.model.ensemble import (
    ComparisonResult,
    SplitData,
    _evaluate_candidates,
    build_eval_split,
)
from fplpredict.model.metrics import home_win_baseline_accuracy
from fplpredict.model.preprocess import feature_frame, make_preprocessor, transform_features
from fplpredict.model.sample_weights import training_sample_weights
from fplpredict.model.selection import pick_best_score_row
from fplpredict.model.splits import SeasonSplitMeta, sorted_seasons

log = logging.getLogger(__name__)

MIN_FOLD_VAL_ROWS = 20
MIN_PRIOR_SEASONS = WF_TRAIN_WINDOW_SEASONS


def walkforward_test_seasons(modeling: pd.DataFrame) -> list[int]:
    """Season end-years used as held-out folds (each scored with only prior data)."""
    seasons = sorted_seasons(modeling)
    current = current_season_end()
    eligible: list[int] = []
    for season in seasons:
        if season >= current:
            continue
        prior = [s for s in seasons if s < season]
        if len(prior) < MIN_PRIOR_SEASONS:
            continue
        n_rows = int((modeling["Season"] == season).sum())
        if n_rows < WF_MIN_TEST_ROWS:
            continue
        eligible.append(season)
    return eligible


def build_fold_split(
    modeling: pd.DataFrame, features: list[str], test_season: int
) -> SplitData:
    """Train on prior window (last season = val for weight tuning), test on ``test_season``."""
    seasons = sorted_seasons(modeling)
    if test_season not in seasons:
        raise RuntimeError(f"Test season {test_season} not in modeling data")
    prior = [s for s in seasons if s < test_season]
    window = prior[-WF_TRAIN_WINDOW_SEASONS:]
    if len(window) < MIN_PRIOR_SEASONS:
        raise RuntimeError(
            f"Need at least {MIN_PRIOR_SEASONS} seasons before {test_season}; got {window}"
        )
    val_season = window[-1]
    train_seasons = tuple(window[:-1])
    train_df = modeling[modeling["Season"].isin(train_seasons)].sort_values("Date")
    val_df = modeling[modeling["Season"] == val_season].sort_values("Date")
    test_df = modeling[modeling["Season"] == test_season].sort_values("Date")
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(f"Empty train/val/test for walk-forward fold {test_season}")

    test_start = test_df["Date"].min()
    meta = SeasonSplitMeta(
        all_seasons=tuple(window) + (test_season,),
        train_seasons=train_seasons,
        val_season=val_season,
        test_season=test_season,
    )

    preprocessor = make_preprocessor()
    train_X = preprocessor.fit_transform(feature_frame(train_df, features))
    val_X = transform_features(preprocessor, val_df, features)
    test_X = transform_features(preprocessor, test_df, features)
    train_y = train_df["Target"].astype(int).to_numpy()
    val_y = val_df["Target"].astype(int).to_numpy()
    test_y = test_df["Target"].astype(int).to_numpy()
    train_w = training_sample_weights(train_y)

    return SplitData(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        train_w=train_w,
        split_meta=meta,
        test_start=test_start,
    )


def _accumulate_fold_scores(
    agg: dict[str, dict[str, float]],
    fold_test_scores: list[dict[str, Any]],
) -> None:
    """Add one season's test metrics (equal weight per fold in the final average)."""
    for row in fold_test_scores:
        name = row["model"]
        n = int(row["n"])
        bucket = agg.setdefault(
            name,
            {
                "log_loss_sum": 0.0,
                "rps_sum": 0.0,
                "accuracy_sum": 0.0,
                "n_sum": 0.0,
                "folds": 0.0,
            },
        )
        bucket["log_loss_sum"] += float(row["log_loss"])
        bucket["rps_sum"] += float(row["rps"])
        bucket["accuracy_sum"] += float(row["accuracy"])
        bucket["n_sum"] += n
        bucket["folds"] += 1.0


def _average_fold_score_rows(
    agg: dict[str, dict[str, float]], folds_used: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, a in agg.items():
        folds = int(a["folds"])
        if folds <= 0:
            continue
        rows.append(
            {
                "model": model,
                "log_loss": a["log_loss_sum"] / folds,
                "accuracy": a["accuracy_sum"] / folds,
                "rps": a["rps_sum"] / folds,
                "n": int(a["n_sum"]),
                "folds": folds_used,
            }
        )
    return rows


def compare_candidates_walkforward(
    modeling: pd.DataFrame, features: list[str]
) -> ComparisonResult:
    """Pick strategy by mean per-season log loss across walk-forward test seasons."""
    test_seasons = walkforward_test_seasons(modeling)
    canonical_split = build_eval_split(modeling, features)
    val_scores, canonical_test_scores, tuned_weights = _evaluate_candidates(
        canonical_split
    )

    if not test_seasons:
        log.warning("No walk-forward folds; selecting on validation season only.")
        best = pick_best_score_row(val_scores)
        winner_test = next(s for s in canonical_test_scores if s["model"] == best["model"])
        winner_test = {
            **winner_test,
            "home_win_baseline_accuracy": home_win_baseline_accuracy(
                canonical_split.test_y
            ),
        }
        return ComparisonResult(
            val_scores=val_scores,
            scores=canonical_test_scores,
            tuned_weights=tuned_weights,
            best_model=best["model"],
            best_accuracy=winner_test["accuracy"],
            test_metrics=winner_test,
            selection_criterion="val_log_loss",
            walkforward_scores=[],
            walkforward_folds=[],
        )

    agg: dict[str, dict[str, float]] = {}
    folds_used: list[dict[str, Any]] = []
    for test_season in test_seasons:
        try:
            split = build_fold_split(modeling, features, test_season)
        except RuntimeError as exc:
            log.debug("Skip walk-forward fold %s: %s", test_season, exc)
            continue
        if len(split.val_y) < MIN_FOLD_VAL_ROWS:
            continue
        if len(split.test_y) < WF_MIN_TEST_ROWS:
            continue
        _val, fold_test, _tw = _evaluate_candidates(split)
        _accumulate_fold_scores(agg, fold_test)
        folds_used.append(
            {
                "testSeason": int(test_season),
                "n": int(len(split.test_y)),
                "trainSeasons": list(split.split_meta.train_seasons),
                "valSeason": int(split.split_meta.val_season),
            }
        )

    wf_scores = _average_fold_score_rows(agg, len(folds_used))
    if not wf_scores:
        log.warning("Walk-forward folds empty; selecting on validation season only.")
        best = pick_best_score_row(val_scores)
        winner_test = next(s for s in canonical_test_scores if s["model"] == best["model"])
        winner_test = {
            **winner_test,
            "home_win_baseline_accuracy": home_win_baseline_accuracy(
                canonical_split.test_y
            ),
        }
        return ComparisonResult(
            val_scores=val_scores,
            scores=canonical_test_scores,
            tuned_weights=tuned_weights,
            best_model=best["model"],
            best_accuracy=winner_test["accuracy"],
            test_metrics=winner_test,
            selection_criterion="val_log_loss",
            walkforward_scores=[],
            walkforward_folds=[],
        )

    best = pick_best_score_row(wf_scores)
    winner_test = next(s for s in canonical_test_scores if s["model"] == best["model"])
    winner_test = {
        **winner_test,
        "home_win_baseline_accuracy": home_win_baseline_accuracy(
            canonical_split.test_y
        ),
    }
    log.info(
        "Walk-forward selection: %d folds, %d test rows → %s (mean log_loss %.4f).",
        len(folds_used),
        int(best["n"]),
        best["model"],
        best["log_loss"],
    )
    return ComparisonResult(
        val_scores=val_scores,
        scores=canonical_test_scores,
        tuned_weights=tuned_weights,
        best_model=best["model"],
        best_accuracy=winner_test["accuracy"],
        test_metrics=winner_test,
        selection_criterion="walkforward_log_loss",
        walkforward_scores=wf_scores,
        walkforward_folds=folds_used,
    )
