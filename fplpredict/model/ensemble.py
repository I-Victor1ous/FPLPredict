"""Train, compare, and persist ensemble strategies for match prediction."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from fplpredict.config import (
    CANDIDATE_FEATURES,
    MODEL_FEATURES,
    MODEL_RESELECT_DAYS,
)
from fplpredict.config import (
    LEAGUE as DEFAULT_LEAGUE,
)
from fplpredict.data_io import read_csv
from fplpredict.model.feature_selection import forward_select_features
from fplpredict.model.features import build_feature_matrix
from fplpredict.model.calibration import (
    MARKET_IMPLIED_FEATURES,
    apply_probability_calibration,
    modeling_feature_pool,
    tune_temperature,
)
from fplpredict.model.draw_tuning import tune_draw_threshold
from fplpredict.model.market import market_baseline
from fplpredict.model.metrics import (
    brier_score,
    home_win_baseline_accuracy,
    ranked_probability_score,
    reliability_bins,
)
from fplpredict.model.preprocess import feature_frame, make_preprocessor, transform_features
from fplpredict.model.selection import pick_best_score_row
from fplpredict.model.sample_weights import training_sample_weights
from fplpredict.model.splits import (
    SeasonSplitMeta,
    completed_past_matches,
    restrict_last_n_seasons,
    season_holdout_frames,
)
from fplpredict.paths import ARTIFACTS_DIR, league_artifacts_dir

log = logging.getLogger(__name__)

FIXED_WEIGHTS = (4, 4, 1, 1, 1)
MIN_SPLIT_ROWS = 40


def soft_vote_fixed_weights(n_models: int) -> np.ndarray:
    """Legacy (4,4,1,1) weights when there are five bases; equal weights otherwise."""
    base = np.asarray(FIXED_WEIGHTS, dtype=float)
    if n_models == len(base):
        return base
    return np.ones(n_models, dtype=float)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "rf": "Random forest",
    "xgb": "XGBoost",
    "svc": "Linear SVM",
    "lr": "Logistic regression",
    "soft_vote_fixed_4_4_1_1": "Soft-voting ensemble (4,4,1,1)",
    "soft_vote_val_log_loss": "Validation-tuned soft vote",
    "soft_vote_val_accuracy": "Validation-tuned soft vote (accuracy)",
    "stacking_lr_meta_oof": "Stacking ensemble",
    "hgb": "HistGradientBoosting",
}


def model_display_name(strategy: str | None) -> str:
    if not strategy:
        return "Auto-selected model"
    return MODEL_DISPLAY_NAMES.get(strategy, strategy.replace("_", " "))


def _artifact_paths(league: str) -> dict[str, Path]:
    base = league_artifacts_dir(league)
    return {
        "selection": base / "model_selection.json",
        "predictor": base / "predictor.joblib",
        "scaler": base / "scaler.joblib",
        "features": base / "features.json",
    }


def build_base_estimators() -> list[tuple[str, object]]:
    return [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=130,
                min_samples_split=4,
                max_depth=6,
                random_state=42,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                learning_rate=0.1,
                max_depth=4,
                alpha=3,
                n_estimators=200,
                random_state=42,
                verbosity=0,
            ),
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.1,
                max_iter=200,
                random_state=42,
            ),
        ),
        (
            "svc",
            SVC(C=0.2, kernel="linear", probability=True, random_state=42),
        ),
        (
            "lr",
            LogisticRegression(
                C=0.9, solver="saga", max_iter=10_000, random_state=42
            ),
        ),
    ]


class WeightedSoftVote:
    def __init__(self, estimators: list, weights: np.ndarray | list[float]):
        self.estimators = estimators
        self.weights = np.asarray(weights, dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        probas = np.stack(
            [est.predict_proba(X) for est in self.estimators], axis=0
        )
        w = self.weights / self.weights.sum()
        out = np.zeros_like(probas[0])
        for i, wi in enumerate(w):
            out += wi * probas[i]
        out = np.clip(out, 0.0, 1.0)
        row_sums = out.sum(axis=1, keepdims=True)
        return out / np.maximum(row_sums, 1e-15)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


@dataclass
class FittedPredictor:
    """Serializable production model (single, soft-vote, or stacking)."""

    strategy: str
    estimator: Any = None
    estimators: list | None = None
    weights: np.ndarray | None = None
    meta: LogisticRegression | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is not None:
            return np.clip(self.estimator.predict_proba(X), 0.0, 1.0)
        if self.meta is not None and self.estimators is not None:
            stack_x = np.hstack([e.predict_proba(X) for e in self.estimators])
            return np.clip(self.meta.predict_proba(stack_x), 0.0, 1.0)
        if self.estimators is not None and self.weights is not None:
            return WeightedSoftVote(self.estimators, self.weights).predict_proba(X)
        raise RuntimeError(f"Invalid predictor state for {self.strategy}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


@dataclass
class SplitData:
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    train_w: np.ndarray | None
    split_meta: SeasonSplitMeta
    test_start: pd.Timestamp


def _row_keys(df: pd.DataFrame) -> set[tuple]:
    return set(zip(df["Date"], df["Team"], df["Opponent"]))


def assert_disjoint_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_start: pd.Timestamp,
) -> None:
    train_k, val_k, test_k = _row_keys(train_df), _row_keys(val_df), _row_keys(test_df)
    if (train_k & val_k) or (train_k & test_k) or (val_k & test_k):
        raise AssertionError("Train, val, and test splits overlap on fixture keys.")
    if len(val_df) and val_df["Date"].max() >= test_start:
        raise AssertionError("Validation dates must be before the test season.")
    if len(test_df) and test_df["Date"].min() < test_start:
        raise AssertionError("Test dates must lie in the held-out season.")


def load_past_matches(league: str = DEFAULT_LEAGUE) -> pd.DataFrame:
    matches = read_csv("matches.csv")
    next_matches = read_csv("next_matches.csv")
    history = read_csv("history.csv")
    past_matches, _, _ = build_feature_matrix(
        matches, next_matches, history, league=league
    )
    past_matches["Date"] = pd.to_datetime(past_matches["Date"])
    past_matches = completed_past_matches(past_matches)
    return restrict_last_n_seasons(past_matches).sort_values("Date")


def partition_frames(
    past_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, SeasonSplitMeta]:
    train_df, val_df, test_df, meta = season_holdout_frames(past_matches)
    if test_df.empty:
        raise RuntimeError(
            f"No completed matches in test season {meta.test_season}"
        )
    test_start = test_df["Date"].min()
    assert_disjoint_splits(train_df, val_df, test_df, test_start)
    return train_df, val_df, test_df, test_start, meta


def build_eval_split(past_matches: pd.DataFrame, features: list[str]) -> SplitData:
    train_df, val_df, test_df, test_start, meta = partition_frames(past_matches)
    if len(train_df) < MIN_SPLIT_ROWS or len(val_df) < MIN_SPLIT_ROWS:
        raise RuntimeError(
            f"Train/val too small for evaluation (train={len(train_df)}, val={len(val_df)})"
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


def _score(name: str, y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "rps": ranked_probability_score(y_true, proba, sklearn_order=True),
        "n": int(len(y_true)),
    }


def _fit_classifier(est, X, y, sample_weight: np.ndarray | None) -> None:
    if sample_weight is None:
        est.fit(X, y)
    else:
        est.fit(X, y, sample_weight=sample_weight)


def _fit_bases(X, y, sample_weight: np.ndarray | None) -> list:
    estimators = [clone(est) for _, est in build_base_estimators()]
    for est in estimators:
        _fit_classifier(est, X, y, sample_weight)
    return estimators


def _stack_probas(estimators: list, X: np.ndarray) -> np.ndarray:
    return np.stack([e.predict_proba(X) for e in estimators], axis=0)


def _optimize_soft_weights(
    probas: np.ndarray, y: np.ndarray, objective: str = "log_loss"
) -> np.ndarray:
    n_models = probas.shape[0]

    def objective_fn(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        if w.sum() < 1e-12:
            return 1e6
        w = w / w.sum()
        combined = np.tensordot(w, probas, axes=(0, 0))
        combined = np.clip(combined, 1e-15, 1.0 - 1e-15)
        if objective == "accuracy":
            pred = np.argmax(combined, axis=1)
            return 1.0 - accuracy_score(y, pred)
        return log_loss(y, combined, labels=[0, 1, 2])

    x0 = np.ones(n_models) / n_models
    result = minimize(
        objective_fn, x0, method="L-BFGS-B", bounds=[(0.0, None)] * n_models
    )
    w = np.maximum(result.x, 0.0)
    return w / w.sum()


def _fit_stacking(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    bases: list | None = None,
) -> tuple[LogisticRegression, list]:
    """Logistic meta-model on 5-fold out-of-fold base probabilities, plus full-data bases."""
    templates = build_base_estimators()
    n_classes = 3
    oof = np.zeros((len(y), len(templates) * n_classes))
    kf = KFold(n_splits=5, shuffle=False)
    for tr_idx, te_idx in kf.split(X):
        for j, (_, est_t) in enumerate(templates):
            est = clone(est_t)
            w = sample_weight[tr_idx] if sample_weight is not None else None
            _fit_classifier(est, X[tr_idx], y[tr_idx], w)
            oof[te_idx, j * n_classes : (j + 1) * n_classes] = est.predict_proba(X[te_idx])
    meta = LogisticRegression(max_iter=10_000, random_state=42, C=1.0)
    meta.fit(oof, y)
    return meta, bases if bases is not None else _fit_bases(X, y, sample_weight)


@dataclass
class ComparisonResult:
    val_scores: list[dict[str, Any]] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    tuned_weights: dict[str, list[float]] = field(default_factory=dict)
    best_model: str = ""
    best_accuracy: float = 0.0
    test_metrics: dict[str, Any] = field(default_factory=dict)
    selection_criterion: str = "val_log_loss"
    walkforward_scores: list[dict[str, Any]] = field(default_factory=list)
    walkforward_folds: list[dict[str, Any]] = field(default_factory=list)


def _evaluate_candidates(
    split: SplitData,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    """Score every strategy on validation and test slices of one split."""
    bases = _fit_bases(split.train_X, split.train_y, split.train_w)
    val_scores: list[dict[str, Any]] = []
    test_scores: list[dict[str, Any]] = []
    tuned_weights: dict[str, list[float]] = {}

    names = [n for n, _ in build_base_estimators()]
    for i, name in enumerate(names):
        val_scores.append(_score(name, split.val_y, bases[i].predict_proba(split.val_X)))
        test_scores.append(_score(name, split.test_y, bases[i].predict_proba(split.test_X)))

    voter_fixed = WeightedSoftVote(bases, soft_vote_fixed_weights(len(bases)))
    val_scores.append(
        _score(
            "soft_vote_fixed_4_4_1_1",
            split.val_y,
            voter_fixed.predict_proba(split.val_X),
        )
    )
    test_scores.append(
        _score(
            "soft_vote_fixed_4_4_1_1",
            split.test_y,
            voter_fixed.predict_proba(split.test_X),
        )
    )

    val_probas = _stack_probas(bases, split.val_X)
    for objective in ("log_loss", "accuracy"):
        weights = _optimize_soft_weights(val_probas, split.val_y, objective)
        key = f"soft_vote_val_{objective}"
        tuned_weights[key] = weights.tolist()
        voter = WeightedSoftVote(bases, weights)
        val_scores.append(_score(key, split.val_y, voter.predict_proba(split.val_X)))
        test_scores.append(_score(key, split.test_y, voter.predict_proba(split.test_X)))

    stack_meta, _ = _fit_stacking(split.train_X, split.train_y, split.train_w, bases=bases)
    stack = FittedPredictor("stacking_lr_meta_oof", estimators=bases, meta=stack_meta)
    stack_val = stack.predict_proba(split.val_X)
    stack_test = stack.predict_proba(split.test_X)
    val_scores.append(_score("stacking_lr_meta_oof", split.val_y, stack_val))
    test_scores.append(_score("stacking_lr_meta_oof", split.test_y, stack_test))

    return val_scores, test_scores, tuned_weights


def compare_candidates(split: SplitData) -> ComparisonResult:
    """Pick strategy on validation season; report metrics on held-out test season."""
    val_scores, test_scores, tuned_weights = _evaluate_candidates(split)
    best = pick_best_score_row(val_scores)
    winner_test = next(s for s in test_scores if s["model"] == best["model"])
    winner_test = {
        **winner_test,
        "home_win_baseline_accuracy": home_win_baseline_accuracy(split.test_y),
    }
    return ComparisonResult(
        val_scores=val_scores,
        scores=test_scores,
        tuned_weights=tuned_weights,
        best_model=best["model"],
        best_accuracy=winner_test["accuracy"],
        test_metrics=winner_test,
        selection_criterion="val_log_loss",
    )


def _best_val_row(val_scores: list[dict[str, Any]]) -> dict[str, Any]:
    return pick_best_score_row(val_scores)


def pick_best_strategy(comparison: ComparisonResult) -> str:
    return comparison.best_model


def fit_production_predictor(
    strategy: str,
    past_matches: pd.DataFrame,
    features: list[str],
    tuned_weights: dict[str, list[float]],
) -> tuple[FittedPredictor, Pipeline]:
    """Fit the chosen strategy on all completed matches in the modeling window."""
    full_df = restrict_last_n_seasons(completed_past_matches(past_matches)).sort_values(
        "Date"
    )
    y = full_df["Target"].astype(int).to_numpy()
    sample_w = training_sample_weights(y)

    scaler = make_preprocessor()
    X_scaled = scaler.fit_transform(feature_frame(full_df, features))

    if strategy in {"rf", "xgb", "hgb", "svc", "lr"}:
        idx = names_index(strategy)
        est = clone(build_base_estimators()[idx][1])
        _fit_classifier(est, X_scaled, y, sample_w)
        return FittedPredictor(strategy=strategy, estimator=est), scaler

    if strategy == "soft_vote_fixed_4_4_1_1":
        estimators = _fit_bases(X_scaled, y, sample_w)
        return (
            FittedPredictor(
                strategy=strategy,
                estimators=estimators,
                weights=soft_vote_fixed_weights(len(estimators)),
            ),
            scaler,
        )

    if strategy in ("soft_vote_val_log_loss", "soft_vote_val_accuracy"):
        objective = "log_loss" if "log_loss" in strategy else "accuracy"
        weights = tuned_weights.get(strategy)
        if weights is None:
            train_df, val_df, _, _, _ = partition_frames(full_df)
            train_mask = full_df.index.isin(train_df.index)
            val_mask = full_df.index.isin(val_df.index)
            fold_w = sample_w[train_mask] if sample_w is not None else None
            bases = _fit_bases(X_scaled[train_mask], y[train_mask], fold_w)
            val_p = _stack_probas(bases, X_scaled[val_mask])
            weights = _optimize_soft_weights(val_p, y[val_mask], objective).tolist()
        estimators = _fit_bases(X_scaled, y, sample_w)
        return (
            FittedPredictor(
                strategy=strategy,
                estimators=estimators,
                weights=np.array(weights, dtype=float),
            ),
            scaler,
        )

    if strategy == "stacking_lr_meta_oof":
        meta, estimators = _fit_stacking(X_scaled, y, sample_w)
        return (
            FittedPredictor(strategy=strategy, estimators=estimators, meta=meta),
            scaler,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def names_index(name: str) -> int:
    for i, (n, _) in enumerate(build_base_estimators()):
        if n == name:
            return i
    raise ValueError(name)


def load_selection_meta(league: str = DEFAULT_LEAGUE) -> dict | None:
    path = _artifact_paths(league)["selection"]
    if not path.exists():
        legacy = ARTIFACTS_DIR / "model_selection.json"
        if league == DEFAULT_LEAGUE and legacy.exists():
            return json.loads(legacy.read_text(encoding="utf-8"))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_meta_timestamp(meta: dict | None, field: str) -> datetime | None:
    if not meta:
        return None
    raw = meta.get(field)
    if not raw:
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def selection_is_stale(meta: dict | None, reselect_days: int = MODEL_RESELECT_DAYS) -> bool:
    """True when strategy comparison should run again (default: monthly)."""
    if meta is None:
        return True
    ts = _parse_meta_timestamp(meta, "evaluated_at")
    if ts is None:
        return True
    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
    return age_days >= reselect_days


_SELECTION_NOTE = (
    "Strategy chosen on mean walk-forward log loss across completed test seasons; "
    "within FPL_WF_LOG_LOSS_TIE_EPS, higher accuracy then simpler model wins. "
    "Soft-vote weights tuned on the current validation season. "
    "Reported test metrics are for the held-out test season (completed matches only). "
    "The winning strategy is re-compared about monthly; production weights are "
    "refit on the latest scraped data every weekly run. "
    "Test-season metrics refresh weekly from the same strategy refit on the seasons "
    "before the test season, so they stay out-of-sample."
)


def save_artifacts(
    predictor: FittedPredictor,
    scaler: Pipeline,
    features: list[str],
    comparison: ComparisonResult,
    league: str,
    split_meta: SeasonSplitMeta,
    feature_selection: list[dict] | None = None,
    *,
    decision_thresholds: dict[str, float] | None = None,
    calibration: dict[str, float] | None = None,
) -> None:
    paths = _artifact_paths(league)
    paths["predictor"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, paths["predictor"])
    joblib.dump(scaler, paths["scaler"])
    paths["features"].write_text(json.dumps(features, indent=2), encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "league": league,
        "selected_model": comparison.best_model,
        "best_accuracy": comparison.best_accuracy,
        "evaluated_at": now,
        "fitted_at": now,
        "selection_note": _SELECTION_NOTE,
        "best_val_log_loss": _best_val_row(comparison.val_scores)["log_loss"],
        "split": {
            "data_seasons": list(split_meta.all_seasons),
            "train_seasons": list(split_meta.train_seasons),
            "val_season": split_meta.val_season,
            "test_season": split_meta.test_season,
        },
        "val_scores": comparison.val_scores,
        "scores": comparison.scores,
        "test_metrics": comparison.test_metrics,
        "tuned_weights": comparison.tuned_weights,
        "reselect_days": MODEL_RESELECT_DAYS,
        "selected_features": features,
        "feature_selection": feature_selection or [],
    }
    meta.update(_comparison_meta_fields(comparison))
    if decision_thresholds is not None:
        meta["decision_thresholds"] = decision_thresholds
    if calibration is not None:
        meta["calibration"] = calibration
    paths["selection"].write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _comparison_meta_fields(comparison: ComparisonResult) -> dict[str, Any]:
    out: dict[str, Any] = {"selection_criterion": comparison.selection_criterion}
    if comparison.walkforward_scores:
        out["walkforward_scores"] = comparison.walkforward_scores
    if comparison.walkforward_folds:
        out["walkforward_folds"] = comparison.walkforward_folds
    return out


def _persist_fitted_joblib(
    league: str,
    predictor: FittedPredictor,
    scaler: Pipeline,
    features: list[str],
) -> None:
    paths = _artifact_paths(league)
    paths["predictor"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, paths["predictor"])
    joblib.dump(scaler, paths["scaler"])
    paths["features"].write_text(json.dumps(features, indent=2), encoding="utf-8")


def _production_training_flags(meta: dict) -> tuple[str, dict[str, list[float]]]:
    strategy = meta.get("selected_model")
    if not strategy:
        raise ValueError("model_selection.json missing selected_model")
    tuned_weights = meta.get("tuned_weights") or {}
    return strategy, tuned_weights


def refresh_test_season_metrics(league: str) -> dict[str, Any] | None:
    """Score the deployed strategy on the test season; update model_selection.json.

    The production model is trained on the test season too, so it is refit here on
    the seasons before the test season to keep the reported metrics out-of-sample.
    """
    meta = load_selection_meta(league)
    if meta is None or not meta.get("selected_model"):
        return None
    modeling = restrict_last_n_seasons(
        completed_past_matches(load_past_matches(league))
    )
    try:
        _, _, test_df, test_start, _ = partition_frames(modeling)
    except RuntimeError:
        return None
    if test_df.empty:
        return None
    strategy, tuned_weights = _production_training_flags(meta)
    features = selected_features(league, modeling.columns)
    pre_test = modeling[modeling["Date"] < test_start]
    predictor, scaler = fit_production_predictor(strategy, pre_test, features, tuned_weights)
    proba = apply_probability_calibration(
        np.clip(
            predictor.predict_proba(transform_features(scaler, test_df, features)),
            0,
            1,
        ),
        meta,
    )
    y = test_df["Target"].astype(int).to_numpy()
    metrics = {
        **_score(strategy, y, proba),
        "home_win_baseline_accuracy": home_win_baseline_accuracy(y),
        "brier": brier_score(y, proba),
        "calibration": reliability_bins(y, proba),
    }
    try:
        market = market_baseline(test_df, proba, league)
    except Exception:
        log.warning("[%s] Bookmaker baseline unavailable.", league, exc_info=True)
        market = None
    if market:
        metrics["market_baseline"] = market
    meta = dict(meta)
    meta["test_metrics"] = metrics
    meta["best_accuracy"] = metrics["accuracy"]
    _artifact_paths(league)["selection"].write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info(
        "[%s] Test season metrics refreshed: n=%d accuracy %.4f.",
        league,
        metrics["n"],
        metrics["accuracy"],
    )
    return metrics


def refit_cached_strategy(
    past_matches: pd.DataFrame,
    features: list[str],
    league: str,
    meta: dict,
) -> tuple[FittedPredictor, Pipeline]:
    """Refit the cached strategy on latest completed matches (weekly)."""
    strategy, tuned_weights = _production_training_flags(meta)
    log.info(
        "[%s] Refitting strategy %s on latest data (strategy last compared %s).",
        league,
        strategy,
        meta.get("evaluated_at"),
    )
    predictor, scaler = fit_production_predictor(
        strategy,
        past_matches,
        features,
        tuned_weights,
    )
    _persist_fitted_joblib(league, predictor, scaler, features)
    meta = dict(meta)
    meta["fitted_at"] = datetime.now(timezone.utc).isoformat()
    meta.setdefault("selection_note", _SELECTION_NOTE)
    _artifact_paths(league)["selection"].write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return predictor, scaler


def select_features_for_league(
    modeling: pd.DataFrame, candidates: list[str], league: str
) -> tuple[list[str], list[dict]]:
    train_df, val_df, _test_df, _start, _meta = partition_frames(modeling)
    selected, history = forward_select_features(train_df, val_df, candidates)
    if not selected:
        selected = [f for f in MODEL_FEATURES if f in modeling.columns]
    log.info("[%s] Selected %d features: %s", league, len(selected), ", ".join(selected))
    return selected, history


def run_strategy_selection(
    past_matches: pd.DataFrame,
    candidates: list[str],
    league: str,
) -> tuple[FittedPredictor, Pipeline, list[str]]:
    """Pick features and strategy on validation; tune temperature and draw threshold."""
    modeling = restrict_last_n_seasons(completed_past_matches(past_matches))
    features, selection_history = select_features_for_league(
        modeling, candidates, league
    )
    split = build_eval_split(modeling, features)
    from fplpredict.model.walkforward_selection import compare_candidates_walkforward

    comparison = compare_candidates_walkforward(modeling, features)
    strategy = pick_best_strategy(comparison)
    best_val = _best_val_row(comparison.val_scores)
    wf_row = next(
        (r for r in comparison.walkforward_scores if r["model"] == strategy),
        None,
    )
    log.info(
        "[%s] Splits — train %s, val %s, test %s | no class reweighting.",
        league,
        split.split_meta.train_seasons,
        split.split_meta.val_season,
        split.split_meta.test_season,
    )
    if comparison.selection_criterion == "walkforward_log_loss" and wf_row:
        log.info(
            "[%s] Selected %s (walk-forward log_loss %.4f on n=%d). "
            "Current test accuracy %.4f, test RPS %.4f.",
            league,
            strategy,
            wf_row["log_loss"],
            wf_row["n"],
            comparison.best_accuracy,
            comparison.test_metrics.get("rps", float("nan")),
        )
    else:
        log.info(
            "[%s] Selected %s (val log_loss %.4f). Test accuracy %.4f, test RPS %.4f.",
            league,
            strategy,
            best_val["log_loss"],
            comparison.best_accuracy,
            comparison.test_metrics.get("rps", float("nan")),
        )

    predictor, scaler = fit_production_predictor(
        strategy,
        past_matches,
        features,
        comparison.tuned_weights,
    )
    _, val_df, _, _, _ = partition_frames(modeling)
    val_proba_raw = np.clip(
        predictor.predict_proba(transform_features(scaler, val_df, features)), 0, 1
    )
    val_y = val_df["Target"].astype(int).to_numpy()
    temperature, temp_loss = tune_temperature(val_y, val_proba_raw)
    val_proba = apply_probability_calibration(
        val_proba_raw, {"calibration": {"temperature": temperature}}
    )
    t_draw, thresh_loss = tune_draw_threshold(val_y, val_proba)

    log.info(
        "[%s] Temperature T=%.3f (val log_loss %.4f); t_draw=%.3f (threshold log_loss %.4f).",
        league,
        temperature,
        temp_loss,
        t_draw,
        thresh_loss,
    )

    save_artifacts(
        predictor,
        scaler,
        features,
        comparison,
        league,
        split.split_meta,
        feature_selection=selection_history,
        decision_thresholds={"t_draw": t_draw},
        calibration={"temperature": temperature},
    )
    return predictor, scaler, features


def load_production_artifacts(
    league: str = DEFAULT_LEAGUE,
) -> tuple[FittedPredictor, Pipeline, list[str]]:
    paths = _artifact_paths(league)
    if not paths["predictor"].exists() or not paths["scaler"].exists():
        if league == DEFAULT_LEAGUE:
            legacy_p = ARTIFACTS_DIR / "predictor.joblib"
            legacy_s = ARTIFACTS_DIR / "scaler.joblib"
            legacy_f = ARTIFACTS_DIR / "features.json"
            if legacy_p.exists() and legacy_s.exists():
                predictor = joblib.load(legacy_p)
                scaler = joblib.load(legacy_s)
                features = json.loads(legacy_f.read_text(encoding="utf-8"))
                return predictor, scaler, features
        raise FileNotFoundError(
            f"Production model artifacts missing for {league}; run model selection."
        )
    predictor = joblib.load(paths["predictor"])
    scaler = joblib.load(paths["scaler"])
    features = json.loads(paths["features"].read_text(encoding="utf-8"))
    return predictor, scaler, features


def selected_features(league: str, available: pd.Index | list[str]) -> list[str]:
    """Features chosen at the last monthly selection, else the fixed fallback list."""
    path = _artifact_paths(league)["features"]
    meta = load_selection_meta(league) or {}
    chosen = meta.get("selected_features")
    if not chosen and path.exists() and meta.get("feature_selection") is not None:
        chosen = json.loads(path.read_text(encoding="utf-8"))
    if not chosen:
        chosen = MODEL_FEATURES
    avail = set(available)
    return [f for f in chosen if f in avail and f not in MARKET_IMPLIED_FEATURES]


def ensure_production_model(
    past_matches: pd.DataFrame,
    league: str = DEFAULT_LEAGUE,
    force_reselect: bool = False,
) -> tuple[FittedPredictor, Pipeline, list[str]]:
    """Monthly: select features + strategy. Weekly: refit the cached choice."""
    candidates = modeling_feature_pool(
        [f for f in CANDIDATE_FEATURES if f in past_matches.columns]
    )
    meta = load_selection_meta(league)
    needs_selection = (
        force_reselect or selection_is_stale(meta) or not (meta or {}).get("selected_features")
    )
    if needs_selection:
        return run_strategy_selection(past_matches, candidates, league)

    features = selected_features(league, past_matches.columns)
    try:
        predictor, scaler = refit_cached_strategy(past_matches, features, league, meta)
    except FileNotFoundError:
        return run_strategy_selection(past_matches, candidates, league)
    return predictor, scaler, features


def run_comparison_report(league: str = DEFAULT_LEAGUE) -> ComparisonResult:
    """Load data, compare candidates, print ranked table (notebook / CLI)."""
    past_matches = load_past_matches(league)
    features = selected_features(league, past_matches.columns)
    split = build_eval_split(past_matches, features)
    comparison = compare_candidates(split)
    df = pd.DataFrame(comparison.scores).sort_values(
        ["log_loss", "accuracy"], ascending=[True, False]
    )
    sm = split.split_meta
    print(
        f"League: {league} | train seasons {sm.train_seasons}, "
        f"val {sm.val_season}, test {sm.test_season} | "
        f"rows train={len(split.train_y)}, val={len(split.val_y)}, "
        f"test={len(split.test_y)} (completed matches only)"
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest on validation season (log loss): {comparison.best_model}")
    print(f"Winner test-season metrics: {comparison.test_metrics}")
    return comparison
