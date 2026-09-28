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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from fplpredict.config import (
    LEAGUE as DEFAULT_LEAGUE,
    MODEL_FEATURES,
    MODEL_RESELECT_DAYS,
)
from fplpredict.data_io import read_csv
from fplpredict.model.features import build_feature_matrix
from fplpredict.model.splits import (
    SeasonSplitMeta,
    completed_past_matches,
    restrict_last_n_seasons,
    season_holdout_frames,
)
from fplpredict.paths import ARTIFACTS_DIR, league_artifacts_dir

log = logging.getLogger(__name__)

FIXED_WEIGHTS = (4, 4, 1, 1)
DRAW_CLASS_WEIGHT = {0: 2, 1: 1, 2: 1}
MIN_SPLIT_ROWS = 40

CANDIDATE_STRATEGIES = (
    "rf",
    "xgb",
    "svc",
    "lr",
    "soft_vote_fixed_4_4_1_1",
    "soft_vote_val_log_loss",
    "soft_vote_val_accuracy",
    "stacking_lr_meta_oof",
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "rf": "Random forest",
    "xgb": "XGBoost",
    "svc": "Linear SVM",
    "lr": "Logistic regression",
    "soft_vote_fixed_4_4_1_1": "Soft-voting ensemble (4,4,1,1)",
    "soft_vote_val_log_loss": "Validation-tuned soft vote",
    "soft_vote_val_accuracy": "Validation-tuned soft vote (accuracy)",
    "stacking_lr_meta_oof": "Stacking ensemble",
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
                eta=0.6,
                max_depth=4,
                alpha=3,
                random_state=42,
                verbosity=0,
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
    train_w: np.ndarray
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


def build_eval_split(
    past_matches: pd.DataFrame,
    features: list[str],
) -> SplitData:
    train_df, val_df, test_df, test_start, meta = partition_frames(past_matches)
    if len(train_df) < MIN_SPLIT_ROWS or len(val_df) < MIN_SPLIT_ROWS:
        raise RuntimeError(
            f"Train/val too small for evaluation (train={len(train_df)}, val={len(val_df)})"
        )

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df[features].fillna(0))
    val_X = scaler.transform(val_df[features].fillna(0))
    test_X = scaler.transform(test_df[features].fillna(0))

    train_y = train_df["Target"].astype(int).to_numpy()
    val_y = val_df["Target"].astype(int).to_numpy()
    test_y = test_df["Target"].astype(int).to_numpy()
    train_w = compute_sample_weight(class_weight=DRAW_CLASS_WEIGHT, y=train_y)

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
    }


def _fit_bases(X, y, sample_weight) -> list:
    estimators = [clone(est) for _, est in build_base_estimators()]
    for est in estimators:
        est.fit(X, y, sample_weight=sample_weight)
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


def _eval_stacking_on_test(split: SplitData) -> np.ndarray:
    templates = build_base_estimators()
    n_models = len(templates)
    n_classes = 3
    n_train = len(split.train_y)
    oof = np.zeros((n_train, n_models * n_classes))
    kf = KFold(n_splits=5, shuffle=False)

    for tr_idx, te_idx in kf.split(split.train_X):
        for j, (_, est_t) in enumerate(templates):
            est = clone(est_t)
            est.fit(
                split.train_X[tr_idx],
                split.train_y[tr_idx],
                sample_weight=split.train_w[tr_idx],
            )
            oof[te_idx, j * n_classes : (j + 1) * n_classes] = est.predict_proba(
                split.train_X[te_idx]
            )

    meta = LogisticRegression(max_iter=10_000, random_state=42, C=1.0)
    meta.fit(oof, split.train_y)

    fitted = _fit_bases(split.train_X, split.train_y, split.train_w)
    test_meta_x = np.hstack([e.predict_proba(split.test_X) for e in fitted])
    return meta.predict_proba(test_meta_x)


@dataclass
class ComparisonResult:
    scores: list[dict[str, Any]] = field(default_factory=list)
    tuned_weights: dict[str, list[float]] = field(default_factory=dict)
    best_model: str = ""
    best_accuracy: float = 0.0


def compare_candidates(split: SplitData) -> ComparisonResult:
    """Evaluate all strategies on the held-out test season (bases fit on train only)."""
    bases = _fit_bases(split.train_X, split.train_y, split.train_w)
    scores: list[dict[str, Any]] = []
    tuned_weights: dict[str, list[float]] = {}

    names = [n for n, _ in build_base_estimators()]
    for i, name in enumerate(names):
        proba = bases[i].predict_proba(split.test_X)
        scores.append(_score(name, split.test_y, proba))

    voter_fixed = WeightedSoftVote(bases, FIXED_WEIGHTS)
    scores.append(
        _score("soft_vote_fixed_4_4_1_1", split.test_y, voter_fixed.predict_proba(split.test_X))
    )

    val_probas = _stack_probas(bases, split.val_X)
    for objective in ("log_loss", "accuracy"):
        weights = _optimize_soft_weights(val_probas, split.val_y, objective)
        key = f"soft_vote_val_{objective}"
        tuned_weights[key] = weights.tolist()
        voter = WeightedSoftVote(bases, weights)
        scores.append(_score(key, split.test_y, voter.predict_proba(split.test_X)))

    stack_proba = _eval_stacking_on_test(split)
    scores.append(_score("stacking_lr_meta_oof", split.test_y, stack_proba))

    best = max(scores, key=lambda r: (r["accuracy"], -r["log_loss"]))
    return ComparisonResult(
        scores=scores,
        tuned_weights=tuned_weights,
        best_model=best["model"],
        best_accuracy=best["accuracy"],
    )


def pick_best_strategy(comparison: ComparisonResult) -> str:
    return comparison.best_model


def fit_production_predictor(
    strategy: str,
    past_matches: pd.DataFrame,
    features: list[str],
    tuned_weights: dict[str, list[float]],
) -> tuple[FittedPredictor, StandardScaler]:
    """Fit the chosen strategy on all completed matches in the modeling window."""
    full_df = restrict_last_n_seasons(completed_past_matches(past_matches)).sort_values(
        "Date"
    )
    X = full_df[features].fillna(0)
    y = full_df["Target"].astype(int).to_numpy()
    sample_w = compute_sample_weight(class_weight=DRAW_CLASS_WEIGHT, y=y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_df, val_df, _, _, _ = partition_frames(full_df)
    train_mask = full_df.index.isin(train_df.index)
    val_mask = full_df.index.isin(val_df.index)
    train_X = X_scaled[train_mask]
    train_y = y[train_mask]
    train_w = sample_w[train_mask]
    val_X = X_scaled[val_mask]
    val_y = y[val_mask]

    if strategy in {"rf", "xgb", "svc", "lr"}:
        idx = names_index(strategy)
        est = clone(build_base_estimators()[idx][1])
        est.fit(X_scaled, y, sample_weight=sample_w)
        return FittedPredictor(strategy=strategy, estimator=est), scaler

    if strategy == "soft_vote_fixed_4_4_1_1":
        estimators = _fit_bases(X_scaled, y, sample_w)
        return (
            FittedPredictor(
                strategy=strategy,
                estimators=estimators,
                weights=np.array(FIXED_WEIGHTS, dtype=float),
            ),
            scaler,
        )

    if strategy in ("soft_vote_val_log_loss", "soft_vote_val_accuracy"):
        bases = _fit_bases(train_X, train_y, train_w)
        objective = "log_loss" if "log_loss" in strategy else "accuracy"
        weights = tuned_weights.get(strategy)
        if weights is None:
            val_p = _stack_probas(bases, val_X)
            weights = _optimize_soft_weights(val_p, val_y, objective).tolist()
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
        templates = build_base_estimators()
        n_models = len(templates)
        n_classes = 3
        oof = np.zeros((len(y), n_models * n_classes))
        kf = KFold(n_splits=5, shuffle=False)
        for tr_idx, te_idx in kf.split(X_scaled):
            for j, (_, est_t) in enumerate(templates):
                est = clone(est_t)
                est.fit(X_scaled[tr_idx], y[tr_idx], sample_weight=sample_w[tr_idx])
                oof[te_idx, j * n_classes : (j + 1) * n_classes] = est.predict_proba(
                    X_scaled[te_idx]
                )
        meta = LogisticRegression(max_iter=10_000, random_state=42, C=1.0)
        meta.fit(oof, y)
        estimators = _fit_bases(X_scaled, y, sample_w)
        return (
            FittedPredictor(
                strategy=strategy, estimators=estimators, meta=meta
            ),
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


def selection_is_stale(meta: dict | None, reselect_days: int = MODEL_RESELECT_DAYS) -> bool:
    if meta is None:
        return True
    evaluated = meta.get("evaluated_at")
    if not evaluated:
        return True
    ts = datetime.fromisoformat(evaluated.replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
    return age_days >= reselect_days


def save_artifacts(
    predictor: FittedPredictor,
    scaler: StandardScaler,
    features: list[str],
    comparison: ComparisonResult,
    league: str,
    split_meta: SeasonSplitMeta,
) -> None:
    paths = _artifact_paths(league)
    paths["predictor"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, paths["predictor"])
    joblib.dump(scaler, paths["scaler"])
    paths["features"].write_text(json.dumps(features, indent=2), encoding="utf-8")

    meta = {
        "league": league,
        "selected_model": comparison.best_model,
        "best_accuracy": comparison.best_accuracy,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": {
            "data_seasons": list(split_meta.all_seasons),
            "train_seasons": list(split_meta.train_seasons),
            "val_season": split_meta.val_season,
            "test_season": split_meta.test_season,
        },
        "scores": comparison.scores,
        "tuned_weights": comparison.tuned_weights,
        "reselect_days": MODEL_RESELECT_DAYS,
    }
    paths["selection"].write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_production_artifacts(
    league: str = DEFAULT_LEAGUE,
) -> tuple[FittedPredictor, StandardScaler, list[str]]:
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


def ensure_production_model(
    past_matches: pd.DataFrame,
    features: list[str],
    league: str = DEFAULT_LEAGUE,
    force_reselect: bool = False,
) -> tuple[FittedPredictor, StandardScaler]:
    meta = load_selection_meta(league)
    if not force_reselect and not selection_is_stale(meta):
        log.info(
            "[%s] Using cached model %s (evaluated %s).",
            league,
            meta.get("selected_model"),
            meta.get("evaluated_at"),
        )
        predictor, scaler, _ = load_production_artifacts(league)
        return predictor, scaler

    modeling = restrict_last_n_seasons(completed_past_matches(past_matches))
    split = build_eval_split(modeling, features)
    log.info(
        "[%s] Model comparison — train seasons %s, val %s, test %s (completed only)…",
        league,
        split.split_meta.train_seasons,
        split.split_meta.val_season,
        split.split_meta.test_season,
    )
    comparison = compare_candidates(split)
    strategy = pick_best_strategy(comparison)
    log.info(
        "[%s] Selected %s (test accuracy %.4f). Scores: %s",
        league,
        strategy,
        comparison.best_accuracy,
        {s["model"]: round(s["accuracy"], 4) for s in comparison.scores},
    )

    predictor, scaler = fit_production_predictor(
        strategy, past_matches, features, comparison.tuned_weights
    )
    save_artifacts(
        predictor, scaler, features, comparison, league, split.split_meta
    )
    return predictor, scaler


def run_comparison_report(league: str = DEFAULT_LEAGUE) -> ComparisonResult:
    """Load data, compare candidates, print ranked table (notebook / CLI)."""
    past_matches = load_past_matches(league)
    features = [f for f in MODEL_FEATURES if f in past_matches.columns]
    split = build_eval_split(past_matches, features)
    comparison = compare_candidates(split)
    df = pd.DataFrame(comparison.scores).sort_values(
        ["accuracy", "log_loss"], ascending=[False, True]
    )
    sm = split.split_meta
    print(
        f"League: {league} | train seasons {sm.train_seasons}, "
        f"val {sm.val_season}, test {sm.test_season} | "
        f"rows train={len(split.train_y)}, val={len(split.val_y)}, "
        f"test={len(split.test_y)} (completed matches only)"
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest on test season: {comparison.best_model}")
    return comparison
